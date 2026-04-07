import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def normalize_ticker(ticker: str) -> str:
    if not ticker:
        return ticker
    ticker = ticker.strip().upper()
    if not ticker.endswith(('.NS', '.BO')):
        ticker = f"{ticker}.NS"
    return ticker


def _model_name_suggests_groq(model: str) -> bool:
    m = (model or "").lower()
    return any(x in m for x in ("llama", "mixtral", "gemma", "qwen"))


def get_grok_llm(temperature: float = 0.1, max_tokens: int = 4096) -> ChatOpenAI:
    groq_key = (settings.groq_api_key or "").strip()
    grok_key = (settings.grok_api_key or "").strip()

    if groq_key:
        return ChatOpenAI(
            model=settings.groq_model,
            openai_api_key=groq_key,
            openai_api_base=settings.groq_api_base.rstrip("/"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if grok_key and _model_name_suggests_groq(settings.grok_model):
        logger.warning(
            "GROK_MODEL looks like a Groq model; using Groq API (%s). "
            "Prefer GROQ_API_KEY + GROQ_MODEL in .env.",
            settings.groq_api_base,
        )
        return ChatOpenAI(
            model=settings.grok_model,
            openai_api_key=grok_key,
            openai_api_base=settings.groq_api_base.rstrip("/"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return ChatOpenAI(
        model=settings.grok_model,
        openai_api_key=grok_key,
        openai_api_base=settings.grok_api_base.rstrip("/"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


class AgentState(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    current_agent: str = ""
    task: str = ""
    ticker: str = ""
    analysis_results: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    completed: bool = False


class BaseAgent(ABC):
    def __init__(self, name: str, description: str, temperature: float = 0.1):
        self.name = name
        self.description = description
        self.llm = get_grok_llm(temperature=temperature)
        self.tools = self._setup_tools()
        self.system_prompt = self._create_system_prompt()

    @abstractmethod
    def _setup_tools(self) -> List[Any]:
        pass

    @abstractmethod
    def _create_system_prompt(self) -> str:
        pass

    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def _format_messages(self, state: Dict[str, Any]) -> List[BaseMessage]:
        messages = [SystemMessage(content=self.system_prompt)]
        for msg in state.get("messages", []):
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))
        return messages

    async def _invoke_llm(self, messages: List[BaseMessage]) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(4):
            try:
                response = await self.llm.ainvoke(messages)
                return response.content if response.content is not None else ""
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                rate_limited = (
                    "429" in str(e)
                    or "rate" in msg
                    or "too many requests" in msg
                )
                transient = rate_limited or "connection" in msg or "timeout" in msg
                if transient and attempt < 3:
                    wait = 12 * (attempt + 1)
                    logger.warning(
                        "LLM call failed (%s), attempt %s/4 — waiting %ss: %s",
                        self.name,
                        attempt + 1,
                        wait,
                        e,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"LLM invocation error in {self.name}: {e}")
                raise
        if last_err:
            raise last_err
        raise RuntimeError("LLM invocation failed with no error detail")
