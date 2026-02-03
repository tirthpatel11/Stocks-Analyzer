"""
Base Agent class and LLM configuration for the Multi-Agent System
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def get_grok_llm(temperature: float = 0.1, max_tokens: int = 4096) -> ChatOpenAI:
    """
    Create a Grok LLM instance using OpenAI-compatible API
    
    Args:
        temperature: Sampling temperature for responses
        max_tokens: Maximum tokens in response
        
    Returns:
        ChatOpenAI instance configured for Grok
    """
    return ChatOpenAI(
        model=settings.grok_model,
        openai_api_key=settings.grok_api_key,
        openai_api_base=settings.grok_api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class AgentState(BaseModel):
    """Base state model for agents"""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    current_agent: str = ""
    task: str = ""
    ticker: str = ""
    analysis_results: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    completed: bool = False


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents
    
    Each agent has:
    - A name and role description
    - Access to the Grok LLM
    - A system prompt defining its behavior
    - Tools specific to its function
    """
    
    def __init__(
        self, 
        name: str, 
        description: str,
        temperature: float = 0.1
    ):
        self.name = name
        self.description = description
        self.llm = get_grok_llm(temperature=temperature)
        self.tools = self._setup_tools()
        self.system_prompt = self._create_system_prompt()
        
    @abstractmethod
    def _setup_tools(self) -> List[Any]:
        """Set up agent-specific tools"""
        pass
    
    @abstractmethod
    def _create_system_prompt(self) -> str:
        """Create the agent's system prompt"""
        pass
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary function"""
        pass
    
    def _format_messages(self, state: Dict[str, Any]) -> List[BaseMessage]:
        """Convert state messages to LangChain message format"""
        messages = [SystemMessage(content=self.system_prompt)]
        
        for msg in state.get("messages", []):
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))
                
        return messages
    
    async def _invoke_llm(self, messages: List[BaseMessage]) -> str:
        """Invoke the LLM with messages"""
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM invocation error in {self.name}: {e}")
            raise

