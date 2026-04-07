import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    grok_api_key: str = ""
    grok_api_base: str = "https://api.x.ai/v1"
    grok_model: str = "grok-3-mini"

    groq_api_key: str = ""
    groq_api_base: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"

    debug: bool = False
    log_level: str = "INFO"

    rate_limit_requests: int = 100
    rate_limit_period: int = 60

    max_agent_iterations: int = 10
    agent_timeout: int = 120

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
