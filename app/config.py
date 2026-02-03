"""
Configuration settings for the Multi-Agent Stock Analysis System
"""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Grok API Configuration
    grok_api_key: str = ""
    grok_api_base: str = "https://api.x.ai/v1"
    grok_model: str = "grok-beta"
    
    # Application Settings
    debug: bool = False
    log_level: str = "INFO"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60
    
    # Agent Configuration
    max_agent_iterations: int = 10
    agent_timeout: int = 120
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()


# Convenience access to settings
settings = get_settings()

