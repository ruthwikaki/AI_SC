"""
Application configuration using Pydantic v2
"""

import os
from typing import List, Optional, Union, Any
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    APP_NAME: str = Field(default="AI Supply Chain", validation_alias="APP_NAME")
    APP_VERSION: str = Field(default="1.0.0", validation_alias="APP_VERSION")
    DEBUG: bool = Field(default=True, validation_alias="DEBUG")
    ENVIRONMENT: str = Field(default="development", validation_alias="ENVIRONMENT")
    
    # API
    API_V1_STR: str = Field(default="/api/v1", validation_alias="API_V1_STR")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-here-change-in-production", validation_alias="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", validation_alias="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./app.db", validation_alias="DATABASE_URL")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"],
        validation_alias="BACKEND_CORS_ORIGINS"
    )
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="llama2", validation_alias="OLLAMA_MODEL")
    
    # Server
    HOST: str = Field(default="0.0.0.0", validation_alias="HOST")
    PORT: int = Field(default=8000, validation_alias="PORT")
    RELOAD: bool = Field(default=True, validation_alias="RELOAD")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    
    # Project root
    PROJECT_ROOT: str = Field(default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), validation_alias="PROJECT_ROOT")
    
    # Additional settings that might be expected
    FIRST_SUPERUSER: Optional[str] = Field(default=None, validation_alias="FIRST_SUPERUSER")
    FIRST_SUPERUSER_PASSWORD: Optional[str] = Field(default=None, validation_alias="FIRST_SUPERUSER_PASSWORD")
    
    # Redis/Cache settings
    REDIS_URL: Optional[str] = Field(default=None, validation_alias="REDIS_URL")
    CACHE_ENABLED: bool = Field(default=True, validation_alias="CACHE_ENABLED")
    CACHE_TTL: int = Field(default=300, validation_alias="CACHE_TTL")  # seconds
    
    # Email settings
    SMTP_TLS: bool = Field(default=True, validation_alias="SMTP_TLS")
    SMTP_PORT: Optional[int] = Field(default=None, validation_alias="SMTP_PORT")
    SMTP_HOST: Optional[str] = Field(default=None, validation_alias="SMTP_HOST")
    SMTP_USER: Optional[str] = Field(default=None, validation_alias="SMTP_USER")
    SMTP_PASSWORD: Optional[str] = Field(default=None, validation_alias="SMTP_PASSWORD")
    EMAILS_FROM_EMAIL: Optional[str] = Field(default=None, validation_alias="EMAILS_FROM_EMAIL")
    EMAILS_FROM_NAME: Optional[str] = Field(default=None, validation_alias="EMAILS_FROM_NAME")
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = Field(default=20, validation_alias="DEFAULT_PAGE_SIZE")
    MAX_PAGE_SIZE: int = Field(default=100, validation_alias="MAX_PAGE_SIZE")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v:
            return "sqlite:///./app.db"
        return v


@lru_cache()
def get_settings() -> Settings:
    """
    Create and cache settings instance.
    Use lru_cache to ensure we only create one instance.
    """
    return Settings()


# Create a settings instance for backward compatibility
settings = get_settings()


# Additional helper functions that might be used
def get_project_root() -> str:
    """Get the project root directory"""
    return settings.PROJECT_ROOT


def get_database_url() -> str:
    """Get the database URL"""
    return settings.DATABASE_URL


def is_debug() -> bool:
    """Check if we're in debug mode"""
    return settings.DEBUG


def get_cors_origins() -> List[str]:
    """Get CORS origins"""
    return settings.BACKEND_CORS_ORIGINS