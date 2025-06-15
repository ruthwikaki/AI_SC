"""
Configuration settings for the AI Supply Chain application
"""

from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with all environment variables"""
    
    # Application info - UPPERCASE (from env)
    APP_NAME: str = Field(default="AI Supply Chain")
    VERSION: str = Field(default="1.0.0")
    DEBUG: bool = Field(default=False)
    ENVIRONMENT: str = Field(default="development")
    
    # API Documentation - UPPERCASE
    DOCS_URL: str = Field(default="/docs")
    REDOC_URL: str = Field(default="/redoc")
    OPENAPI_URL: str = Field(default="/openapi.json")
    
    # Database Configuration - UPPERCASE
    DATABASE_URL: str = Field(
        default="postgresql://postgres:123456789@localhost:5432/Supplychain_AI"
    )
    DB_NAME: str = Field(default="Supplychain_AI")
    DB_USER: str = Field(default="postgres")
    DB_PASSWORD: str = Field(default="123456789")
    DB_HOST: str = Field(default="localhost")
    DB_PORT: str = Field(default="5432")
    DATABASE_ECHO: bool = Field(default=False)
    
    # Security & JWT - UPPERCASE
    SECRET_KEY: str = Field(
        default="your-secret-key-here-change-in-production"
    )
    JWT_SECRET_KEY: str = Field(
        default="your-secret-key-change-this-in-production"
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION_MINUTES: int = Field(default=1440)
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    
    # CORS settings - UPPERCASE
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"]
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])
    
    # Redis (optional) - UPPERCASE
    REDIS_URL: Optional[str] = Field(default=None)
    
    # LLM Configuration - UPPERCASE
    LLM_MODEL: str = Field(default="deepseek-coder-v2:16b-lite-instruct-q4_0")
    LLM_API_URL: str = Field(default="http://localhost:11434")
    LLM_CONFIG_ADMIN_HASH: str = Field(
        default="5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"
    )
    
    # Lowercase aliases for compatibility with server.py
    app_name: str = Field(default="AI Supply Chain", alias="APP_NAME")
    version: str = Field(default="1.0.0", alias="VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    docs_url: str = Field(default="/docs", alias="DOCS_URL")
    redoc_url: str = Field(default="/redoc", alias="REDOC_URL")
    openapi_url: str = Field(default="/openapi.json", alias="OPENAPI_URL")
    database_url: str = Field(default="postgresql://postgres:123456789@localhost:5432/Supplychain_AI", alias="DATABASE_URL")
    database_echo: bool = Field(default=False, alias="DATABASE_ECHO")
    secret_key: str = Field(default="your-secret-key-here-change-in-production", alias="SECRET_KEY")
    algorithm: str = Field(default="HS256", alias="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:8000"], alias="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], alias="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], alias="CORS_ALLOW_HEADERS")
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")

    
    # File upload and misc settings

    
    upload_dir: str = Field(default="./uploads")

    
    UPLOAD_DIR: str = Field(default="./uploads")

    
    max_upload_size: int = Field(default=10485760)  # 10MB

    
    MAX_UPLOAD_SIZE: int = Field(default=10485760)



    
    # Additional settings from server.py



    
    llm_provider: str = Field(default="")



    
    LLM_PROVIDER: str = Field(default="")




    
    class Config:
        env_file = ".env"
        case_sensitive = True
        # Allow extra fields
        extra = "allow"
        # Allow population by field name (for aliases)
        populate_by_name = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Create a global settings instance
settings = get_settings()
