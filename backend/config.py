"""
Application configuration using Pydantic Settings
Located at: /backend/app/config.py
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from pydantic_settings import SettingsConfigDict
import os
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = Field(default="Supply Chain AI", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database - Note: using lowercase for consistency
    database_url: str = Field(
        default="postgresql://postgres:123456789@localhost:5432/Supplychain_AI",
        env="DATABASE_URL"
    )
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-here-change-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    # Redis Cache
    redis_url: Optional[str] = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    cache_ttl: int = Field(default=300, env="CACHE_TTL")
    
    # LLM Configuration
    llm_provider: str = Field(default="ollama", env="LLM_PROVIDER")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama2", env="OLLAMA_MODEL")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # External Database Connections
    mysql_url: Optional[str] = Field(default=None, env="MYSQL_URL")
    oracle_url: Optional[str] = Field(default=None, env="ORACLE_URL")
    sqlserver_url: Optional[str] = Field(default=None, env="SQLSERVER_URL")
    
    # File Storage
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    max_upload_size: int = Field(default=10485760, env="MAX_UPLOAD_SIZE")  # 10MB
    
    # Analytics
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    analytics_batch_size: int = Field(default=100, env="ANALYTICS_BATCH_SIZE")
    
    # Multi-tier settings
    max_tier_depth: int = Field(default=5, env="MAX_TIER_DEPTH")
    risk_calculation_timeout: int = Field(default=30, env="RISK_CALCULATION_TIMEOUT")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @validator("cors_origins", pre=True)
    def parse_cors(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("database_url")
    def validate_database_url(cls, v):
        if not v:
            raise ValueError("DATABASE_URL must be set")
        
        return v
    
    @validator("secret_key")
    def validate_secret_key(cls, v, values):
        if values.get("environment") == "production" and v == "your-secret-key-here-change-in-production":
            raise ValueError("You must set a secure SECRET_KEY in production")
        return v
    
    def get_db_settings(self) -> Dict[str, Any]:
        """Get database-specific settings"""
        return {
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "echo": self.database_echo,
            "pool_pre_ping": True,
            "pool_recycle": 3600
        }

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Create a single settings instance
settings = get_settings()