"""
Configuration loader.

This module provides functions for loading and accessing application settings.
"""

import os
from typing import List, Dict, Any, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from functools import lru_cache
import json
import dotenv

# Load environment variables from .env file if present
dotenv.load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # General settings
    TIMEZONE: str = "UTC"
    app_name: str = "Supply Chain LLM API"
    api_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    enable_scheduler: bool = Field(default=True, description='Enable job scheduler')
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    uvicorn_workers: int = 1
    allowed_hosts: List[str] = ["*"]
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:3002",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002"
    ]
    
    # Database settings - FIXED: Using Field with validation_alias for env mapping
    database_url: str = Field(
        default="postgresql://postgres:123456789@localhost:5432/supplychain_AI",
        validation_alias="DATABASE_URL"
    )
    database_pool_size: int = 5
    database_pool_overflow: int = 10
    
    # Authentication settings
    jwt_secret_key: str = Field(
        default="CHANGE_THIS_IN_PRODUCTION",
        validation_alias="JWT_SECRET_KEY"
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Encryption settings
    encryption_key: Optional[str] = Field(
        default=None,
        validation_alias="ENCRYPTION_KEY"
    )
    
    # Rate limiting settings
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    token_limit_count: int = 10000
    token_limit_window: int = 3600  # seconds
    
    # LLM settings
    llm_api_key: Optional[str] = Field(
        default=None,
        validation_alias="LLM_API_KEY"
    )
    llm_api_base: Optional[str] = Field(
        default=None,
        validation_alias="LLM_API_BASE"
    )
    default_model: str = "mistral-medium"
    active_model: Optional[str] = None
    model_config_path: str = "app/llm/config"
    llama3_model_path: Optional[str] = Field(
        default=None,
        validation_alias="LLAMA3_MODEL_PATH"
    )
    
    # LLM health check settings
    llm_health_check_interval: int = 300  # seconds
    llm_max_health_check_latency: float = 5.0  # seconds
    llm_max_consecutive_failures: int = 3
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = False
    log_directory: str = "logs"
    
    # Template settings
    templates_dir: str = "app/llm/prompt/templates"
    
    # Cache settings (updated for new ResultCache)
    cache_type: str = "memory"  # memory, redis (kept for backward compatibility)
    cache_redis_url: Optional[str] = None  # deprecated
    cache_ttl: int = 300  # 5 minutes default
    query_cache_ttl: int = 3600  # 1 hour for query results
    
    # Result Cache specific settings (NEW)
    result_cache_size: int = Field(
        default=200,  # MB
        validation_alias="RESULT_CACHE_SIZE"
    )
    result_cache_ttl: int = Field(
        default=300,  # seconds (5 minutes)
        validation_alias="RESULT_CACHE_TTL"
    )
    max_result_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB in bytes
        validation_alias="MAX_RESULT_SIZE"
    )
    result_cache_dir: Optional[str] = Field(
        default="./cache/results",
        validation_alias="RESULT_CACHE_DIR"
    )
    compress_result_cache: bool = Field(
        default=True,
        validation_alias="COMPRESS_RESULT_CACHE"
    )
    
    # Redis settings (deprecated - kept for backward compatibility)
    redis_host: str = Field(
        default="localhost",
        validation_alias="REDIS_HOST"
    )
    redis_port: int = Field(
        default=6379,
        validation_alias="REDIS_PORT"
    )
    redis_db: int = Field(
        default=0,
        validation_alias="REDIS_DB"
    )
    
    # Analytics settings
    max_forecast_periods: int = Field(
        default=24,
        validation_alias="MAX_FORECAST_PERIODS"
    )
    default_confidence_level: float = Field(
        default=0.95,
        validation_alias="DEFAULT_CONFIDENCE_LEVEL"
    )
    
    # Export settings
    export_directory: str = Field(
        default="./exports",
        validation_alias="EXPORT_DIRECTORY"
    )
    max_export_size_mb: int = Field(
        default=100,
        validation_alias="MAX_EXPORT_SIZE_MB"
    )
    
    # Report settings
    report_directory: str = Field(
        default="./reports",
        validation_alias="REPORT_DIRECTORY"
    )
    report_retention_days: int = Field(
        default=30,
        validation_alias="REPORT_RETENTION_DAYS"
    )
    
    # Client settings
    default_client_id: Optional[str] = None
    admin_db_client_id: str = "admin"
    
    # File storage settings
    storage_provider: str = "local"  # local, s3
    storage_local_path: str = "storage"
    storage_s3_bucket: Optional[str] = None
    storage_s3_region: Optional[str] = None
    
    # Email settings
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_sender: str = "noreply@example.com"
    
    # FIXED: Using Pydantic v2 ConfigDict
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # This prevents "extra inputs are not permitted" error
    )
    
    # FIXED: Updated validators to Pydantic v2 syntax
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed = ["development", "testing", "staging", "production"]
        if v.lower() not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v.lower()
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level setting."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()
    
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v, info):
        """Validate database URL based on environment."""
        if info.data.get("environment") == "production" and "localhost" in v:
            raise ValueError("Production environment should not use localhost database")
        return v
    
    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v, info):
        """Validate JWT secret key."""
        if info.data.get("environment") == "production" and v == "CHANGE_THIS_IN_PRODUCTION":
            raise ValueError("Production environment requires a secure JWT secret key")
        return v
    
    @field_validator("default_confidence_level")
    @classmethod
    def validate_confidence_level(cls, v):
        """Validate confidence level is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("default_confidence_level must be between 0 and 1")
        return v
    
    @field_validator("max_export_size_mb")
    @classmethod
    def validate_export_size(cls, v):
        """Validate export size is positive."""
        if v <= 0:
            raise ValueError("max_export_size_mb must be positive")
        return v
    
    @field_validator("result_cache_size")
    @classmethod
    def validate_cache_size(cls, v):
        """Validate cache size is positive."""
        if v <= 0:
            raise ValueError("result_cache_size must be positive")
        return v
    
    @field_validator("result_cache_ttl")
    @classmethod
    def validate_cache_ttl(cls, v):
        """Validate cache TTL is positive."""
        if v <= 0:
            raise ValueError("result_cache_ttl must be positive")
        return v
    
    @field_validator("max_result_size")
    @classmethod
    def validate_result_size(cls, v):
        """Validate max result size is positive."""
        if v <= 0:
            raise ValueError("max_result_size must be positive")
        return v

@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings.
    
    This function uses lru_cache to cache the settings object
    for improved performance.
    
    Returns:
        Settings object
    """
    return Settings()

def load_config_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        print(f"Error loading config file {file_path}: {e}")
        return {}

def save_config_file(file_path: str, config: Dict[str, Any]) -> bool:
    """
    Save a configuration dictionary to a JSON file.
    
    Args:
        file_path: Path to save configuration file
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save file
        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving config file {file_path}: {e}")
        return False

def get_environment_variables(prefix: str = "APP_") -> Dict[str, str]:
    """
    Get all environment variables with a specified prefix.
    
    Args:
        prefix: Prefix to filter environment variables
        
    Returns:
        Dictionary of environment variables
    """
    return {
        k[len(prefix):]: v
        for k, v in os.environ.items()
        if k.startswith(prefix)
    }