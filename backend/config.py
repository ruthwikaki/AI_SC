"""
Configuration loader.

This module provides functions for loading and accessing application settings.
"""

import os
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseSettings, Field, validator
from functools import lru_cache
import json
import dotenv

# Load environment variables from .env file if present
dotenv.load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # General settings
    app_name: str = "Supply Chain LLM API"
    api_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    uvicorn_workers: int = 1
    allowed_hosts: List[str] = ["*"]
    cors_origins: List[str] = ["*"]
    
    # Database settings
    database_url: str = "postgresql://postgres:postgres@localhost:5432/supply_chain"
    database_pool_size: int = 5
    database_pool_overflow: int = 10
    
    # Authentication settings
    jwt_secret_key: str = Field("CHANGE_THIS_IN_PRODUCTION", env="JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Encryption settings
    encryption_key: Optional[str] = Field(None, env="ENCRYPTION_KEY")
    
    # Rate limiting settings
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    token_limit_count: int = 10000
    token_limit_window: int = 3600  # seconds
    
    # LLM settings
    llm_api_key: Optional[str] = Field(None, env="LLM_API_KEY")
    llm_api_base: Optional[str] = Field(None, env="LLM_API_BASE")
    default_model: str = "mistral-medium"
    active_model: Optional[str] = None
    model_config_path: str = "app/llm/config"
    llama3_model_path: Optional[str] = Field(None, env="LLAMA3_MODEL_PATH")
    
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
    
    # Cache settings
    cache_type: str = "memory"  # memory, redis
    cache_redis_url: Optional[str] = None
    cache_ttl: int = 3600  # seconds
    
    # Client settings
    default_client_id: Optional[str] = None
    
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
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed = ["development", "testing", "staging", "production"]
        if v.lower() not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v.lower()
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level setting."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()
    
    @validator("database_url")
    def validate_database_url(cls, v, values):
        """Validate database URL based on environment."""
        if values.get("environment") == "production" and "localhost" in v:
            raise ValueError("Production environment should not use localhost database")
        return v
    
    @validator("jwt_secret_key")
    def validate_jwt_secret(cls, v, values):
        """Validate JWT secret key."""
        if values.get("environment") == "production" and v == "CHANGE_THIS_IN_PRODUCTION":
            raise ValueError("Production environment requires a secure JWT secret key")
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