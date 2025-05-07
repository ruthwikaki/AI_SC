from typing import Dict, List, Any, Optional, Union, Type
import asyncio
import json
from datetime import datetime
import os


from app.llm.interface.model_interface import ModelInterface
from app.utils.logger import get_logger
from app.config import get_settings


# Initialize logger
logger = get_logger(__name__)


# Get settings
settings = get_settings()


# Store for available model implementations
_model_registry: Dict[str, Type[ModelInterface]] = {}


# Active model instance
_active_model: Optional[ModelInterface] = None


# Model switching lock
_model_switch_lock = asyncio.Lock()


# Model configuration cache
_model_configs: Dict[str, Dict[str, Any]] = {}


def register_model(model_name: str, model_class: Type[ModelInterface]) -> None:
    """
    Register a model implementation.
    
    Args:
        model_name: Name of the model
        model_class: Model class implementation
    """
    global _model_registry
    _model_registry[model_name] = model_class
    logger.info(f"Registered model implementation: {model_name}")


async def initialize_models() -> None:
    """
    Initialize the model registry by importing all available model implementations.
    
    This function should be called at application startup.
    """
    try:
        # Import model implementations
        # This will trigger the register_model() calls in each module
        from app.llm.models.mistral import MistralModel
        from app.llm.models.llama3 import Llama3Model
        
        # Load model configurations
        await load_model_configs()
        
        # Set active model based on configuration
        default_model = settings.default_model
        if default_model and default_model in _model_registry:
            await set_active_model(default_model)
        elif _model_registry:
            # Use first available model if default not specified
            first_model = next(iter(_model_registry.keys()))
            await set_active_model(first_model)
            
        logger.info(f"Initialized models: {list(_model_registry.keys())}")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")


async def load_model_configs() -> None:
    """
    Load model configurations from config files.
    """
    global _model_configs
    
    try:
        config_path = settings.model_config_path
        
        if not config_path or not os.path.exists(config_path):
            logger.warning(f"Model config path not found: {config_path}")
            return
        
        # Scan the directory for model config files
        for filename in os.listdir(config_path):
            if filename.endswith(".json"):
                model_name = filename.split(".")[0]
                file_path = os.path.join(config_path, filename)
                
                with open(file_path, "r") as f:
                    config = json.load(f)
                    
                _model_configs[model_name] = config
                logger.info(f"Loaded config for model: {model_name}")
                
    except Exception as e:
        logger.error(f"Error loading model configs: {str(e)}")


def get_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration or None if not found
    """
    return _model_configs.get(model_name)


async def set_active_model(model_name: str) -> bool:
    """
    Set the active model by name.
    
    Args:
        model_name: Name of the model to activate
        
    Returns:
        True if successful, False otherwise
    """
    global _active_model
    
    if model_name not in _model_registry:
        logger.error(f"Model not found in registry: {model_name}")
        return False
    
    # Get model configuration
    config = get_model_config(model_name) or {}
    
    # Create model instance
    try:
        # Acquire lock to prevent concurrent model switching
        async with _model_switch_lock:
            # Initialize the new model
            model_class = _model_registry[model_name]
            
            # Extract model parameters from config
            model_params = config.get("model_params", {})
            
            # Create new instance
            new_model = model_class(
                model_name=model_name,
                api_key=model_params.get("api_key") or settings.llm_api_key,
                api_base=model_params.get("api_base") or settings.llm_api_base,
                max_tokens=model_params.get("max_tokens", 4096),
                temperature=model_params.get("temperature", 0.7),
                timeout=model_params.get("timeout", 30.0),
                max_retries=model_params.get("max_retries", 3)
            )
            
            # Perform health check
            health_status = await new_model.health_check()
            
            if not health_status.get("is_healthy", False):
                logger.error(f"Health check failed for model {model_name}: {health_status.get('message', 'Unknown error')}")
                return False
            
            # Switch the active model
            _active_model = new_model
            
            # Update settings
            settings.active_model = model_name
            
            logger.info(f"Active model switched to: {model_name}")
            return True
            
    except Exception as e:
        logger.error(f"Error setting active model to {model_name}: {str(e)}")
        return False


def get_active_model() -> Optional[ModelInterface]:
    """
    Get the currently active model.
    
    Returns:
        Active model instance or None if not set
    """
    return _active_model


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of available models with their configurations.
    
    Returns:
        List of model information dictionaries
    """
    models = []
    
    for model_name in _model_registry.keys():
        config = get_model_config(model_name) or {}
        
        # Create model info
        model_info = {
            "name": model_name,
            "description": config.get("description", ""),
            "capabilities": config.get("capabilities", []),
            "max_tokens": config.get("model_params", {}).get("max_tokens", 4096),
            "is_active": (_active_model and _active_model.name == model_name)
        }
        
        models.append(model_info)
    
    return models


async def get_model_stats() -> Dict[str, Any]:
    """
    Get statistics for all models.
    
    Returns:
        Dictionary of model statistics
    """
    stats = {
        "active_model": _active_model.name if _active_model else None,
        "available_models": list(_model_registry.keys()),
        "model_metrics": {}
    }
    
    # Get metrics for active model
    if _active_model:
        stats["model_metrics"][_active_model.name] = _active_model.get_metrics()
    
    return stats


async def fallback_to_backup_model() -> bool:
    """
    Switch to a backup model if the active model is unhealthy.
    
    Returns:
        True if switched successfully, False otherwise
    """
    if not _active_model:
        logger.warning("No active model to fallback from")
        return False
    
    current_model = _active_model.name
    
    # Get backup model configuration
    config = get_model_config(current_model) or {}
    backup_model = config.get("backup_model")
    
    if not backup_model or backup_model not in _model_registry:
        # If no specific backup configured, try to find any other available model
        available_models = [m for m in _model_registry.keys() if m != current_model]
        
        if not available_models:
            logger.error("No backup models available")
            return False
        
        backup_model = available_models[0]
    
    logger.warning(f"Falling back from {current_model} to backup model {backup_model}")
    return await set_active_model(backup_model)