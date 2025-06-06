"""
Model loader that checks external model paths
"""

import os
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Load model configuration
if os.path.exists('.env.models'):
    load_dotenv('.env.models')

class ModelPathResolver:
    """Resolves model paths from various sources"""
    
    @staticmethod
    def find_model_path(model_name: str = None) -> Optional[str]:
        """Find model path from environment or config"""
        
        # Check direct MODEL_PATH first
        model_path = os.getenv('MODEL_PATH')
        if model_path and os.path.exists(model_path):
            logger.info(f"Found model at MODEL_PATH: {model_path}")
            return model_path
        
        # Check model-specific paths
        if model_name:
            env_key = f"{model_name.upper().replace('-', '_')}_PATH"
            model_path = os.getenv(env_key)
            if model_path and os.path.exists(model_path):
                logger.info(f"Found model at {env_key}: {model_path}")
                return model_path
        
        # Check MODEL_BASE_DIR for models
        base_dir = os.getenv('MODEL_BASE_DIR')
        if base_dir:
            # Look for any .gguf files
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith('.gguf'):
                        model_path = os.path.join(root, file)
                        logger.info(f"Found model in base dir: {model_path}")
                        return model_path
        
        # Check local models directory
        local_models = ['models', 'backend/models', '../models']
        for model_dir in local_models:
            if os.path.exists(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith('.gguf'):
                            model_path = os.path.join(root, file)
                            logger.info(f"Found model in local dir: {model_path}")
                            return model_path
        
        logger.warning("No model file found")
        return None
    
    @staticmethod
    def get_model_config(model_path: str) -> Dict[str, Any]:
        """Load model configuration if available"""
        if not model_path:
            return {}
        
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        return {}
