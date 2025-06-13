import json
import os
from typing import Dict, Any
from pathlib import Path
import hashlib

class LLMConfigManager:
    """Centralized LLM configuration manager"""
    
    def __init__(self):
        # Simple relative path - config is in backend/config
        self.config_path = Path(__file__).parent.parent.parent / "config" / "llm_config.json"
        
        # If running from backend directory, also check local config
        if not self.config_path.exists():
            self.config_path = Path("config/llm_config.json")
        
        self.config_backup_path = self.config_path.parent / "llm_config.backup.json"
        self._config = None
        self._admin_password_hash = os.getenv("LLM_CONFIG_ADMIN_HASH", 
            hashlib.sha256("admin_password".encode()).hexdigest())
        
        print(f"Looking for config at: {self.config_path.absolute()}")
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path.absolute()}")
                
            with open(self.config_path, 'r', encoding='utf-8-sig') as f:
                self._config = json.load(f)
            
            print(f"✅ LLM Config loaded: {self.get_model_name()}")
            return self._config
            
        except Exception as e:
            print(f"❌ Error loading LLM config: {e}")
            # Create a default config if not found
            default_config = {
                "llm": {
                    "provider": "ollama",
                    "base_url": "http://localhost:11434",
                    "model": {
                        "name": "deepseek-coder-v2:16b-lite-instruct-q4_0",
                        "display_name": "DeepSeek Coder v2 16B",
                        "description": "Optimized for code and SQL generation"
                    },
                    "parameters": {
                        "temperature": 0.1,
                        "max_tokens": 2000,
                        "top_p": 0.9,
                        "timeout": 120,
                        "num_predict": 1000,
                        "num_ctx": 4096,
                        "repeat_penalty": 1.1
                    }
                }
            }
            self._config = default_config
            print("⚠️  Using default configuration")
            return self._config
    
    def get_model_name(self) -> str:
        """Get current model name"""
        return self._config["llm"]["model"]["name"]
    
    def get_model_display_name(self) -> str:
        """Get current model display name"""
        return self._config["llm"]["model"]["display_name"]
    
    def get_base_url(self) -> str:
        """Get Ollama base URL"""
        return self._config["llm"]["base_url"]
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self._config["llm"]["parameters"].copy()
    
    def get_timeout(self) -> int:
        """Get request timeout"""
        return self._config["llm"]["parameters"]["timeout"]
    
    def get_max_tokens(self) -> int:
        """Get max tokens"""
        return self._config["llm"]["parameters"]["max_tokens"]
    
    def get_temperature(self) -> float:
        """Get temperature"""
        return self._config["llm"]["parameters"]["temperature"]
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return self._config.copy()
    
    def validate_config(self) -> bool:
        """Validate current configuration"""
        return True

# Global instance
llm_config = LLMConfigManager()

