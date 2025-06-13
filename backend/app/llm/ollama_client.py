import httpx
from typing import Dict, Any, Optional
from app.llm.config_manager import llm_config

class OllamaClient:
    """Ollama client using centralized configuration"""
    
    def __init__(self):
        # Always use config from manager
        self.config = llm_config
        self.client = None
        
    def _get_client(self):
        """Get HTTP client with current config"""
        if not self.client:
            self.client = httpx.AsyncClient(
                timeout=self.config.get_timeout(),
                base_url=self.config.get_base_url()
            )
        return self.client
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using current model and parameters"""
        # Get current configuration
        model = self.config.get_model_name()
        params = self.config.get_parameters()
        
        # Override with any provided kwargs
        params.update(kwargs)
        
        # Build request
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": params.get("stream", False),
            "options": {
                "temperature": params["temperature"],
                "num_predict": params.get("num_predict", params["max_tokens"]),
                "top_p": params["top_p"],
                "repeat_penalty": params["repeat_penalty"],
                "stop": params.get("stop_sequences", []),
                "num_ctx": params.get("num_ctx", 4096),
                "seed": params.get("seed")
            }
        }
        
        print(f"🤖 Using model: {self.config.get_model_display_name()}")
        
        try:
            client = self._get_client()
            response = await client.post("/api/generate", json=request_data)
            response.raise_for_status()
            return response.json()
            
        except httpx.TimeoutException:
            raise Exception(f"Model timeout after {self.config.get_timeout()}s")
        except Exception as e:
            raise Exception(f"Model error: {str(e)}")
    
    async def check_model_availability(self) -> bool:
        """Check if current model is available"""
        try:
            client = self._get_client()
            response = await client.get("/api/tags")
            models = response.json().get("models", [])
            
            current_model = self.config.get_model_name()
            available = any(m["name"] == current_model for m in models)
            
            if not available:
                print(f"⚠️  Model {current_model} not found in Ollama")
                print(f"Available models: {[m['name'] for m in models]}")
                
            return available
            
        except Exception as e:
            print(f"❌ Error checking model: {e}")
            return False
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
