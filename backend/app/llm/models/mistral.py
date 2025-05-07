from typing import Dict, List, Any, Optional, Union
import aiohttp
import time
import json
import asyncio
from datetime import datetime
import re


from app.llm.interface.model_interface import ModelInterface, ModelResponse, TemplateManager
from app.llm.controller.active_model_manager import register_model
from app.utils.logger import get_logger
from app.config import get_settings


# Initialize logger
logger = get_logger(__name__)


# Get settings
settings = get_settings()


class MistralModel(ModelInterface):
    """Mistral AI model implementation"""
    
    def __init__(
        self,
        model_name: str = "mistral-medium",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize the Mistral model.
        
        Args:
            model_name: Name of specific Mistral model (mistral-tiny, mistral-small, mistral-medium, etc.)
            api_key: API key for Mistral API
            api_base: API base URL
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Set Mistral-specific attributes
        self.api_base = api_base or "https://api.mistral.ai/v1"
        self.chat_endpoint = f"{self.api_base}/chat/completions"
        self.embeddings_endpoint = f"{self.api_base}/embeddings"
        self.tokenizer_name = "Mistral"
        
        # Session for API requests
        self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._session
    
    async def generate(
        self,
        prompt: str = None,
        prompt_template: str = None,
        context: Dict[str, Any] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        structured_output: Optional[Dict[str, Any]] = None
    ) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            prompt: Raw prompt text (either prompt or prompt_template must be provided)
            prompt_template: Optional template to use
            context: Optional context to use with template
            max_tokens: Maximum tokens to generate (overrides instance value)
            temperature: Sampling temperature (overrides instance value)
            stop_sequences: Optional list of stop sequences
            structured_output: Optional schema for structured output
            
        Returns:
            ModelResponse object
        """
        # Process prompt
        if prompt_template and context:
            prompt = TemplateManager.render_template(prompt_template, context)
        elif not prompt:
            raise ValueError("Either prompt or (prompt_template and context) must be provided")
        
        # Override parameters if provided
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        # Add structured output format if provided
        if structured_output:
            response_format = {
                "type": "json_object",
                "schema": structured_output
            }
            payload["response_format"] = response_format
        
        # Track metrics
        start_time = time.time()
        success = False
        tokens_used = 0
        
        try:
            # Get session
            session = await self._get_session()
            
            # Send request with retries
            response_data = None
            last_error = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    async with session.post(
                        self.chat_endpoint,
                        json=payload,
                        timeout=self.timeout
                    ) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            break
                        else:
                            error_text = await response.text()
                            last_error = f"HTTP {response.status}: {error_text}"
                            logger.warning(f"Mistral API error (attempt {attempt+1}/{self.max_retries+1}): {last_error}")
                            
                            # Check for rate limiting
                            if response.status == 429:
                                # Exponential backoff
                                wait_time = 2 ** attempt
                                await asyncio.sleep(wait_time)
                            
                except asyncio.TimeoutError:
                    last_error = "Request timed out"
                    logger.warning(f"Mistral API timeout (attempt {attempt+1}/{self.max_retries+1})")
                    
                    # Wait before retry
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Mistral API error (attempt {attempt+1}/{self.max_retries+1}): {last_error}")
                    
                    # Wait before retry
                    await asyncio.sleep(1)
            
            # If all retries failed
            if not response_data:
                raise Exception(f"Failed after {self.max_retries + 1} attempts: {last_error}")
            
            # Process response
            message = response_data.get("choices", [{}])[0].get("message", {})
            text = message.get("content", "")
            finish_reason = response_data.get("choices", [{}])[0].get("finish_reason")
            
            # Get token usage
            usage = response_data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            
            # Parse structured data if requested
            parsed_data = None
            if structured_output:
                try:
                    # Mistral should return valid JSON if response_format is set
                    if message.get("function_call") and "arguments" in message["function_call"]:
                        parsed_data = json.loads(message["function_call"]["arguments"])
                    else:
                        # Try to parse JSON from the response
                        parsed_data = TemplateManager.parse_structured_response(text, structured_output)
                except Exception as e:
                    logger.warning(f"Failed to parse structured response: {str(e)}")
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Mark as successful
            success = True
            
            # Create response
            model_response = ModelResponse(
                text=text,
                model_name=self.model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                parsed_data=parsed_data,
                raw_response=response_data
            )
            
            return model_response
            
        except Exception as e:
            # Calculate latency even for errors
            latency_ms = (time.time() - start_time) * 1000
            
            logger.error(f"Error generating response from Mistral: {str(e)}")
            
            # Create error response
            model_response = ModelResponse(
                text=f"Error: {str(e)}",
                model_name=self.model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                finish_reason="error",
                parsed_data=None,
                raw_response=None
            )
            
            return model_response
            
        finally:
            # Update metrics
            self.update_metrics(success, tokens_used, (time.time() - start_time) * 1000)
    
    async def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[ModelResponse]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens to generate (overrides instance value)
            temperature: Sampling temperature (overrides instance value)
            
        Returns:
            List of ModelResponse objects
        """
        # Create tasks for each prompt
        tasks = [
            self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            for prompt in prompts
        ]
        
        # Run tasks concurrently
        return await asyncio.gather(*tasks)
    
    async def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        # Get session
        session = await self._get_session()
        
        try:
            # Use the tokenization endpoint if available
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            async with session.post(
                f"{self.api_base}/tokenize",
                json=payload,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    return len(response_data.get("tokens", []))
                else:
                    # Fallback to approximate token counting
                    # Rough estimate: 1 token ≈ 4 characters for English text
                    return len(text) // 4 + 1
                    
        except Exception as e:
            logger.warning(f"Error counting tokens, using estimate: {str(e)}")
            # Fallback to approximate token counting
            return len(text) // 4 + 1
    
    async def embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Get session
        session = await self._get_session()
        
        try:
            # Prepare request payload
            payload = {
                "model": f"{self.model_name}-embed",  # Mistral embedding model
                "input": texts
            }
            
            # Send request
            async with session.post(
                self.embeddings_endpoint,
                json=payload,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    embeddings = [item["embedding"] for item in response_data["data"]]
                    return embeddings
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return empty embeddings in case of error
            return [[] for _ in texts]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the model is healthy.
        
        Returns:
            Dictionary with health status
        """
        start_time = time.time()
        issues = []
        
        try:
            # Test a simple generation
            test_prompt = "Hello, are you operational?"
            response = await self.generate(
                prompt=test_prompt,
                max_tokens=10,
                temperature=0.0
            )
            
            # Check if response is valid
            if not response.text or "error" in response.text.lower():
                issues.append(f"Model returned error or empty response: {response.text}")
            
            # Check latency
            latency = time.time() - start_time
            if latency > 5.0:  # More than 5 seconds is concerning
                issues.append(f"High latency: {latency:.2f}s")
            
            # Return health status
            return {
                "is_healthy": len(issues) == 0,
                "model_name": self.model_name,
                "latency": latency,
                "issues": issues,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            
            return {
                "is_healthy": False,
                "model_name": self.model_name,
                "latency": time.time() - start_time,
                "issues": [str(e)],
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Close resources"""
        if self._session:
            await self._session.close()
            self._session = None


# Register this model with the model manager
register_model("mistral-tiny", MistralModel)
register_model("mistral-small", MistralModel)
register_model("mistral-medium", MistralModel)
register_model("mistral-large", MistralModel)