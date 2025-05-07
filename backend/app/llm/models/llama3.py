from typing import Dict, List, Any, Optional, Union
import aiohttp
import time
import json
import asyncio
from datetime import datetime
import re
import os


from app.llm.interface.model_interface import ModelInterface, ModelResponse, TemplateManager
from app.llm.controller.active_model_manager import register_model
from app.utils.logger import get_logger
from app.config import get_settings


# Initialize logger
logger = get_logger(__name__)


# Get settings
settings = get_settings()


class Llama3Model(ModelInterface):
    """Meta Llama 3 model implementation"""
    
    def __init__(
        self,
        model_name: str = "llama3-8b",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 30.0,
        max_retries: int = 3,
        local_model_path: Optional[str] = None
    ):
        """
        Initialize the Llama 3 model.
        
        Args:
            model_name: Name of specific Llama model (llama3-8b, llama3-70b, etc.)
            api_key: API key for hosted API
            api_base: API base URL
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            local_model_path: Path to local model files (if running locally)
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
        
        # Determine if using local model or API
        self.local_model_path = local_model_path or settings.llama3_model_path
        self.use_local_model = bool(self.local_model_path and os.path.exists(self.local_model_path))
        
        # Set API endpoints if using API
        if not self.use_local_model:
            self.api_base = api_base or "http://localhost:8000"  # Default to local inference server
            self.completion_endpoint = f"{self.api_base}/generate"
            self.embeddings_endpoint = f"{self.api_base}/embeddings"
        
        self.tokenizer_name = "Llama3"
        
        # Session for API requests
        self._session = None
        
        # Local model and tokenizer instances
        self._local_model = None
        self._local_tokenizer = None
        
        # Initialize local model if needed
        if self.use_local_model:
            asyncio.create_task(self._init_local_model())
    
    async def _init_local_model(self):
        """Initialize the local model and tokenizer"""
        if not self.use_local_model:
            return
        
        try:
            # Import here to avoid dependency if not using local model
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            self._local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
            
            # Load model with hardware acceleration if available
            if torch.cuda.is_available():
                logger.info(f"Loading Llama 3 model on GPU: {self.local_model_path}")
                self._local_model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                logger.info(f"Loading Llama 3 model on CPU: {self.local_model_path}")
                self._local_model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_path,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )
                
            logger.info("Llama 3 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading local Llama 3 model: {str(e)}")
            self.use_local_model = False
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            self._session = aiohttp.ClientSession(headers=headers)
            
        return self._session
    
    async def _generate_with_local_model(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate response using local model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Optional stop sequences
            
        Returns:
            Response data
        """
        if not self.use_local_model or not self._local_model or not self._local_tokenizer:
            raise ValueError("Local model not initialized")
        
        try:
            import torch
            
            # Format the prompt for Llama 3
            formatted_prompt = f"<|begin_of_text|><|user|>\n{prompt}<|end_of_turn|>\n<|assistant|>\n"
            
            # Encode the prompt
            inputs = self._local_tokenizer(formatted_prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self._local_model.device)
            
            # Count input tokens
            input_token_count = len(inputs.input_ids[0])
            
            # Create stop token IDs if stop sequences provided
            stop_token_ids = []
            if stop_sequences:
                for seq in stop_sequences:
                    ids = self._local_tokenizer.encode(seq, add_special_tokens=False)
                    stop_token_ids.append(torch.tensor(ids, dtype=torch.long, device=self._local_model.device))
            
            # Add default stop sequences for Llama 3
            end_of_turn_token_ids = self._local_tokenizer.encode("<|end_of_turn|>", add_special_tokens=False)
            stop_token_ids.append(torch.tensor(end_of_turn_token_ids, dtype=torch.long, device=self._local_model.device))
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                output_ids = self._local_model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 0.01,
                    top_p=0.95,
                    do_sample=temperature > 0,
                    pad_token_id=self._local_tokenizer.pad_token_id or self._local_tokenizer.eos_token_id,
                    eos_token_id=self._local_tokenizer.eos_token_id,
                    stopping_criteria=self._create_stopping_criteria(stop_token_ids) if stop_token_ids else None
                )
            
            generation_time = time.time() - start_time
            
            # Decode the output
            output_text = self._local_tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # Count output tokens
            output_token_count = len(output_ids[0]) - input_token_count
            
            # Create response data structure
            return {
                "text": output_text,
                "tokens_used": {
                    "prompt": input_token_count,
                    "completion": output_token_count,
                    "total": input_token_count + output_token_count
                },
                "latency": generation_time,
                "finish_reason": "stop"
            }
            
        except Exception as e:
            logger.error(f"Error generating with local model: {str(e)}")
            raise
    
    def _create_stopping_criteria(self, stop_token_ids):
        """Create stopping criteria based on stop token IDs"""
        try:
            import torch
            from transformers import StoppingCriteria, StoppingCriteriaList
            
            class StopSequenceCriteria(StoppingCriteria):
                def __init__(self, stop_token_ids, prompt_length):
                    super().__init__()
                    self.stop_token_ids = stop_token_ids
                    self.prompt_length = prompt_length
                
                def __call__(self, input_ids, scores, **kwargs):
                    for stop_ids in self.stop_token_ids:
                        if self._check_sequence(input_ids[0, self.prompt_length:], stop_ids):
                            return True
                    return False
                
                def _check_sequence(self, generated_ids, stop_ids):
                    stop_length = len(stop_ids)
                    if len(generated_ids) < stop_length:
                        return False
                    return torch.all(generated_ids[-stop_length:] == stop_ids).item()
            
            input_ids = torch.tensor([[]], device=self._local_model.device)  # Dummy input_ids
            prompt_length = 0
            
            return StoppingCriteriaList([StopSequenceCriteria(stop_token_ids, prompt_length)])
            
        except Exception as e:
            logger.error(f"Error creating stopping criteria: {str(e)}")
            return None
    
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
        
        # Track metrics
        start_time = time.time()
        success = False
        tokens_used = 0
        
        try:
            response_data = None
            
            # Handle structured output prompt formatting
            if structured_output:
                # Add structure instructions to prompt
                prompt = f"{prompt}\n\nPlease respond with a JSON object that matches this schema:\n{json.dumps(structured_output, indent=2)}"
            
            # Generate using local model or API
            if self.use_local_model and self._local_model:
                # Local model generation
                response_data = await self._generate_with_local_model(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences
                )
                
                text = response_data.get("text", "")
                tokens_used = response_data.get("tokens_used", {}).get("total", 0)
                finish_reason = response_data.get("finish_reason", "stop")
                latency_ms = response_data.get("latency", 0) * 1000
                
            else:
                # API generation
                # Get session
                session = await self._get_session()
                
                # Prepare request payload
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                # Add stop sequences if provided
                if stop_sequences:
                    payload["stop"] = stop_sequences
                
                # Send request with retries
                last_error = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        async with session.post(
                            self.completion_endpoint,
                            json=payload,
                            timeout=self.timeout
                        ) as response:
                            if response.status == 200:
                                response_data = await response.json()
                                break
                            else:
                                error_text = await response.text()
                                last_error = f"HTTP {response.status}: {error_text}"
                                logger.warning(f"Llama API error (attempt {attempt+1}/{self.max_retries+1}): {last_error}")
                                
                                # Check for rate limiting
                                if response.status == 429:
                                    # Exponential backoff
                                    wait_time = 2 ** attempt
                                    await asyncio.sleep(wait_time)
                                
                    except asyncio.TimeoutError:
                        last_error = "Request timed out"
                        logger.warning(f"Llama API timeout (attempt {attempt+1}/{self.max_retries+1})")
                        
                        # Wait before retry
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        last_error = str(e)
                        logger.warning(f"Llama API error (attempt {attempt+1}/{self.max_retries+1}): {last_error}")
                        
                        # Wait before retry
                        await asyncio.sleep(1)
                
                # If all retries failed
                if not response_data:
                    raise Exception(f"Failed after {self.max_retries + 1} attempts: {last_error}")
                
                # Extract response data from API response
                text = response_data.get("generation", "")
                tokens_used = response_data.get("usage", {}).get("total_tokens", 0)
                finish_reason = response_data.get("stop_reason", "stop")
                latency_ms = (time.time() - start_time) * 1000
            
            # Parse structured data if requested
            parsed_data = None
            if structured_output:
                try:
                    parsed_data = TemplateManager.parse_structured_response(text, structured_output)
                except Exception as e:
                    logger.warning(f"Failed to parse structured response: {str(e)}")
            
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
            
            logger.error(f"Error generating response from Llama 3: {str(e)}")
            
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
        # Use local tokenizer if available
        if self.use_local_model and self._local_tokenizer:
            tokens = self._local_tokenizer.encode(text)
            return len(tokens)
        
        # If API mode, check if tokenization endpoint exists
        session = await self._get_session()
        
        try:
            payload = {
                "text": text
            }
            
            async with session.post(
                f"{self.api_base}/tokenize",
                json=payload,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    return response_data.get("token_count", 0)
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
        # Use local model if available
        if self.use_local_model and self._local_model and self._local_tokenizer:
            try:
                import torch
                import numpy as np
                
                # We'll use last hidden states as embeddings
                embeddings = []
                
                for text in texts:
                    # Tokenize
                    inputs = self._local_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    input_ids = inputs.input_ids.to(self._local_model.device)
                    attention_mask = inputs.attention_mask.to(self._local_model.device)
                    
                    # Get model output
                    with torch.no_grad():
                        outputs = self._local_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True
                        )
                    
                    # Use last hidden state of the last token as embedding
                    last_hidden_state = outputs.hidden_states[-1].detach().cpu().numpy()
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape).cpu().numpy()
                    
                    # Average pool over sequence length
                    sum_embeddings = np.sum(last_hidden_state * attention_mask_expanded, axis=1)
                    sum_mask = attention_mask_expanded.sum(axis=1)
                    pooled_embedding = sum_embeddings / sum_mask
                    
                    # Normalize
                    embedding = pooled_embedding.flatten()
                    norm = np.linalg.norm(embedding)
                    normalized_embedding = embedding / norm if norm > 0 else embedding
                    
                    embeddings.append(normalized_embedding.tolist())
                
                return embeddings
                
            except Exception as e:
                logger.error(f"Error generating embeddings with local model: {str(e)}")
                # Fall back to API if available
        
        # Use API if not using local model or local embedding failed
        if not self.use_local_model:
            # Get session
            session = await self._get_session()
            
            try:
                # Prepare request payload
                payload = {
                    "model": f"{self.model_name}-embed",
                    "texts": texts
                }
                
                # Send request
                async with session.post(
                    self.embeddings_endpoint,
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        embeddings = response_data.get("embeddings", [])
                        return embeddings
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                        
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                # Return empty embeddings in case of error
                return [[] for _ in texts]
        
        # If all methods fail, return empty embeddings
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
            
            # Check model loading for local model
            if self.use_local_model and not self._local_model:
                issues.append("Local model not loaded")
            
            # Return health status
            return {
                "is_healthy": len(issues) == 0,
                "model_name": self.model_name,
                "mode": "local" if self.use_local_model else "api",
                "latency": latency,
                "issues": issues,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            
            return {
                "is_healthy": False,
                "model_name": self.model_name,
                "mode": "local" if self.use_local_model else "api",
                "latency": time.time() - start_time,
                "issues": [str(e)],
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Close resources"""
        if self._session:
            await self._session.close()
            self._session = None
        
        # Clear model from GPU if using local model
        if self.use_local_model and self._local_model:
            try:
                import torch
                
                # Move model to CPU to free GPU memory
                self._local_model = self._local_model.to("cpu")
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Set to None to allow garbage collection
                self._local_model = None
                self._local_tokenizer = None
                
            except Exception as e:
                logger.error(f"Error closing local model: {str(e)}")


# Register this model with the model manager
register_model("llama3-8b", Llama3Model)
register_model("llama3-70b", Llama3Model)