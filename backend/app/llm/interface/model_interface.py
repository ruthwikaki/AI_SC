from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime
import json


from app.utils.logger import get_logger
from app.config import get_settings


# Initialize logger
logger = get_logger(__name__)


# Get settings
settings = get_settings()


class ModelResponse:
    """Response from LLM model"""
    
    def __init__(
        self,
        text: str,
        model_name: str,
        tokens_used: int,
        latency_ms: float,
        finish_reason: Optional[str] = None,
        parsed_data: Optional[Dict[str, Any]] = None,
        raw_response: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None
    ):
        """
        Initialize the model response.
        
        Args:
            text: The text response from the model
            model_name: Name of the model used
            tokens_used: Total tokens used (prompt + response)
            latency_ms: Latency in milliseconds
            finish_reason: Reason for completion (e.g., "stop", "length", etc.)
            parsed_data: Optional structured data parsed from the response
            raw_response: Optional raw response from the model
            confidence_score: Optional confidence score (0-1)
        """
        self.text = text
        self.model_name = model_name
        self.tokens_used = tokens_used
        self.latency_ms = latency_ms
        self.finish_reason = finish_reason
        self.parsed_data = parsed_data or {}
        self.raw_response = raw_response
        self.confidence_score = confidence_score
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "text": self.text,
            "model_name": self.model_name,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
            "parsed_data": self.parsed_data,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat()
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the parsed data.
        
        Args:
            key: Key to retrieve
            default: Default value if key not found
            
        Returns:
            The value or default
        """
        return self.parsed_data.get(key, default)
    
    def __str__(self) -> str:
        """String representation"""
        return self.text


class ModelInterface(ABC):
    """Abstract interface for LLM models"""
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize the model interface.
        
        Args:
            model_name: Name of the model
            api_key: Optional API key for hosted models
            api_base: Optional API base URL for hosted models
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.model_name = model_name
        self.api_key = api_key or settings.llm_api_key
        self.api_base = api_base or settings.llm_api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0
        }
    
    @property
    def name(self) -> str:
        """Get model name"""
        return self.model_name
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
    
    @abstractmethod
    async def embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the model is healthy.
        
        Returns:
            Dictionary with health status
        """
        pass
    
    def update_metrics(self, success: bool, tokens: int, latency_ms: float) -> None:
        """
        Update model metrics.
        
        Args:
            success: Whether the request was successful
            tokens: Number of tokens used
            latency_ms: Latency in milliseconds
        """
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        self.metrics["total_tokens"] += tokens
        self.metrics["total_latency_ms"] += latency_ms
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get model metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate derived metrics
        if metrics["total_requests"] > 0:
            metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
            metrics["avg_latency_ms"] = metrics["total_latency_ms"] / metrics["total_requests"]
        else:
            metrics["success_rate"] = 0
            metrics["avg_latency_ms"] = 0
        
        if metrics["successful_requests"] > 0:
            metrics["avg_tokens_per_request"] = metrics["total_tokens"] / metrics["successful_requests"]
        else:
            metrics["avg_tokens_per_request"] = 0
        
        return metrics


class TemplateManager:
    """Helper for managing prompt templates"""
    
    @staticmethod
    def render_template(template: str, context: Dict[str, Any]) -> str:
        """
        Render a template with context.
        
        Args:
            template: Template string with {placeholders}
            context: Dictionary of values to insert
            
        Returns:
            Rendered template
        """
        try:
            return template.format(**context)
        except KeyError as e:
            logger.error(f"Missing key in template context: {e}")
            # Return template with missing keys marked
            for key in context:
                template = template.replace(f"{{{key}}}", str(context[key]))
            return template
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return template
    
    @staticmethod
    def parse_structured_response(text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse structured data from a response.
        
        Args:
            text: Response text
            schema: JSON schema for expected structure
            
        Returns:
            Parsed data or empty dict if parsing fails
        """
        try:
            # Look for JSON block in the response
            json_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_pattern, text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                parsed_data = json.loads(json_str)
                
                # Simple validation against schema
                if not TemplateManager._validate_against_schema(parsed_data, schema):
                    logger.warning("Parsed data does not match schema")
                
                return parsed_data
            
            # Try to find JSON without code block markers
            try:
                # Find first { and last }
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    json_str = text[start_idx:end_idx+1]
                    parsed_data = json.loads(json_str)
                    
                    # Simple validation against schema
                    if not TemplateManager._validate_against_schema(parsed_data, schema):
                        logger.warning("Parsed data does not match schema")
                    
                    return parsed_data
            except (ValueError, json.JSONDecodeError):
                pass
            
            logger.warning("Could not find valid JSON in response")
            return {}
            
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing structured response: {e}")
            return {}
    
    @staticmethod
    def _validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Simple schema validation.
        
        Args:
            data: Data to validate
            schema: JSON schema
            
        Returns:
            True if data matches schema, False otherwise
        """
        # This is a simplified schema validation
        # In a real implementation, you'd use a library like jsonschema
        
        if "properties" not in schema:
            return True
        
        for prop, details in schema["properties"].items():
            # Check if required property is present
            if schema.get("required", []) and prop in schema["required"] and prop not in data:
                logger.warning(f"Required property '{prop}' missing")
                return False
            
            # If property exists, check type
            if prop in data and data[prop] is not None and "type" in details:
                prop_type = details["type"]
                
                if prop_type == "string" and not isinstance(data[prop], str):
                    logger.warning(f"Property '{prop}' should be string, got {type(data[prop])}")
                    return False
                
                elif prop_type == "number" and not isinstance(data[prop], (int, float)):
                    logger.warning(f"Property '{prop}' should be number, got {type(data[prop])}")
                    return False
                
                elif prop_type == "integer" and not isinstance(data[prop], int):
                    logger.warning(f"Property '{prop}' should be integer, got {type(data[prop])}")
                    return False
                
                elif prop_type == "boolean" and not isinstance(data[prop], bool):
                    logger.warning(f"Property '{prop}' should be boolean, got {type(data[prop])}")
                    return False
                
                elif prop_type == "array" and not isinstance(data[prop], list):
                    logger.warning(f"Property '{prop}' should be array, got {type(data[prop])}")
                    return False
                
                elif prop_type == "object" and not isinstance(data[prop], dict):
                    logger.warning(f"Property '{prop}' should be object, got {type(data[prop])}")
                    return False
        
        return True


import re


class SQLParser:
    """Helper for extracting SQL from responses"""
    
    @staticmethod
    def extract_sql(text: str) -> Optional[str]:
        """
        Extract SQL query from text.
        
        Args:
            text: Text containing SQL
            
        Returns:
            Extracted SQL or None if not found
        """
        # Look for SQL in code blocks
        sql_pattern = r'```(?:sql)?\s*(SELECT[\s\S]*?)(?:```|$)'
        match = re.search(sql_pattern, text, re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # If not in code blocks, try to find SQL statement directly
        sql_pattern = r'(SELECT[\s\S]*?;)'
        match = re.search(sql_pattern, text, re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        return None
    
    @staticmethod
    def validate_sql(sql: str) -> bool:
        """
        Basic validation for SQL queries.
        
        Args:
            sql: SQL query
            
        Returns:
            True if valid, False otherwise
        """
        # Check if it's a SELECT query (for safety)
        if not sql.strip().upper().startswith("SELECT"):
            logger.warning("SQL query doesn't start with SELECT")
            return False
        
        # Check for common SQL injection patterns
        dangerous_patterns = [
            r'--',                   # SQL comment
            r'/\*.*?\*/',            # Block comment
            r';.*?(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)',  # Chained dangerous commands
            r'UNION\s+(?:ALL\s+)?SELECT',  # UNION-based injection
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                logger.warning(f"SQL query contains potentially dangerous pattern: {pattern}")
                return False
        
        return True