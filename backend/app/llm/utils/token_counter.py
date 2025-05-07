# app/llm/utils/token_counter.py

from typing import Dict, List, Any, Optional, Union
import re
import math

# Import tiktoken if available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Cache for tokenizers
_tokenizers = {}

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: Text to count tokens for
        model: Model name to use for counting
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    
    # Try to use tiktoken if available
    if TIKTOKEN_AVAILABLE:
        try:
            return count_tokens_tiktoken(text, model)
        except Exception as e:
            logger.warning(f"Error counting tokens with tiktoken: {str(e)}")
    
    # Fall back to approximation
    return count_tokens_approximate(text)

def count_tokens_tiktoken(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens using the tiktoken library.
    
    Args:
        text: Text to count tokens for
        model: Model name to use for counting
        
    Returns:
        Number of tokens
    """
    if not TIKTOKEN_AVAILABLE:
        raise ImportError("tiktoken is not available")
    
    # Map model names to encoding names
    encoding_name = "cl100k_base"  # Default for newer models
    
    # Check for common models
    if "gpt-4" in model:
        encoding_name = "cl100k_base"
    elif "gpt-3.5" in model:
        encoding_name = "cl100k_base"
    elif "davinci" in model or "text-davinci" in model:
        encoding_name = "p50k_base"
    elif "mistral" in model:
        encoding_name = "cl100k_base"  # Approximation
    elif "llama" in model:
        encoding_name = "cl100k_base"  # Approximation
    
    # Get or create tokenizer
    if encoding_name not in _tokenizers:
        _tokenizers[encoding_name] = tiktoken.get_encoding(encoding_name)
    
    encoding = _tokenizers[encoding_name]
    
    # Count tokens
    token_ids = encoding.encode(text)
    return len(token_ids)

def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count based on word and character count.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Approximate number of tokens
    """
    # Simple approximation: ~4 characters per token for English text
    char_count = len(text)
    
    # Count words
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    
    # For English text: ~0.75 tokens per word and ~0.25 tokens per character
    # This is a very rough approximation
    return max(1, math.ceil(word_count * 0.75 + char_count * 0.25 / 4))

def count_messages_tokens(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in a list of chat messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model name to use for counting
        
    Returns:
        Number of tokens
    """
    if not messages:
        return 0
    
    token_count = 0
    
    # Add base tokens for message formatting
    # For ChatML format, each message has some overhead
    token_count += len(messages) * 4  # Approximate overhead per message
    
    # Count tokens in each message
    for message in messages:
        content = message.get("content", "")
        token_count += count_tokens(content, model)
    
    return token_count

def truncate_text_to_token_limit(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """
    Truncate text to fit within a token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name to use for counting
        
    Returns:
        Truncated text
    """
    if not text or max_tokens <= 0:
        return ""
    
    # Check if already within limit
    current_tokens = count_tokens(text, model)
    if current_tokens <= max_tokens:
        return text
    
    # If using tiktoken, we can do precise truncation
    if TIKTOKEN_AVAILABLE:
        try:
            # Get encoding
            encoding_name = "cl100k_base"
            if encoding_name not in _tokenizers:
                _tokenizers[encoding_name] = tiktoken.get_encoding(encoding_name)
            
            encoding = _tokenizers[encoding_name]
            
            # Encode and truncate
            token_ids = encoding.encode(text)
            truncated_ids = token_ids[:max_tokens]
            
            # Decode truncated tokens
            return encoding.decode(truncated_ids)
        except Exception as e:
            logger.warning(f"Error truncating with tiktoken: {str(e)}")
    
    # Fall back to approximate truncation
    ratio = max_tokens / current_tokens
    char_limit = int(len(text) * ratio)
    
    # Truncate to character limit
    truncated = text[:char_limit]
    
    # Check if we need to truncate more
    while count_tokens(truncated, model) > max_tokens and len(truncated) > 0:
        truncated = truncated[:int(len(truncated) * 0.9)]  # Remove 10% more
    
    return truncated

def estimate_response_tokens(prompt_tokens: int, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate the number of tokens in a response based on prompt size.
    
    Args:
        prompt_tokens: Number of tokens in the prompt
        model: Model name
        
    Returns:
        Estimated response tokens
    """
    # Different models have different response patterns
    if "gpt-4" in model:
        # GPT-4 tends to be more verbose
        return min(4096, int(prompt_tokens * 1.5))
    elif "gpt-3.5" in model:
        # GPT-3.5 is usually less verbose
        return min(2048, int(prompt_tokens * 1.2))
    elif "mistral" in model:
        # Approximation for Mistral
        return min(2048, int(prompt_tokens * 1.3))
    elif "llama" in model:
        # Approximation for Llama
        return min(2048, int(prompt_tokens * 1.3))
    else:
        # Default guess
        return min(2048, int(prompt_tokens * 1.2))