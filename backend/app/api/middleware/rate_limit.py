from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, Optional, Tuple, Callable, Any
import time
import asyncio
from datetime import datetime
import hashlib

from app.utils.logger import get_logger
from app.config import get_settings
from app.llm.utils.token_counter import count_tokens

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

class RateLimiter:
    """
    Base rate limiter class to track and limit request rates.
    """
    
    def __init__(self, rate_limit: int, time_window: int):
        """
        Initialize the rate limiter.
        
        Args:
            rate_limit: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
        """
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.requests = {}  # Store request counts by key
        self.locks = {}     # Lock per client for thread safety
    
    async def get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for the given key."""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        return self.locks[key]
    
    async def is_rate_limited(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if the request should be rate limited.
        
        Args:
            key: The key to identify the client (e.g., IP address, client ID)
            
        Returns:
            Tuple of (is_limited, rate_limit_info)
        """
        # Get lock for thread safety
        lock = await self.get_lock(key)
        
        async with lock:
            now = time.time()
            
            # Initialize or cleanup expired requests
            if key not in self.requests:
                self.requests[key] = []
            
            # Remove requests outside the time window
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if now - req_time < self.time_window
            ]
            
            # Check if rate limit is exceeded
            is_limited = len(self.requests[key]) >= self.rate_limit
            
            # If not limited, add this request
            if not is_limited:
                self.requests[key].append(now)
            
            # Calculate rate limit info
            rate_info = {
                "limit": self.rate_limit,
                "remaining": max(0, self.rate_limit - len(self.requests[key])),
                "reset": int(now + self.time_window - (
                    0 if not self.requests[key] else 
                    (now - min(self.requests[key]))
                )),
                "window": self.time_window
            }
            
            return is_limited, rate_info

class TokenRateLimiter(RateLimiter):
    """
    Rate limiter that counts tokens instead of requests.
    Used for LLM token rate limiting.
    """
    
    def __init__(self, token_limit: int, time_window: int):
        """
        Initialize the token rate limiter.
        
        Args:
            token_limit: Maximum number of tokens allowed in the time window
            time_window: Time window in seconds
        """
        super().__init__(token_limit, time_window)
        self.token_counts = {}  # Store token counts by key and timestamp
    
    async def is_token_rate_limited(
        self, 
        key: str, 
        tokens: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if the token usage exceeds the rate limit.
        
        Args:
            key: The key to identify the client
            tokens: Number of tokens in this request
            
        Returns:
            Tuple of (is_limited, rate_limit_info)
        """
        # Get lock for thread safety
        lock = await self.get_lock(key)
        
        async with lock:
            now = time.time()
            
            # Initialize or cleanup expired token counts
            if key not in self.token_counts:
                self.token_counts[key] = []
            
            # Remove token counts outside the time window
            self.token_counts[key] = [
                (ts, count) for ts, count in self.token_counts[key]
                if now - ts < self.time_window
            ]
            
            # Calculate total tokens used in the time window
            total_tokens = sum(count for _, count in self.token_counts[key])
            
            # Check if token limit is exceeded
            is_limited = total_tokens + tokens > self.rate_limit
            
            # If not limited, add this token count
            if not is_limited:
                self.token_counts[key].append((now, tokens))
            
            # Calculate rate limit info
            rate_info = {
                "limit": self.rate_limit,
                "remaining": max(0, self.rate_limit - total_tokens),
                "reset": int(now + self.time_window - (
                    0 if not self.token_counts[key] else 
                    (now - min(ts for ts, _ in self.token_counts[key]))
                )),
                "window": self.time_window,
                "tokens_requested": tokens
            }
            
            return is_limited, rate_info

class RateLimitMiddleware:
    """
    Middleware for API rate limiting.
    
    This middleware applies different rate limits based on:
    1. General API request rate limit
    2. LLM token usage rate limit
    3. Rate limits based on client subscription tier
    """
    
    def __init__(self):
        """Initialize the rate limit middleware with different limiters."""
        # General API request rate limiter
        self.api_limiter = RateLimiter(
            rate_limit=settings.rate_limit_requests,
            time_window=settings.rate_limit_window
        )
        
        # LLM token usage rate limiter
        self.token_limiter = TokenRateLimiter(
            token_limit=settings.token_limit_count,
            time_window=settings.token_limit_window
        )
        
        # Rate limits by subscription tier
        self.tier_limits = {
            "free": {"requests": 100, "tokens": 10000},
            "basic": {"requests": 1000, "tokens": 100000},
            "pro": {"requests": 10000, "tokens": 1000000},
            "enterprise": {"requests": 100000, "tokens": 10000000}
        }
    
    def _get_client_key(self, request: Request) -> str:
        """
        Get a unique key to identify the client.
        
        Priority:
        1. Client ID from authenticated user
        2. API key header
        3. IP address
        """
        # Try to get client ID from user state
        user = getattr(request.state, "user", None)
        if user and user.get("client_id"):
            return f"client:{user.get('client_id')}"
        
        # Try to get API key from header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key}"
        
        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"
    
    def _get_subscription_tier(self, request: Request) -> str:
        """Get the client's subscription tier."""
        # Try to get from user state
        user = getattr(request.state, "user", None)
        if user and user.get("client_id"):
            # In a real implementation, you would look up the client's tier
            # For now, we'll assume a default tier
            return "basic"
        
        # Default for unauthenticated requests
        return "free"
    
    def _is_llm_endpoint(self, request: Request) -> bool:
        """Check if the current endpoint is an LLM endpoint."""
        llm_paths = [
            "/queries",
            "/analytics/custom-analysis"
        ]
        
        path = request.url.path
        return any(path.startswith(p) for p in llm_paths)
    
    async def __call__(self, request: Request, call_next):
        # Get client identification key
        client_key = self._get_client_key(request)
        
        # Get subscription tier and limits
        tier = self._get_subscription_tier(request)
        tier_limits = self.tier_limits.get(tier, self.tier_limits["free"])
        
        # Check general API rate limit
        is_limited, rate_info = await self.api_limiter.is_rate_limited(client_key)
        
        if is_limited:
            logger.warning(f"Rate limit exceeded for {client_key}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": "Too many requests, please try again later",
                        "details": rate_info
                    }
                },
                headers={"Retry-After": str(rate_info["reset"] - int(time.time()))}
            )
        
        # For LLM endpoints, check token rate limit
        if self._is_llm_endpoint(request):
            # For POST requests that might contain text to tokenize
            if request.method == "POST":
                # We'll estimate tokens from the content length as a proxy
                # In a real implementation, you'd parse the request body and count tokens
                content_length = request.headers.get("content-length", 0)
                try:
                    content_length = int(content_length)
                except ValueError:
                    content_length = 0
                
                # Rough estimate: 1 token per 4 characters
                estimated_tokens = content_length // 4
                
                # Check token rate limit
                is_token_limited, token_rate_info = await self.token_limiter.is_token_rate_limited(
                    client_key, estimated_tokens
                )
                
                if is_token_limited:
                    logger.warning(f"Token rate limit exceeded for {client_key}")
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "error": {
                                "code": "token_limit_exceeded",
                                "message": "Token usage limit exceeded, please try again later",
                                "details": token_rate_info
                            }
                        },
                        headers={"Retry-After": str(token_rate_info["reset"] - int(time.time()))}
                    )
        
        # Add rate limit headers to the response
        response = await call_next(request)
        
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
        
        return response