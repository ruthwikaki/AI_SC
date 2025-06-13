# backend/app/api/middleware/rate_limit.py
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, Optional, Tuple, Any
import time
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict, deque
from datetime import datetime, timedelta

from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with configurable limits per environment.
    Uses a sliding window approach for more accurate rate limiting.
    """
    
    def __init__(self, app, requests_per_minute: int = None):
        super().__init__(app)
        
        # Get rate limit from settings or use defaults based on environment
        if settings.environment == "development":
            # Much higher limits for development
            self.requests_per_minute = requests_per_minute or 1000
            self.requests_per_hour = 10000
            self.burst_size = 50  # Allow bursts of requests
        else:
            # Production limits
            self.requests_per_minute = requests_per_minute or settings.rate_limit_requests or 100
            self.requests_per_hour = self.requests_per_minute * 30  # Not quite 60 to be lenient
            self.burst_size = 20
        
        # Storage for request timestamps
        self.requests = defaultdict(lambda: deque())
        self.locks = {}
        
        # Whitelist certain paths that shouldn't be rate limited
        self.whitelist_paths = [
            "/api/health",
            "/api/health/db",
            "/api/docs",
            "/api/openapi.json",
            "/favicon.ico"
        ]
        
        # Different limits for different endpoints
        self.endpoint_limits = {
            "/api/auth/login": 10,  # Stricter for login attempts
            "/api/auth/register": 5,  # Even stricter for registration
            "/api/queries/execute": 30,  # Lower limit for expensive operations
            "/api/export": 10,  # Lower limit for export operations
        }
        
        logger.info(f"Rate limiter initialized: {self.requests_per_minute} requests/minute, environment: {settings.environment}")
    
    async def get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for the given key."""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        return self.locks[key]
    
    def get_client_key(self, request: Request) -> str:
        """Get a unique key to identify the client."""
        # Try to get from user state
        user = getattr(request.state, "user", None)
        if user and hasattr(user, "id"):
            return f"user:{user.id}"
        elif user and isinstance(user, dict) and user.get("id"):
            return f"user:{user.get('id')}"
        
        # Try to get client_id from state
        if hasattr(request.state, "client_id"):
            return f"client:{request.state.client_id}"
        
        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        
        # Handle forwarded IPs (when behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_host = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_host}"
    
    def get_endpoint_limit(self, path: str) -> int:
        """Get the rate limit for a specific endpoint."""
        # Check if path matches any specific endpoint limits
        for endpoint_pattern, limit in self.endpoint_limits.items():
            if path.startswith(endpoint_pattern):
                return limit
        return self.requests_per_minute
    
    def clean_old_requests(self, request_times: deque, current_time: float, window_seconds: int = 60):
        """Remove requests older than the window."""
        cutoff_time = current_time - window_seconds
        while request_times and request_times[0] < cutoff_time:
            request_times.popleft()
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for whitelisted paths
        if any(request.url.path.startswith(path) for path in self.whitelist_paths):
            return await call_next(request)
        
        # Skip rate limiting for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        client_key = self.get_client_key(request)
        lock = await self.get_lock(client_key)
        
        async with lock:
            now = time.time()
            request_times = self.requests[client_key]
            
            # Clean old requests from the sliding window
            self.clean_old_requests(request_times, now)
            
            # Get the limit for this endpoint
            limit = self.get_endpoint_limit(request.url.path)
            
            # Check if rate limit is exceeded
            if len(request_times) >= limit:
                # Calculate when the client can retry
                oldest_request = request_times[0]
                retry_after = int(60 - (now - oldest_request))
                
                logger.warning(
                    f"Rate limit exceeded for {client_key} on {request.url.path}. "
                    f"Requests: {len(request_times)}/{limit}"
                )
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": {
                            "code": "rate_limit_exceeded",
                            "message": f"Too many requests. Please try again in {retry_after} seconds.",
                            "retry_after": retry_after,
                            "limit": limit,
                            "window": "60 seconds"
                        }
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(oldest_request + 60))
                    }
                )
            
            # Check burst protection
            recent_requests = [t for t in request_times if now - t < 10]  # Last 10 seconds
            if len(recent_requests) >= self.burst_size:
                logger.warning(f"Burst limit exceeded for {client_key}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": {
                            "code": "burst_limit_exceeded",
                            "message": "Too many requests in a short time. Please slow down.",
                            "burst_limit": self.burst_size,
                            "burst_window": "10 seconds"
                        }
                    },
                    headers={"Retry-After": "10"}
                )
            
            # Add this request
            request_times.append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        remaining = max(0, limit - len(request_times))
        reset_time = int(now + 60)
        
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        # Add warning header if getting close to limit
        if remaining < limit * 0.2:  # Less than 20% remaining
            response.headers["X-RateLimit-Warning"] = f"Only {remaining} requests remaining"
        
        return response
    
    async def __call__(self, scope, receive, send):
        """ASGI3 interface."""
        if scope["type"] == "http":
            # Create request object for middleware
            request = Request(scope, receive)
            
            # Create a response handler
            async def send_wrapper(message):
                await send(message)
            
            # Dispatch through middleware
            response = await self.dispatch(request, lambda req: self.app(scope, receive, send))
            
            # If response is a JSONResponse (rate limit error), handle it
            if isinstance(response, JSONResponse):
                await response(scope, receive, send)
            else:
                # Otherwise, pass through
                await self.app(scope, receive, send)
        else:
            # Not HTTP, pass through
            await self.app(scope, receive, send)


# Token-based rate limiter for LLM operations
class TokenRateLimiter:
    """
    Separate rate limiter for LLM token usage.
    Tracks token consumption rather than request count.
    """
    
    def __init__(self, tokens_per_hour: int = 100000, tokens_per_minute: int = 10000):
        self.tokens_per_hour = tokens_per_hour
        self.tokens_per_minute = tokens_per_minute
        self.token_usage = defaultdict(lambda: {"hour": deque(), "minute": deque()})
        self.locks = {}
    
    async def check_token_limit(self, user_id: str, token_count: int) -> Tuple[bool, Optional[str]]:
        """
        Check if user can consume the requested tokens.
        Returns (allowed, error_message)
        """
        if user_id not in self.locks:
            self.locks[user_id] = asyncio.Lock()
        
        async with self.locks[user_id]:
            now = time.time()
            usage = self.token_usage[user_id]
            
            # Clean old entries
            hour_cutoff = now - 3600
            minute_cutoff = now - 60
            
            usage["hour"] = deque(
                (timestamp, tokens) for timestamp, tokens in usage["hour"]
                if timestamp > hour_cutoff
            )
            usage["minute"] = deque(
                (timestamp, tokens) for timestamp, tokens in usage["minute"]
                if timestamp > minute_cutoff
            )
            
            # Calculate current usage
            hour_tokens = sum(tokens for _, tokens in usage["hour"])
            minute_tokens = sum(tokens for _, tokens in usage["minute"])
            
            # Check limits
            if minute_tokens + token_count > self.tokens_per_minute:
                return False, f"Token limit exceeded: {self.tokens_per_minute} tokens per minute"
            
            if hour_tokens + token_count > self.tokens_per_hour:
                return False, f"Token limit exceeded: {self.tokens_per_hour} tokens per hour"
            
            # Record usage
            usage["hour"].append((now, token_count))
            usage["minute"].append((now, token_count))
            
            return True, None
    
    def get_usage_stats(self, user_id: str) -> Dict[str, Any]:
        """Get current token usage statistics for a user."""
        now = time.time()
        usage = self.token_usage.get(user_id, {"hour": deque(), "minute": deque()})
        
        # Clean and calculate
        hour_tokens = sum(
            tokens for timestamp, tokens in usage["hour"]
            if timestamp > now - 3600
        )
        minute_tokens = sum(
            tokens for timestamp, tokens in usage["minute"]
            if timestamp > now - 60
        )
        
        return {
            "tokens_used_last_minute": minute_tokens,
            "tokens_used_last_hour": hour_tokens,
            "tokens_remaining_minute": max(0, self.tokens_per_minute - minute_tokens),
            "tokens_remaining_hour": max(0, self.tokens_per_hour - hour_tokens),
            "limits": {
                "per_minute": self.tokens_per_minute,
                "per_hour": self.tokens_per_hour
            }
        }


# Global token rate limiter instance
token_limiter = TokenRateLimiter()