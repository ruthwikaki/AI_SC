# backend/app/api/middleware/rate_limit.py
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, Optional
import time
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict, deque

from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware that works properly with ASGI.
    """
    
    def __init__(self, app, requests_per_minute: int = None):
        super().__init__(app)
        
        # Set limits based on environment
        if settings.environment == "development":
            self.requests_per_minute = 1000  # Very high for development
        else:
            self.requests_per_minute = requests_per_minute or settings.rate_limit_requests or 100
        
        # Storage for request timestamps
        self.requests = defaultdict(lambda: deque())
        self.locks = {}
        
        # Whitelist paths that shouldn't be rate limited
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
        if hasattr(request.state, "user"):
            user = request.state.user
            if isinstance(user, dict) and user.get("id"):
                return f"user:{user.get('id')}"
        
        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        
        # Handle forwarded IPs
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
        """
        Process the request with rate limiting.
        """
        # Skip rate limiting for whitelisted paths
        if any(request.url.path.startswith(path) for path in self.whitelist_paths):
            return await call_next(request)
        
        # Skip rate limiting for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        client_key = self.get_client_key(request)
        
        # Use lock to prevent race conditions
        lock = await self.get_lock(client_key)
        
        async with lock:
            now = time.time()
            request_times = self.requests[client_key]
            
            # Clean old requests
            self.clean_old_requests(request_times, now)
            
            # Get the limit for this endpoint
            limit = self.get_endpoint_limit(request.url.path)
            
            # Check if rate limit is exceeded
            if len(request_times) >= limit:
                # Calculate retry time
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
            
            # Add current request timestamp
            request_times.append(now)
            remaining = limit - len(request_times)
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers to the response
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(int(now + 60))
        
        return response