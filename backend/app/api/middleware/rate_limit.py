from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, Optional, Tuple, Any
import time
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simplified rate limiting middleware.
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}
        self.locks = {}
    
    async def get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for the given key."""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        return self.locks[key]
    
    def get_client_key(self, request: Request) -> str:
        """Get a unique key to identify the client."""
        # Try to get from user state
        user = getattr(request.state, "user", None)
        if user and user.get("client_id"):
            return f"client:{user.get('client_id')}"
        
        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"
    
    async def dispatch(self, request: Request, call_next):
        client_key = self.get_client_key(request)
        lock = await self.get_lock(client_key)
        
        async with lock:
            now = time.time()
            
            # Initialize or cleanup expired requests
            if client_key not in self.requests:
                self.requests[client_key] = []
            
            # Remove requests older than 60 seconds
            self.requests[client_key] = [
                req_time for req_time in self.requests[client_key]
                if now - req_time < 60
            ]
            
            # Check if rate limit is exceeded
            if len(self.requests[client_key]) >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for {client_key}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": {
                            "code": "rate_limit_exceeded",
                            "message": "Too many requests, please try again later"
                        }
                    },
                    headers={"Retry-After": "60"}
                )
            
            # Add this request
            self.requests[client_key].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - len(self.requests[client_key]))
        )
        
        return response