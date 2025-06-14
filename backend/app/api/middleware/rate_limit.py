"""
Rate limiting middleware
"""
from fastapi import Request, HTTPException, status
from typing import Callable, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RateLimitMiddleware:
    """Simple in-memory rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, list] = {}
    
    async def __call__(self, request: Request, call_next: Callable):
        # Skip rate limiting for certain paths
        skip_paths = ["/health", "/api/health", "/metrics"]
        if request.url.path in skip_paths:
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"
        
        # Initialize or get request timestamps for this client
        now = datetime.now()
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        # Remove timestamps older than 1 minute
        minute_ago = now - timedelta(minutes=1)
        self.request_counts[client_ip] = [
            ts for ts in self.request_counts[client_ip] 
            if ts > minute_ago
        ]
        
        # Check rate limit
        if len(self.request_counts[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Add current request timestamp
        self.request_counts[client_ip].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(self.request_counts[client_ip])
        )
        
        return response
