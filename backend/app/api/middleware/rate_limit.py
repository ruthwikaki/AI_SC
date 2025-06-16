import time
import asyncio
from collections import defaultdict, deque
from typing import Dict, Deque
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute: int = 60, cleanup_interval: int = 300):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.cleanup_interval = cleanup_interval
        self.requests: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=requests_per_minute))
        self.locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next):
        """Process the request with rate limiting"""
        # Skip rate limiting for health checks and OPTIONS requests
        if request.url.path in ["/", "/api/health", "/api/health/db"] or request.method == "OPTIONS":
            return await call_next(request)
        
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        
        # Periodic cleanup
        await self._cleanup_old_entries()
        
        # Check rate limit
        async with self.locks[client_ip]:
            now = time.time()
            minute_ago = now - 60
            
            # Remove old requests
            while self.requests[client_ip] and self.requests[client_ip][0] < minute_ago:
                self.requests[client_ip].popleft()
            
            # Check if limit exceeded
            if len(self.requests[client_ip]) >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "rate_limit_exceeded",
                            "message": "Too many requests. Please try again later.",
                            "retry_after": 60
                        }
                    },
                    headers={
                        "Retry-After": "60",
                        "X-RateLimit-Limit": str(self.requests_per_minute),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(minute_ago + 60))
                    }
                )
            
            # Add current request
            self.requests[client_ip].append(now)
            remaining = self.requests_per_minute - len(self.requests[client_ip])
        
        # Process request and add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(minute_ago + 60))
        
        return response
    
    async def _cleanup_old_entries(self):
        """Periodically clean up old entries to prevent memory leak"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = now
        minute_ago = now - 60
        
        # Clean up IPs that haven't made requests recently
        ips_to_remove = []
        for ip, timestamps in list(self.requests.items()):
            if not timestamps or timestamps[-1] < minute_ago:
                ips_to_remove.append(ip)
        
        for ip in ips_to_remove:
            del self.requests[ip]
            if ip in self.locks:
                del self.locks[ip]
        
        if ips_to_remove:
            logger.info(f"Cleaned up rate limit data for {len(ips_to_remove)} IPs")