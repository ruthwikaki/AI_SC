from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from jose import JWTError, jwt
from typing import Optional

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
# Lazy load settings to avoid circular import
_settings = None

def get_settings_cached():
    global _settings
    if _settings is None:
        from ..config import get_settings
        _settings = get_settings()
    return _settings

settings = property(lambda self: get_settings_cached())


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """JWT Authentication middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.public_paths = [
            "/",
            "/api/health",
            "/api/health/db",
            "/api/docs",
            "/api/openapi.json",
            "/api/auth/login",
            "/api/auth/register",
            "/api/auth/token",
            "/api/version"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process the request with JWT validation"""
        # Skip auth for public paths and OPTIONS requests
        if request.url.path in self.public_paths or request.method == "OPTIONS":
            return await call_next(request)
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing authentication token"}
            )
        
        token = auth_header.split(" ")[1]
        
        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm]
            )
            
            # Store user info in request state
            request.state.user_id = payload.get("user_id")
            request.state.username = payload.get("sub")
            request.state.role = payload.get("role", "user")
            
        except JWTError as e:
            logger.warning(f"JWT validation failed: {e}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authentication token"}
            )
        
        # Process request
        response = await call_next(request)
        return response


class AdminOnlyMiddleware(BaseHTTPMiddleware):
    """Middleware to restrict admin routes"""
    
    def __init__(self, app, admin_path_prefix: str = "/api/admin"):
        super().__init__(app)
        self.admin_path_prefix = admin_path_prefix
    
    async def dispatch(self, request: Request, call_next):
        """Check if user has admin access for admin routes"""
        # Only check admin routes
        if not request.url.path.startswith(self.admin_path_prefix):
            return await call_next(request)
        
        # OPTIONS requests are allowed
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Check if user role is admin
        user_role = getattr(request.state, "role", None)
        if user_role != "admin":
            return JSONResponse(
                status_code=403,
                content={"detail": "Admin access required"}
            )
        
        return await call_next(request)

# Alias for backward compatibility
AuthMiddleware = JWTAuthMiddleware
