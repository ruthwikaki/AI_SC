from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional, Dict, Any
import time
from datetime import datetime

from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

class JWTAuthMiddleware:
    """
    Middleware for JWT authentication.
    
    This middleware handles JWT token verification but doesn't enforce authentication.
    The actual enforcement is done by the auth dependencies in the route handlers.
    This middleware just attaches user info to the request state if a valid token is provided.
    """
    
    def __init__(self):
        self.security = HTTPBearer(auto_error=False)
    
    async def __call__(self, request: Request, call_next):
        # Initialize user in request state
        request.state.user = None
        request.state.token = None
        request.state.token_payload = None
        
        # Try to get credentials
        credentials: Optional[HTTPAuthorizationCredentials] = None
        try:
            credentials = await self.security(request)
        except:
            # If extraction fails, continue without authentication
            pass
            
        if credentials:
            try:
                # Validate token
                payload = jwt.decode(
                    credentials.credentials, 
                    settings.jwt_secret_key, 
                    algorithms=[settings.jwt_algorithm]
                )
                
                # Check if token has expired
                if payload.get("exp") and time.time() > payload["exp"]:
                    logger.warning(f"Expired token received: {payload.get('sub', 'unknown')}")
                else:
                    # Store user info in request state
                    request.state.token = credentials.credentials
                    request.state.token_payload = payload
                    
                    # Basic user info from token
                    user_info = {
                        "id": payload.get("user_id"),
                        "username": payload.get("sub"),
                        "role": payload.get("role")
                    }
                    request.state.user = user_info
                    
                    logger.debug(f"Authenticated user: {user_info.get('username')}")
            
            except JWTError as e:
                logger.warning(f"Invalid token received: {str(e)}")
                # Continue without authentication
        
        # Continue with request processing
        response = await call_next(request)
        return response

class AdminOnlyMiddleware:
    """
    Middleware to restrict access to admin-only routes.
    
    This middleware checks if the current path starts with the admin prefix
    and enforces admin role authentication for those routes.
    """
    
    def __init__(self, admin_path_prefix: str = "/admin"):
        self.admin_path_prefix = admin_path_prefix
    
    async def __call__(self, request: Request, call_next):
        # Check if this is an admin route
        if request.url.path.startswith(self.admin_path_prefix):
            # Check if user is authenticated and has admin role
            user = getattr(request.state, "user", None)
            
            if not user:
                logger.warning(f"Unauthenticated access attempt to admin route: {request.url.path}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required for admin routes",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            if user.get("role") != "admin":
                logger.warning(f"Unauthorized access attempt to admin route by {user.get('username')}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin privileges required for this resource"
                )
        
        # Continue with request processing
        response = await call_next(request)
        return response