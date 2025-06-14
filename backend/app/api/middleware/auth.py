"""
Authentication middleware
"""
from fastapi import Request, HTTPException, status
from fastapi.security.utils import get_authorization_scheme_param
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class AuthMiddleware:
    """Middleware for handling authentication"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable):
        # Skip auth for public endpoints
        public_paths = [
            "/",
            "/health",
            "/api/health",
            "/api/docs",
            "/api/openapi.json",
            "/api/auth/token",
            "/api/auth/register"
        ]
        
        if request.url.path in public_paths:
            return await call_next(request)
        
        # For other endpoints, just pass through for now
        # In production, you would validate JWT tokens here
        response = await call_next(request)
        return response


class JWTAuthMiddleware:
    """JWT Authentication middleware"""
    
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, request: Request, call_next: Callable):
        # Get token from header
        authorization = request.headers.get("Authorization")
        scheme, token = get_authorization_scheme_param(authorization)
        
        # Store token in request state for later use
        request.state.token = token if scheme.lower() == "bearer" else None
        
        response = await call_next(request)
        return response


class AdminOnlyMiddleware:
    """Middleware to restrict admin routes"""
    
    def __init__(self, app, admin_path_prefix: str = "/admin"):
        self.app = app
        self.admin_path_prefix = admin_path_prefix
    
    async def __call__(self, request: Request, call_next: Callable):
        # Check if this is an admin route
        if request.url.path.startswith(self.admin_path_prefix):
            # In a real app, check if user is admin
            # For now, just pass through
            pass
        
        response = await call_next(request)
        return response
