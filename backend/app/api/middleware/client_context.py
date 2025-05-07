from fastapi import Request
from typing import Optional, Dict, Any
import json

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class ClientContextMiddleware:
    """
    Middleware for handling client context resolution.
    
    This middleware determines the client context for each request, which is 
    needed for accessing the correct database, enforcing client isolation,
    and ensuring proper multi-tenancy.
    """
    
    async def __call__(self, request: Request, call_next):
        # Initialize client context in request state
        request.state.client_id = None
        request.state.connection_id = None
        request.state.client_settings = None
        
        # Try to resolve client context from multiple sources, in order of priority
        
        # 1. From authenticated user
        user = getattr(request.state, "user", None)
        if user and user.get("client_id"):
            request.state.client_id = user.get("client_id")
        
        # 2. From request header
        if not request.state.client_id:
            client_id = request.headers.get("X-Client-ID")
            if client_id:
                # In a real implementation, validate that this client ID exists
                # and that the API key or token has access to it
                request.state.client_id = client_id
        
        # 3. From query parameter (lowest priority, mostly for testing)
        if not request.state.client_id:
            client_id = request.query_params.get("client_id")
            if client_id:
                request.state.client_id = client_id
        
        # Try to get connection ID for database access
        connection_id = request.headers.get("X-Connection-ID") or request.query_params.get("connection_id")
        if connection_id:
            request.state.connection_id = connection_id
        
        # If we have a client ID, load client settings
        if request.state.client_id:
            # In a real implementation, you would load these from database or cache
            # For now, we'll use a simple mapping
            if request.state.client_id == "client-1":
                request.state.client_settings = {
                    "name": "Acme Corporation",
                    "domain": "manufacturing",
                    "default_connection_id": "conn-1",
                    "features": {
                        "multi_tier_enabled": True,
                        "advanced_analytics": True
                    }
                }
            elif request.state.client_id == "client-2":
                request.state.client_settings = {
                    "name": "TechStart Inc",
                    "domain": "electronics",
                    "default_connection_id": "conn-2",
                    "features": {
                        "multi_tier_enabled": False,
                        "advanced_analytics": False
                    }
                }
        
        # If connection ID wasn't specified but we have client settings, use default
        if not request.state.connection_id and request.state.client_settings:
            request.state.connection_id = request.state.client_settings.get("default_connection_id")
        
        # Log client context
        if request.state.client_id:
            logger.debug(
                f"Client context: client_id={request.state.client_id}, "
                f"connection_id={request.state.connection_id}"
            )
        
        # Continue with request processing
        response = await call_next(request)
        return response

# Helper function to be used in route handlers
async def get_client_context(request: Request) -> Optional[str]:
    """
    Get the client ID from the request context.
    
    This helper function can be used in route handlers to get the client ID.
    """
    return getattr(request.state, "client_id", None)

async def get_connection_id(request: Request) -> Optional[str]:
    """
    Get the connection ID from the request context.
    
    This helper function can be used in route handlers to get the connection ID.
    """
    return getattr(request.state, "connection_id", None)

async def get_client_settings(request: Request) -> Optional[Dict[str, Any]]:
    """
    Get the client settings from the request context.
    
    This helper function can be used in route handlers to get client-specific settings.
    """
    return getattr(request.state, "client_settings", None)

async def is_feature_enabled(request: Request, feature_name: str) -> bool:
    """
    Check if a feature is enabled for the client.
    
    This helper function can be used in route handlers to check if a specific
    feature is enabled for the client.
    """
    settings = await get_client_settings(request)
    if not settings or "features" not in settings:
        return False
    
    return settings["features"].get(feature_name, False)