"""
Client context middleware for multi-tenant support
"""
from fastapi import Request
from typing import Callable
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ClientContextMiddleware:
    """Middleware for handling client context"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable):
        # Extract client ID from header or query params
        client_id = (
            request.headers.get("X-Client-ID") or 
            request.query_params.get("client_id") or
            "default"
        )
        
        # Store in request state
        request.state.client_id = client_id
        
        # Add to response headers
        response = await call_next(request)
        response.headers["X-Client-ID"] = client_id
        
        return response

# Helper functions
async def get_client_context(request: Request) -> Optional[str]:
    """Get the client ID from the request context."""
    return getattr(request.state, "client_id", None)

async def get_connection_id(request: Request) -> Optional[str]:
    """Get the connection ID from the request context."""
    return getattr(request.state, "connection_id", None)

async def get_client_settings(request: Request) -> Optional[Dict[str, Any]]:
    """Get the client settings from the request context."""
    return getattr(request.state, "client_settings", None)

async def is_feature_enabled(request: Request, feature_name: str) -> bool:
    """Check if a feature is enabled for the client."""
    settings = await get_client_settings(request)
    if not settings or "features" not in settings:
        return False
    
    return settings["features"].get(feature_name, False)
