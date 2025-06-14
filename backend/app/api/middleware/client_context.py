from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ClientContextMiddleware(BaseHTTPMiddleware):
    """Middleware to set client context for multi-tenant operations"""
    
    async def dispatch(self, request: Request, call_next):
        """Extract and set client context"""
        # Extract client context from headers or token
        client_id = request.headers.get("X-Client-ID")
        client_name = request.headers.get("X-Client-Name")
        
        # Store in request state
        request.state.client_id = client_id
        request.state.client_name = client_name
        
        # Log client context
        if client_id:
            logger.debug(f"Request from client: {client_id} ({client_name})")
        
        response = await call_next(request)
        return response


def get_client_context(request: Request) -> dict:
    """Get client context from request"""
    return {
        "client_id": getattr(request.state, "client_id", None),
        "client_name": getattr(request.state, "client_name", None)
    }


def require_client_context(request: Request) -> dict:
    """Get client context, raise error if not present"""
    context = get_client_context(request)
    if not context["client_id"]:
        raise HTTPException(
            status_code=400,
            detail="Client context required. Please provide X-Client-ID header."
        )
    return context