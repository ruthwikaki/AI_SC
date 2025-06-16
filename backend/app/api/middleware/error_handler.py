from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from jose import JWTError
from typing import Union, Dict, Any
import time
import traceback
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
# Lazy load settings to avoid circular import
_settings = None

def get_settings_cached():
    global _settings
    if _settings is None:
        from ..config import get_settings
        _settings = get_settings()
    return _settings

settings = property(lambda self: get_settings_cached())

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for centralized error handling.
    
    This middleware catches exceptions that weren't handled by route handlers,
    logs them, and returns appropriate error responses.
    """
    
    async def dispatch(self, request: Request, call_next):
        try:
            # Try to process the request
            return await call_next(request)
        
        except ValidationError as exc:
            # Handle request validation errors
            logger.warning(f"Validation error: {str(exc)}")
            return self._create_error_response(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="validation_error",
                message="Invalid request data",
                details=exc.errors() if hasattr(exc, "errors") else None
            )
            
        except JWTError as exc:
            # Handle JWT authentication errors
            logger.warning(f"Authentication error: {str(exc)}")
            return self._create_error_response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                error_code="authentication_error",
                message="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        except Exception as exc:
            # Handle unexpected exceptions
            error_id = f"err_{int(time.time())}"
            
            # Log the full error with traceback
            logger.error(
                f"Unhandled exception ({error_id}): {str(exc)}\n"
                f"URL: {request.method} {request.url}\n"
                f"{traceback.format_exc()}"
            )
            
            # In development, include the error details
            details = None
            if settings.environment == "development":
                details = {
                    "exception": str(exc),
                    "traceback": traceback.format_exc().split("\n")
                }
            
            # Return a generic server error
            return self._create_error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_code="server_error",
                message="An unexpected error occurred",
                details=details,
                meta={"error_id": error_id}
            )
    
    def _create_error_response(
        self, 
        status_code: int, 
        error_code: str, 
        message: str, 
        details: Any = None,
        meta: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> JSONResponse:
        """Create a standardized error response."""
        content = {
            "error": {
                "code": error_code,
                "message": message
            }
        }
        
        if details:
            content["error"]["details"] = details
            
        if meta:
            content["error"]["meta"] = meta
        
        return JSONResponse(
            status_code=status_code,
            content=content,
            headers=headers
        )


def add_exception_handlers(app: FastAPI):
    """Add exception handlers to the FastAPI app"""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation error",
                "message": "Request validation failed",
                "detail": exc.errors()
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Bad request",
                "message": str(exc)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "detail": str(exc) if app.debug else None
            }
        )
