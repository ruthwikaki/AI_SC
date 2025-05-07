"""
Logging utilities.

This module provides functions for consistent logging throughout the application.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import traceback

from app.config import get_settings

# Get settings
settings = get_settings()

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log levels mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

def setup_logging() -> None:
    """
    Set up logging configuration.
    
    This function configures the logging system based on application settings.
    """
    # Get log level from settings
    log_level_name = settings.log_level.lower() if hasattr(settings, 'log_level') else "info"
    log_level = LOG_LEVELS.get(log_level_name, DEFAULT_LOG_LEVEL)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Set log level for external libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    
    # Add file handler if log_to_file is enabled
    if getattr(settings, 'log_to_file', False):
        log_dir = getattr(settings, 'log_directory', 'logs')
        
        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        
        # Add file handler to root logger
        logging.getLogger().addHandler(file_handler)
    
    # Log initial message
    logging.info(f"Logging initialized at level: {log_level_name.upper()}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class JSONLogFormatter(logging.Formatter):
    """Custom formatter for structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra attributes
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        return json.dumps(log_data)

def setup_json_logging() -> None:
    """Configure logging to output JSON format (for production environments)."""
    # Get log level from settings
    log_level_name = settings.log_level.lower() if hasattr(settings, 'log_level') else "info"
    log_level = LOG_LEVELS.get(log_level_name, DEFAULT_LOG_LEVEL)
    
    # Create JSON formatter
    json_formatter = JSONLogFormatter()
    
    # Configure handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add JSON handler
    root_logger.addHandler(console_handler)
    
    # Set log level for external libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    
    # Log initial message
    logging.info("JSON logging initialized")

def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    context: Dict[str, Any] = None
) -> None:
    """
    Log a message with additional context.
    
    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        context: Additional context to include in the log
    """
    # Default empty context
    ctx = context or {}
    
    # Get log method based on level
    log_method = getattr(logger, level.lower(), logger.info)
    
    # Create extra dict with context
    extra = {"extra": ctx}
    
    # Log with extra context
    log_method(message, extra=extra)

def log_request(
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    client_id: Optional[str] = None,
    user_id: Optional[str] = None,
    duration_ms: Optional[float] = None
) -> None:
    """
    Log an API request.
    
    Args:
        request_id: Unique request ID
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        client_id: Optional client ID
        user_id: Optional user ID
        duration_ms: Optional request duration in milliseconds
    """
    # Get logger
    logger = get_logger("api.request")
    
    # Create context
    context = {
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code
    }
    
    # Add optional fields
    if client_id:
        context["client_id"] = client_id
    
    if user_id:
        context["user_id"] = user_id
    
    if duration_ms:
        context["duration_ms"] = duration_ms
    
    # Determine log level based on status code
    if status_code >= 500:
        level = "error"
    elif status_code >= 400:
        level = "warning"
    else:
        level = "info"
    
    # Log the request
    log_with_context(
        logger=logger,
        level=level,
        message=f"{method} {path} {status_code}",
        context=context
    )

def log_exception(
    exc: Exception,
    request_id: Optional[str] = None,
    client_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> None:
    """
    Log an exception with context.
    
    Args:
        exc: Exception instance
        request_id: Optional request ID
        client_id: Optional client ID
        user_id: Optional user ID
    """
    # Get logger
    logger = get_logger("api.exception")
    
    # Create context
    context = {
        "exception_type": exc.__class__.__name__,
        "exception_message": str(exc)
    }
    
    # Add optional fields
    if request_id:
        context["request_id"] = request_id
    
    if client_id:
        context["client_id"] = client_id
    
    if user_id:
        context["user_id"] = user_id
    
    # Log the exception
    log_with_context(
        logger=logger,
        level="error",
        message=f"Exception: {exc.__class__.__name__}: {str(exc)}",
        context=context
    )
    
    # Also log the traceback
    logger.error("Traceback:", exc_info=True)