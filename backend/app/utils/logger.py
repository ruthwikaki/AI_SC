"""
Logging configuration and utilities
Located at: /backend/app/utils/logger.py
"""
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name or __name__)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
    
    # Set level
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return setup_logger(name)

# Create a default logger for the module
logger = setup_logger(__name__)
