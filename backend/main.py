"""
Application entry point.

This module serves as the main entry point for the FastAPI application.
"""

import os
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.server import app as api_app
from app.config import get_settings
from app.utils.logger import setup_logging, get_logger
from app.security.rbac_manager import initialize_roles
from app.llm.controller.active_model_manager import initialize_models
from app.llm.controller.health_checker import (
    start_health_checker,
    cleanup_audit_logger,
    stop_health_checker,
)

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Define startup and shutdown context
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle context manager for FastAPI.
    
    This handles startup and shutdown events.
    """
    # ===== Startup =====
    logger.info(f"Starting Supply Chain LLM API (version {settings.api_version})")
    logger.info(f"Environment: {settings.environment}")
    
    # Setup logging
    setup_logging()
    
    # Initialize RBAC roles
    initialize_roles()
    
    # Initialize LLM models
    await initialize_models()
    
    # Start health checker
    await start_health_checker()
    
    logger.info("Application startup complete")
    
    yield
    
    # ===== Shutdown =====
    logger.info("Shutting down application")
    
    # Stop health checker
    await stop_health_checker()
    
    # Cleanup audit logger
    await cleanup_audit_logger()
    
    logger.info("Application shutdown complete")

# Create the main application
app = FastAPI(
    title="Supply Chain LLM API",
    description="API for the Supply Chain LLM SaaS platform",
    version=settings.api_version,
    lifespan=lifespan,
)

# Mount the API application
app.mount("/api", api_app)

# Add root route
@app.get("/")
async def root():
    """Root route that redirects to API documentation."""
    return {
        "app": "Supply Chain LLM API",
        "version": settings.api_version,
        "environment": settings.environment,
        "docs_url": "/api/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.api_version,
        "environment": settings.environment,
    }

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.environment == "development",
        workers=settings.uvicorn_workers
    )