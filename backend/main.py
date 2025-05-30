"""
Application entry point.
This module serves as the main entry point for the FastAPI application.
"""

import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import config with better error handling
try:
    from config import get_settings
    settings = get_settings()
    print("Configuration loaded successfully")
except ImportError as e:
    print(f"WARNING: Could not import config: {str(e)}")
    # Fallback settings
    class FallbackSettings:
        app_name = "Supply Chain LLM API"
        api_version = "1.0.0"
        environment = "development"
        host = "0.0.0.0"
        port = 8000
        uvicorn_workers = 1
        cors_origins = ["http://localhost:3001", "http://127.0.0.1:3001"]
    
    settings = FallbackSettings()
except Exception as e:
    print(f"WARNING: Config error: {str(e)}")
    # Use fallback settings
    class FallbackSettings:
        app_name = "Supply Chain LLM API"
        api_version = "1.0.0"
        environment = "development"
        host = "0.0.0.0"
        port = 8000
        uvicorn_workers = 1
        cors_origins = ["http://localhost:3001", "http://127.0.0.1:3001"]
    
    settings = FallbackSettings()

# Simple logger for now (instead of complex logger setup)
def log_info(message):
    print(f"[INFO] {datetime.now().isoformat()} - {message}")

def log_error(message):
    print(f"[ERROR] {datetime.now().isoformat()} - {message}")

# Define startup and shutdown context
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle context manager for FastAPI.
    This handles startup and shutdown events.
    """
    # ===== Startup =====
    log_info(f"Starting {getattr(settings, 'app_name', 'Supply Chain LLM API')}")
    log_info(f"Version: {getattr(settings, 'api_version', '1.0.0')}")
    log_info(f"Environment: {getattr(settings, 'environment', 'development')}")
    
    # TODO: Initialize components when they're ready
    # - Initialize RBAC roles
    # - Initialize LLM models  
    # - Start health checker
    
    log_info("Application startup complete (simplified mode)")
    
    yield
    
    # ===== Shutdown =====
    log_info("Shutting down application")
    
    # TODO: Cleanup when components are ready
    # - Stop health checker
    # - Cleanup audit logger
    
    log_info("Application shutdown complete")

# Create the main application
app = FastAPI(
    title=getattr(settings, 'app_name', 'Supply Chain LLM API'),
    description="API for the Supply Chain LLM SaaS platform",
    version=getattr(settings, 'api_version', '1.0.0'),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
cors_origins = getattr(settings, 'cors_origins', [
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:3000"
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route
@app.get("/")
async def root():
    """Root route that provides API information."""
    return {
        "app": getattr(settings, 'app_name', 'Supply Chain LLM API'),
        "version": getattr(settings, 'api_version', '1.0.0'),
        "environment": getattr(settings, 'environment', 'development'),
        "docs_url": "/docs",
        "api_prefix": "/api",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": getattr(settings, 'api_version', '1.0.0'),
        "environment": getattr(settings, 'environment', 'development'),
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "database": "not configured",  # Will be configured later
            "llm": "not configured",       # Will be configured later
            "cache": "not configured"      # Will be configured later
        }
    }

# API Routes (Simplified versions)
@app.get("/api/status")
async def api_status():
    """API status endpoint."""
    return {
        "api": "running",
        "version": getattr(settings, 'api_version', '1.0.0'),
        "environment": getattr(settings, 'environment', 'development'),
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/",
            "/health", 
            "/api/status",
            "/api/test",
            "/api/queries",
            "/api/analytics",
            "/docs",
            "/redoc"
        ]
    }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint for frontend connection."""
    return {
        "message": "Backend connection successful!",
        "timestamp": datetime.now().isoformat(),
        "frontend_connected": True,
        "cors_enabled": True,
        "config_loaded": hasattr(settings, 'app_name'),
        "environment": getattr(settings, 'environment', 'development')
    }

# Placeholder API routes for your frontend
@app.get("/api/queries")
async def get_queries():
    """Placeholder for queries endpoint."""
    return {
        "queries": [],
        "message": "Queries endpoint working (placeholder)",
        "timestamp": datetime.now().isoformat(),
        "note": "This will be replaced with actual query functionality"
    }

@app.post("/api/queries")
async def create_query():
    """Placeholder for creating queries."""
    return {
        "message": "Query creation endpoint working (placeholder)",
        "timestamp": datetime.now().isoformat(),
        "note": "This will handle natural language query processing"
    }

@app.get("/api/analytics")
async def get_analytics():
    """Placeholder for analytics endpoint."""
    return {
        "analytics": {
            "inventory": {"status": "placeholder", "data": []},
            "supplier": {"status": "placeholder", "data": []},
            "logistics": {"status": "placeholder", "data": []}
        },
        "message": "Analytics endpoint working (placeholder)",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/analytics/inventory")
async def get_inventory_analytics():
    """Placeholder for inventory analytics."""
    return {
        "inventory_analytics": {
            "abc_analysis": {"status": "placeholder"},
            "safety_stock": {"status": "placeholder"},
            "forecasting": {"status": "placeholder"}
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/visualizations")
async def get_visualizations():
    """Placeholder for visualizations endpoint."""
    return {
        "visualizations": [],
        "chart_types": ["bar", "line", "pie", "heatmap", "network"],
        "message": "Visualizations endpoint working (placeholder)",
        "timestamp": datetime.now().isoformat()
    }

# Authentication placeholder endpoints
@app.post("/api/auth/login")
async def login():
    """Placeholder for login endpoint."""
    return {
        "message": "Login endpoint (placeholder)",
        "note": "Authentication will be implemented later",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/auth/me")
async def get_current_user():
    """Placeholder for current user endpoint."""
    return {
        "user": {"id": 1, "username": "demo", "role": "admin"},
        "message": "User info endpoint (placeholder)",
        "timestamp": datetime.now().isoformat()
    }

# Error handling
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    log_error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "message": str(exc),
        "timestamp": datetime.now().isoformat(),
        "environment": getattr(settings, 'environment', 'development')
    }

if __name__ == "__main__":
    # Get port from settings or environment
    port = getattr(settings, 'port', int(os.environ.get("PORT", 8000)))
    
    log_info("Starting Supply Chain LLM Backend...")
    log_info(f"API Server: http://localhost:{port}")
    log_info(f"API Documentation: http://localhost:{port}/docs")
    log_info(f"Health Check: http://localhost:{port}/health")
    log_info(f"Test Endpoint: http://localhost:{port}/api/test")
    log_info("Note: Running in simplified mode for development")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=getattr(settings, 'host', '0.0.0.0'),
        port=port,
        reload=getattr(settings, 'environment', 'development') == "development",
        workers=getattr(settings, 'uvicorn_workers', 1),
        log_level="info"
    )
