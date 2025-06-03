"""
Application entry point.
This module serves as the main entry point for the FastAPI application.
"""

import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
import jwt

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
        database_url = "postgresql://postgres:123456789@localhost:5432/AI_SC"
    
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
        database_url = "postgresql://postgres:123456789@localhost:5432/AI_SC"
    
    settings = FallbackSettings()

# Simple logger for now (instead of complex logger setup)
def log_info(message):
    print(f"[INFO] {datetime.now().isoformat()} - {message}")

def log_error(message):
    print(f"[ERROR] {datetime.now().isoformat()} - {message}")

# Test user for development
TEST_USER = {
    "email": "test@example.com",
    "username": "testuser",
    "password": "testpassword",
    "role": "user"
}

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
    
    log_info("Application startup complete (simplified mode)")
    
    yield
    
    # ===== Shutdown =====
    log_info("Shutting down application")
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
    "http://localhost:3000",
    "*"
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
    # Try database connection
    db_status = "not configured"
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(getattr(settings, 'database_url', ''))
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_status = "connected"
    except:
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "version": getattr(settings, 'api_version', '1.0.0'),
        "environment": getattr(settings, 'environment', 'development'),
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "database": db_status,
            "llm": "not configured",
            "cache": "not configured"
        }
    }

# API Routes
@app.get("/api/status")
async def api_status():
    """API status endpoint."""
    return {
        "api": "running",
        "version": getattr(settings, 'api_version', '1.0.0'),
        "environment": getattr(settings, 'environment', 'development'),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint for frontend connection."""
    return {
        "message": "Backend connection successful!",
        "timestamp": datetime.now().isoformat(),
        "frontend_connected": True,
        "cors_enabled": True
    }

# Authentication endpoints
@app.post("/api/auth/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint that returns JWT token."""
    if form_data.username == TEST_USER["email"] and form_data.password == TEST_USER["password"]:
        token = jwt.encode(
            {
                "sub": form_data.username,
                "user_id": "test-user-id",
                "role": TEST_USER["role"],
                "exp": datetime.utcnow() + timedelta(hours=24)
            },
            "secret-key",
            algorithm="HS256"
        )
        return {"access_token": token, "token_type": "bearer"}
    
    raise HTTPException(
        status_code=401,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )

@app.get("/api/auth/me")
async def get_current_user():
    """Get current user endpoint."""
    return {
        "id": "test-user-id",
        "username": TEST_USER["username"],
        "email": TEST_USER["email"],
        "role": TEST_USER["role"]
    }

# Placeholder endpoints
@app.get("/api/queries")
async def get_queries():
    """Placeholder for queries endpoint."""
    return {
        "queries": [],
        "total": 0,
        "page": 1,
        "per_page": 10
    }

@app.post("/api/queries")
async def create_query():
    """Placeholder for creating queries."""
    return {
        "id": "query-123",
        "status": "created",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics():
    """Placeholder for dashboard analytics."""
    return {
        "inventory": {
            "total_value": 1500000,
            "items_count": 350,
            "low_stock_alerts": 12
        },
        "orders": {
            "pending": 45,
            "processing": 23,
            "completed": 156
        },
        "suppliers": {
            "active": 28,
            "performance_score": 87.5
        }
    }

# Error handling
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    log_error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "message": str(exc) if getattr(settings, 'environment', 'development') == 'development' else "An error occurred",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Get port from settings or environment
    port = getattr(settings, 'port', int(os.environ.get("PORT", 8000)))
    
    log_info("Starting Supply Chain LLM Backend...")
    log_info(f"API Server: http://localhost:{port}")
    log_info(f"API Documentation: http://localhost:{port}/docs")
    log_info(f"Health Check: http://localhost:{port}/health")
    log_info(f"Test Endpoint: http://localhost:{port}/api/test")
    log_info("Test User: test@example.com / testpassword")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=getattr(settings, 'host', '0.0.0.0'),
        port=port,
        reload=getattr(settings, 'environment', 'development') == "development",
        workers=getattr(settings, 'uvicorn_workers', 1),
        log_level="info"
    )