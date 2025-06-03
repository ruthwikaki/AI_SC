from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, HTMLResponse
import time
import os
from datetime import datetime
import logging
from sqlalchemy import text
from sqlalchemy import create_engine

from app.api.routes import auth, queries, visualizations, database, analytics, admin
from app.api.middleware.auth import JWTAuthMiddleware, AdminOnlyMiddleware
from app.api.middleware.error_handler import ErrorHandlerMiddleware
from app.api.middleware.rate_limit import RateLimitMiddleware
from app.api.middleware.client_context import ClientContextMiddleware
from app.config import get_settings
from app.utils.logger import get_logger, setup_logging

# Get settings
settings = get_settings()

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create database engine for health checks
engine = create_engine(settings.database_url, echo=False)

# Create FastAPI application
app = FastAPI(
    title="Supply Chain LLM API",
    description="API for the Supply Chain LLM SaaS platform",
    version="1.0.0",
    docs_url=None,  # Disable default docs URL
    redoc_url=None  # Disable default redoc URL
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(JWTAuthMiddleware)
app.add_middleware(AdminOnlyMiddleware, admin_path_prefix="/api/admin")
app.add_middleware(RateLimitMiddleware)
app.add_middleware(ClientContextMiddleware)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint that redirects to API documentation."""
    return {
        "message": "Supply Chain LLM API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health",
        "status": "running"
    }

# Health check endpoint
@app.get("/api/health", tags=["system"])
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "environment": settings.environment,
    }

# Database health check endpoint
@app.get("/api/health/db", tags=["system"])
async def health_db():
    """
    Database health check endpoint to verify database connectivity.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Include routers
app.include_router(auth.router, prefix="/api")
app.include_router(queries.router, prefix="/api")
app.include_router(visualizations.router, prefix="/api")
app.include_router(database.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(admin.router, prefix="/api")

# Custom OpenAPI docs
@app.get("/api/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title=f"{app.title} - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
    )

@app.get("/api/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

# Error handlers
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "code": "not_found",
                "message": "The requested resource was not found",
                "path": request.url.path
            }
        }
    )

# Startup event handler
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting Supply Chain LLM API (version 1.0.0)")
    logger.info(f"Environment: {settings.environment}")
    logger.info("Database connected successfully")

# Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Supply Chain LLM API")
    await engine.dispose()

# Export the app for ASGI servers (like Uvicorn)
api_app = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.server:api_app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )