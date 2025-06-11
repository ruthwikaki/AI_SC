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
from contextlib import asynccontextmanager

# Import ALL route modules
from app.api.routes import (
    auth, 
    queries, 
    visualizations, 
    database, 
    analytics, 
    admin,
    multi_tier,
    reports,
    settings,
    dashboards,
    suggestions,
    export,
    analytics_enhanced,
    reference_data
)

from app.api.middleware.auth import JWTAuthMiddleware, AdminOnlyMiddleware
from app.api.middleware.error_handler import ErrorHandlerMiddleware
from app.api.middleware.rate_limit import RateLimitMiddleware
from app.api.middleware.client_context import ClientContextMiddleware
from app.config import get_settings
from app.utils.logger import get_logger, setup_logging
from app.jobs.scheduler import start_scheduler, stop_scheduler

# Get settings
settings = get_settings()

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create database engine for health checks
engine = create_engine(settings.database_url, echo=False)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting Supply Chain AI-Powered API (version 2.0.0)")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"AI Model: {os.getenv('LLM_MODEL', 'deepseek-coder-v2:16b-lite-instruct-q4_0')}")
    logger.info(f"Ollama URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    
    # Start background scheduler if enabled
    if settings.enable_scheduler:
        start_scheduler()
        logger.info("Background scheduler started")
    
    # Log all loaded routes
    logger.info("API Routes loaded:")
    logger.info("  ✓ Authentication & Authorization")
    logger.info("  ✓ AI-Powered Natural Language Queries")
    logger.info("  ✓ Dynamic Visualizations")
    logger.info("  ✓ Advanced Analytics")
    logger.info("  ✓ Multi-tier Supply Chain Analysis")
    logger.info("  ✓ Custom Dashboards")
    logger.info("  ✓ Automated Reports")
    logger.info("  ✓ Data Export")
    logger.info("  ✓ System Settings")
    logger.info("  ✓ Admin Panel")
    logger.info("  ✓ Database Management")
    logger.info("  ✓ Reference Data")
    logger.info("  ✓ AI Suggestions")
    
    logger.info("="*60)
    logger.info("🚀 AI-Powered Supply Chain System Ready!")
    logger.info("="*60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Supply Chain AI-Powered API")
    
    # Stop scheduler if running
    if settings.enable_scheduler:
        stop_scheduler()
        logger.info("Background scheduler stopped")
    
    # Close database connections
    engine.dispose()
    logger.info("Database connections closed")

# Create FastAPI application
app = FastAPI(
    title="Supply Chain AI-Powered API",
    description="""
    AI-Powered Supply Chain Management API with Natural Language Processing
    
    Features:
    - 🤖 Natural language to SQL query generation using LLMs
    - 📊 Automatic visualization generation
    - 🔍 Advanced analytics and forecasting
    - 🕸️ Multi-tier supply chain network analysis
    - 📈 Custom dashboard builder
    - 🔄 Real-time data synchronization
    - 🔐 Enterprise-grade security
    
    Powered by Deepseek-Coder LLM for intelligent query understanding.
    """,
    version="2.0.0",
    docs_url=None,  # Custom docs URL
    redoc_url=None,  # Custom redoc URL
    lifespan=lifespan
)

# Add CORS middleware - configure based on environment
cors_origins = settings.cors_origins if hasattr(settings, 'cors_origins') else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)

# Add custom middleware in correct order
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(ClientContextMiddleware)

# Request tracking middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time and request ID to response headers"""
    import uuid
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Process request
    response = await call_next(request)
    
    # Add headers
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    # Log request
    logger.info(
        f"Request {request_id}: {request.method} {request.url.path} "
        f"completed in {process_time:.3f}s with status {response.status_code}"
    )
    
    return response

# Authentication middleware - apply after CORS
app.add_middleware(JWTAuthMiddleware)
app.add_middleware(AdminOnlyMiddleware, admin_path_prefix="/api/admin")

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Supply Chain AI-Powered API",
        "version": "2.0.0",
        "status": "running",
        "features": {
            "ai_queries": True,
            "natural_language": True,
            "visualizations": True,
            "multi_tier_analysis": True,
            "forecasting": True,
            "custom_dashboards": True
        },
        "endpoints": {
            "docs": "/api/docs",
            "health": "/api/health",
            "query": "/api/queries/execute"
        },
        "ai_model": os.getenv('LLM_MODEL', 'deepseek-coder-v2:16b-lite-instruct-q4_0')
    }

# Health check endpoint
@app.get("/api/health", tags=["system"])
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "environment": settings.environment,
        "components": {
            "api": "operational",
            "database": "checking...",
            "ai_service": "checking..."
        }
    }

# Enhanced database health check
@app.get("/api/health/db", tags=["system"])
async def health_db():
    """Database connectivity and status check"""
    try:
        with engine.connect() as conn:
            # Basic connectivity
            conn.execute(text("SELECT 1"))
            
            # Get table counts
            tables = {}
            for table in ['products', 'suppliers', 'inventory', 'orders']:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    tables[table] = result.scalar()
                except:
                    tables[table] = "error"
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat(),
            "tables": tables
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

# AI service health check
@app.get("/api/health/ai", tags=["system"])
async def health_ai():
    """AI service (Ollama) connectivity check"""
    import httpx
    
    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("LLM_MODEL", "deepseek-coder-v2:16b-lite-instruct-q4_0")
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check if Ollama is running
            response = await client.get(f"{ollama_url}/api/tags")
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                return {
                    "status": "healthy",
                    "ai_service": "connected",
                    "ollama_url": ollama_url,
                    "configured_model": model,
                    "model_available": model in model_names,
                    "available_models": model_names,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise Exception(f"Ollama returned status {response.status_code}")
                
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "ai_service": "disconnected",
                "error": str(e),
                "message": "Make sure Ollama is running and the model is loaded",
                "timestamp": datetime.now().isoformat()
            }
        )

# Include all routers with proper prefixes
# Core functionality
app.include_router(auth.router, prefix="/api")
app.include_router(queries.router, prefix="/api")
app.include_router(visualizations.router, prefix="/api")
app.include_router(database.router, prefix="/api")

# Analytics and insights
app.include_router(analytics.router, prefix="/api")
app.include_router(analytics_enhanced.router, prefix="/api")
app.include_router(reports.router, prefix="/api")

# Supply chain specific
app.include_router(multi_tier.router)  # Has its own /api prefix
app.include_router(dashboards.router)  # Has its own /api prefix

# Utilities and configuration
app.include_router(suggestions.router, prefix="/api")
app.include_router(export.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
app.include_router(reference_data.router, prefix="/api")

# Admin functionality
app.include_router(admin.router, prefix="/api")

# Custom OpenAPI documentation
@app.get("/api/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced styling"""
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title=f"{app.title} - Interactive Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="/favicon.ico"
    )

@app.get("/api/redoc", include_in_schema=False)
async def redoc_html():
    """ReDoc documentation"""
    return HTMLResponse(
        content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{app.title} - ReDoc</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
        </head>
        <body>
            <redoc spec-url="/api/openapi.json"></redoc>
            <script src="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"></script>
        </body>
        </html>
        """,
        status_code=200
    )

@app.get("/api/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """OpenAPI schema endpoint"""
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        servers=[
            {"url": "/", "description": "Current server"},
            {"url": "http://localhost:8000", "description": "Local development"},
            {"url": "https://api.supplychain-ai.com", "description": "Production"}
        ]
    )

# Global error handlers
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc):
    """Handle 404 errors with helpful message"""
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "code": "not_found",
                "message": "The requested resource was not found",
                "path": request.url.path,
                "suggestion": "Check /api/docs for available endpoints"
            },
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, 'request_id', 'unknown')
            }
        }
    )

# Catch-all OPTIONS handler for CORS preflight
@app.options("/{path:path}")
async def options_handler(request: Request):
    """Handle CORS preflight requests"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# API statistics endpoint
@app.get("/api/stats", tags=["system"])
async def api_statistics():
    """Get API usage statistics"""
    return {
        "version": app.version,
        "routes": len(app.routes),
        "uptime": "calculating...",
        "total_requests": "tracking...",
        "timestamp": datetime.now().isoformat()
    }

# Export the app for ASGI servers
api_app = app

if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "app.api.server:api_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )