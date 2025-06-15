
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import time
from typing import Callable
import os

from app.config import get_settings
from app.utils.logger import setup_logger
from app.db.database import engine, init_db
from app.api.middleware.error_handler import add_exception_handlers
from app.api.middleware.rate_limit import RateLimitMiddleware
from app.api.middleware.auth import JWTAuthMiddleware, AdminOnlyMiddleware
from app.api.middleware.client_context import ClientContextMiddleware

# Import all routers
from app.api.routes import (
    auth, queries, analytics, analytics_enhanced, dashboards,
    database, export, forecasting, multi_tier, reference_data,
    reports, settings as settings_router, suggestions, visualizations, admin
)

logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    logger.info("Starting up Supply Chain AI Backend...")
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
        
        # Initialize schema cache
        from app.db.schema.schema_discovery import SchemaDiscovery
        from app.db.database import SessionLocal
        db = SessionLocal()
        schema_discovery = SchemaDiscovery(db)
        # schema_discovery.discover_schema()  # Pre-cache schema - Method not implemented yet
        db.close()
        logger.info("Schema cache initialized")
        
        # Initialize analytics engine
        if get_settings().enable_analytics:
            from app.analytics.inventory_optimization.forecast_engine import ForecastEngine
            forecast_engine = ForecastEngine()
            logger.info("Analytics engine initialized")
        
        # Initialize multi-tier network
        from app.multiTier.supplier_mapping.network_builder import NetworkBuilder
        network_builder = NetworkBuilder()
        logger.info("Multi-tier network initialized")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    
    # Close database connections
    engine.dispose()
    
    # Stop background jobs
    try:
        from app.jobs.scheduler import JobScheduler
        scheduler = JobScheduler()
        scheduler.shutdown()
    except:
        pass
    
    # Clear caches
    try:
        from app.cache.cache_invalidation import CacheInvalidator
        invalidator = CacheInvalidator()
        await invalidator.clear_all()
    except:
        pass
    
    logger.info("Shutdown complete")

# Global app instance
app = None

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    global app
    
    if app is not None:
        return app
    
    settings = get_settings()
    
    # Create FastAPI instance
    app = FastAPI(
        title=settings.APP_NAME,
        description="AI-powered Supply Chain Management System",
        version=settings.VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
        docs_url="/api/docs" if settings.DEBUG else None,
        redoc_url="/api/redoc" if settings.DEBUG else None,
        openapi_url="/api/openapi.json" if settings.DEBUG else None,
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "syntaxHighlight.theme": "obsidian",
            "tryItOutEnabled": True,
        }
    )
    
    # Add middlewares in correct order (outermost to innermost)
    
    # CORS middleware (should be first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173", "*"],
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
        expose_headers=["X-Total-Count", "X-Page", "X-Per-Page"],
    )
    
    # Trusted host middleware
    if settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.cors_origins
        )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middlewares
    app.add_middleware(ClientContextMiddleware)
    app.add_middleware(JWTAuthMiddleware)
    app.add_middleware(RateLimitMiddleware)
    
    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Add exception handlers
    add_exception_handlers(app)
    
    # Mount static files if upload directory exists
    if os.path.exists(settings.upload_dir):
        app.mount("/static", StaticFiles(directory=settings.upload_dir), name="static")
    
    # Root endpoints
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "message": "Supply Chain AI Backend API",
            "version": settings.VERSION,
            "status": "operational",
            "documentation": {
                "interactive": "/api/docs" if settings.DEBUG else "Disabled",
                "redoc": "/api/redoc" if settings.DEBUG else "Disabled",
                "openapi": "/api/openapi.json" if settings.DEBUG else "Disabled"
            }
        }
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint for monitoring"""
        from app.db.database import check_database_health
        
        db_healthy = await check_database_health()
        
        health_status = {
            "status": "healthy" if db_healthy else "degraded",
            "version": settings.VERSION,
            "environment": settings.environment,
            "database": "connected" if db_healthy else "disconnected",
            "timestamp": time.time()
        }
        
        # Check optional services
        if settings.redis_url:
            try:
                from app.cache.query_cache import QueryCache
                cache = QueryCache()
                await cache.ping()
                health_status["cache"] = "connected"
            except:
                health_status["cache"] = "disconnected"
                health_status["status"] = "degraded"
        
        if settings.llm_provider:
            try:
                from app.services.llm_service import get_llm_service
                llm = get_llm_service()
                health_status["llm"] = "connected"
            except:
                health_status["llm"] = "disconnected"
                health_status["status"] = "degraded"
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        return JSONResponse(content=health_status, status_code=status_code)
    
    @app.get("/metrics", tags=["Monitoring"])
    async def metrics():
        """Basic metrics endpoint"""
        from app.utils.metrics import get_application_metrics
        return await get_application_metrics()
    
    # Include all routers
    app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
    app.include_router(queries.router, prefix="/api/queries", tags=["Natural Language Queries"])
    app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
    app.include_router(analytics_enhanced.router, prefix="/api/analytics-enhanced", tags=["Enhanced Analytics"])
    app.include_router(dashboards.router, prefix="/api/dashboards", tags=["Dashboards"])
    app.include_router(database.router, prefix="/api/database", tags=["Database Management"])
    app.include_router(export.router, prefix="/api/export", tags=["Data Export"])
    app.include_router(forecasting.router, prefix="/api/forecasting", tags=["Forecasting"])
    app.include_router(multi_tier.router, prefix="/api/multi-tier", tags=["Multi-Tier Supply Chain"])
    app.include_router(reference_data.router, prefix="/api/reference-data", tags=["Reference Data"])
    app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
    app.include_router(settings_router.router, prefix="/api/settings", tags=["User Settings"])
    app.include_router(suggestions.router, prefix="/api/suggestions", tags=["Query Suggestions"])
    app.include_router(visualizations.router, prefix="/api/visualizations", tags=["Data Visualizations"])
    app.include_router(admin.router, prefix="/api/admin", tags=["Administration"])
    
    # Custom error handlers
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "message": f"The requested URL {request.url.path} was not found on this server.",
                "path": request.url.path
            }
        )
    
    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc):
        logger.error(f"Internal server error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Please try again later.",
                "request_id": request.state.request_id if hasattr(request.state, 'request_id') else None
            }
        )
    
    # API versioning preparation
    @app.get("/api/version", tags=["Version"])
    async def api_version():
        return {
            "api_version": "1.0",
            "app_version": settings.VERSION,
            "minimum_client_version": "1.0",
            "deprecated_endpoints": [],
            "new_features": [
                "Natural Language Queries",
                "Multi-Tier Supply Chain Visualization",
                "Advanced Forecasting Models",
                "Real-time Analytics"
            ]
        }
    
    logger.info(f"FastAPI app created successfully in {settings.environment} mode")
    return app

# Create app instance for uvicorn
app = create_app()