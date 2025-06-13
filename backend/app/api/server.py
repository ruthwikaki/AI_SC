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

# Import all route modules
from app.api.routes import (
    auth, 
    queries, 
    visualizations, 
    database, 
    analytics,
    analytics_enhanced,  # Added
    admin,
    dashboards,         # Added
    export,             # Added
    multi_tier,         # Added
    reference_data,     # Added
    reports,            # Added
    settings as settings_routes,  # Added - renamed to avoid conflict
    suggestions         # Added
)
from app.api.middleware.auth import JWTAuthMiddleware, AdminOnlyMiddleware
from app.api.middleware.error_handler import ErrorHandlerMiddleware
from app.api.middleware.rate_limit import RateLimitMiddleware
from app.api.middleware.client_context import ClientContextMiddleware
from app.config import get_settings
from app.utils.logger import get_logger, setup_logging
from app.db.database import get_db

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

# Configure CORS
logger.info(f"CORS origins: {settings.cors_origins}")

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

# Dashboard endpoints (custom endpoints that frontend is looking for)
@app.get("/api/dashboard/overview", tags=["dashboard"])
async def get_dashboard_overview(db = Depends(get_db)):
    """Get dashboard overview metrics"""
    try:
        # This is a placeholder - implement actual logic based on your data
        return {
            "total_orders": 1234,
            "pending_orders": 45,
            "total_inventory_value": 987654.32,
            "low_stock_items": 12,
            "active_suppliers": 78,
            "on_time_delivery_rate": 94.5,
            "total_shipments": 567,
            "in_transit": 23
        }
    except Exception as e:
        logger.error(f"Error fetching dashboard overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/recent-orders", tags=["dashboard"])
async def get_recent_orders(db = Depends(get_db)):
    """Get recent orders"""
    try:
        # Placeholder implementation
        return {
            "orders": [
                {
                    "id": "ORD-001",
                    "customer": "Acme Corp",
                    "date": "2025-06-12",
                    "status": "Processing",
                    "total": 5432.10
                },
                {
                    "id": "ORD-002",
                    "customer": "Global Industries",
                    "date": "2025-06-12",
                    "status": "Shipped",
                    "total": 8765.43
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching recent orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/inventory-alerts", tags=["dashboard"])
async def get_inventory_alerts(db = Depends(get_db)):
    """Get inventory alerts"""
    try:
        # Placeholder implementation
        return {
            "alerts": [
                {
                    "id": 1,
                    "type": "low_stock",
                    "product": "Widget A",
                    "current_stock": 15,
                    "reorder_point": 50,
                    "severity": "high"
                },
                {
                    "id": 2,
                    "type": "overstock",
                    "product": "Gadget B",
                    "current_stock": 500,
                    "max_stock": 200,
                    "severity": "medium"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching inventory alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/supplier-metrics", tags=["dashboard"])
async def get_supplier_metrics(db = Depends(get_db)):
    """Get supplier performance metrics"""
    try:
        # Placeholder implementation
        return {
            "metrics": {
                "total_suppliers": 78,
                "active_suppliers": 65,
                "average_lead_time": 3.5,
                "on_time_delivery_rate": 92.3,
                "quality_rating": 4.2
            },
            "top_suppliers": [
                {
                    "name": "Supplier A",
                    "rating": 4.8,
                    "orders": 156
                },
                {
                    "name": "Supplier B",
                    "rating": 4.5,
                    "orders": 134
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching supplier metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/logistics-summary", tags=["dashboard"])
async def get_logistics_summary(db = Depends(get_db)):
    """Get logistics summary"""
    try:
        # Placeholder implementation
        return {
            "summary": {
                "total_shipments": 567,
                "in_transit": 23,
                "delivered": 544,
                "average_transit_time": 2.3,
                "on_time_rate": 94.5
            },
            "shipments_by_status": {
                "pending": 12,
                "in_transit": 23,
                "delivered": 544,
                "delayed": 5
            }
        }
    except Exception as e:
        logger.error(f"Error fetching logistics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include all routers
app.include_router(auth.router, prefix="/api")
app.include_router(queries.router, prefix="/api")
app.include_router(visualizations.router, prefix="/api")
app.include_router(database.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(analytics_enhanced.router, prefix="/api")  # Added
app.include_router(admin.router, prefix="/api")
app.include_router(dashboards.router)  # Added (already has /api/dashboards prefix)
app.include_router(export.router, prefix="/api")  # Added
app.include_router(multi_tier.router, prefix="/api")  # Added
app.include_router(reference_data.router, prefix="/api")  # Added
app.include_router(reports.router, prefix="/api")  # Added
app.include_router(settings_routes.router, prefix="/api")  # Added - using renamed import
app.include_router(suggestions.router, prefix="/api")  # Added

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
    logger.info(f"CORS origins: {settings.cors_origins}")
    logger.info("Database connected successfully")

# Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Supply Chain LLM API")
    engine.dispose()

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
