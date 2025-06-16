"""FastAPI application setup with proper imports"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Import routers
from .routes import (
    auth, queries, analytics, visualizations, 
    dashboards, forecasting, multi_tier, reports,
    admin, database, export, reference_data, 
    settings, suggestions, analytics_enhanced
)

# Import database setup
from ..db.database import engine, Base
from ..db.init_db import init_db

# Setup logging
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("Starting up...")
    Base.metadata.create_all(bind=engine)
    init_db()
    yield
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="AI Supply Chain API",
    description="AI-powered supply chain management system",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(queries.router, prefix="/api/queries", tags=["queries"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(analytics_enhanced.router, prefix="/api/analytics/enhanced", tags=["analytics-enhanced"])
app.include_router(visualizations.router, prefix="/api/visualizations", tags=["visualizations"])
app.include_router(dashboards.router, prefix="/api/dashboards", tags=["dashboards"])
app.include_router(forecasting.router, prefix="/api/forecasting", tags=["forecasting"])
app.include_router(multi_tier.router, prefix="/api/multi-tier", tags=["multi-tier"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(database.router, prefix="/api/database", tags=["database"])
app.include_router(export.router, prefix="/api/export", tags=["export"])
app.include_router(reference_data.router, prefix="/api/reference-data", tags=["reference-data"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
app.include_router(suggestions.router, prefix="/api/suggestions", tags=["suggestions"])

@app.get("/")
async def root():
    return {"message": "AI Supply Chain API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
