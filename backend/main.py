#!/usr/bin/env python3
"""
Main entry point for the Supply Chain AI Backend
Located at: /backend/main.py
"""
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Dict, Any
import asyncio

from app.config import get_settings
from app.utils.logger import setup_logger
from app.db.database import engine, get_db, init_db
from app.api.server import create_app
from app.db.init_db import init_db

# Setup logging
logger = setup_logger(__name__)

# Test user for development
TEST_USER = {
    "email": "test@example.com",
    "password": "testpassword",
    "username": "testuser",
    "role": "admin"
}

async def create_test_user(db: Session):
    """Create test user for development"""
    try:
        from app.models.user import User
        from app.security.password_utils import get_password_hash
        
        # Check if user exists
        existing_user = db.query(User).filter(User.email == TEST_USER["email"]).first()
        if not existing_user:
            user = User(
                email=TEST_USER["email"],
                username=TEST_USER["username"],
                hashed_password=get_password_hash(TEST_USER["password"]),
                role=TEST_USER["role"],
                is_active=True,
                is_superuser=True
            )
            db.add(user)
            db.commit()
            logger.info(f"Test user created: {TEST_USER['email']}")
        else:
            logger.info(f"Test user already exists: {TEST_USER['email']}")
    except Exception as e:
        logger.error(f"Failed to create test user: {e}")
        db.rollback()

async def initialize_application():
    """Initialize the application with all required setup"""
    try:
        settings = get_settings()
        logger.info(f"Initializing {settings.app_name} v{settings.version}")
        
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        
        # Create test user in development
        if settings.environment == "development":
            db = next(get_db())
            await create_test_user(db)
            db.close()
        
        # Initialize cache if Redis is configured
        if settings.redis_url:
            try:
                from app.cache.cache_invalidation import CacheInvalidator
                cache_invalidator = CacheInvalidator()
                await cache_invalidator.initialize()
                logger.info("Cache system initialized")
            except Exception as e:
                logger.warning(f"Cache initialization failed: {e}")
        
        # Initialize scheduler for background jobs
        if settings.enable_analytics:
            try:
                from app.jobs.scheduler import JobScheduler
                scheduler = JobScheduler()
                scheduler.start()
                logger.info("Job scheduler started")
            except Exception as e:
                logger.warning(f"Scheduler initialization failed: {e}")
        
        # Initialize LLM service
        try:
            from app.services.llm_service import get_llm_service
            llm_service = get_llm_service()
            logger.info("LLM service initialized")
        except Exception as e:
            logger.warning(f"LLM service initialization failed: {e}")
        
        logger.info("Application initialization complete")
        
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        raise

def main():
    """Main function to start the application"""
    try:
        # Get settings
        settings = get_settings()
        
        # Run initialization
        asyncio.run(initialize_application())
        
        # Get or create app
        app = create_app()
        
        logger.info(f"Starting {settings.app_name} on {settings.host}:{settings.port}")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info(f"Database: {settings.database_url.split('@')[1] if '@' in settings.database_url else 'local'}")
        
        if settings.debug:
            logger.info(f"API Documentation: http://{settings.host}:{settings.port}/api/docs")
        
        # Run server
        uvicorn.run(
            "app.api.server:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level=settings.log_level.lower(),
            access_log=True,
            use_colors=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application shutdown requested")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Performing cleanup...")
        try:
            from app.jobs.scheduler import JobScheduler
            scheduler = JobScheduler()
            scheduler.shutdown()
        except:
            pass

if __name__ == "__main__":
    main()