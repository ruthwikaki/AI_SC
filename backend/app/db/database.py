"""
Database connection and session management
"""

import os
from typing import Generator, Optional
from contextlib import contextmanager
import logging
import time

from sqlalchemy import create_engine, event, pool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.engine import Engine

from app.config import get_settings
settings = get_settings()

logger = logging.getLogger(__name__)

# Base class for all models - declared here, not imported
Base = declarative_base()

# Database URL construction
def get_database_url() -> str:
    """Construct database URL from settings"""
    return settings.DATABASE_URL

# Engine configuration based on environment
def get_engine_config():
    """Get engine configuration based on environment"""
    if settings.ENVIRONMENT == "production":
        return {
            "pool_size": 20,
            "max_overflow": 40,
            "pool_timeout": 30,
            "pool_recycle": 1800,
            "pool_pre_ping": True,
            "echo": False,
            "poolclass": QueuePool
        }
    elif settings.ENVIRONMENT == "testing":
        return {
            "poolclass": NullPool,
            "echo": False
        }
    else:  # development
        return {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 3600,
            "pool_pre_ping": True,
            "echo": settings.DEBUG,
            "poolclass": QueuePool
        }

# Create engine
engine = create_engine(
    get_database_url(),
    **get_engine_config()
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

# Create scoped session for thread safety
ScopedSession = scoped_session(SessionLocal)

# Database dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """
    Database dependency for FastAPI endpoints.
    Creates a new database session for each request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Context manager for database sessions
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Useful for background tasks and scripts.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# Database health check
def check_database_connection() -> bool:
    """Check if database is accessible"""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

# Database utilities
class DatabaseManager:
    """Database management utilities"""
    
    @staticmethod
    def create_all():
        """Create all tables"""
        Base.metadata.create_all(bind=engine)
        logger.info("All database tables created")
    
    @staticmethod
    def drop_all():
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(bind=engine)
        logger.info("All database tables dropped")

# Simple init_db function for main.py
def init_db():
    """Initialize database tables"""
    # Import all models to register them with Base
    try:
        from app.models import (
            user, query, visualization, supply_chain, analytics, extended_models
        )
        logger.info("All models imported successfully")
    except ImportError as e:
        logger.warning(f"Some models could not be imported: {e}")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
