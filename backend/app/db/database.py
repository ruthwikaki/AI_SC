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

# Base class for all models
Base = declarative_base()

# Database URL construction
def get_database_url() -> str:
    """Construct database URL from settings"""
    return settings.database_url

# Engine configuration based on environment
def get_engine_config():
    """Get engine configuration based on environment"""
    if settings.environment == "production":
        return {
            "pool_size": 20,
            "max_overflow": 40,
            "pool_timeout": 30,
            "pool_recycle": 1800,  # 30 minutes
            "pool_pre_ping": True,
            "echo": False,
            "poolclass": QueuePool
        }
    elif settings.environment == "testing":
        return {
            "poolclass": NullPool,  # No connection pooling for tests
            "echo": False
        }
    else:  # development
        return {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 3600,  # 1 hour
            "pool_pre_ping": True,
            "echo": settings.debug,
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
    expire_on_commit=False  # Prevent expiration of objects after commit
)

# Create scoped session for thread safety
ScopedSession = scoped_session(SessionLocal)

# Event listeners for performance monitoring (optional)
@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log slow queries in development"""
    conn.info.setdefault('query_start_time', []).append(time.time())
    if settings.debug:
        logger.debug(f"Start Query: {statement[:100]}...")

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log query execution time"""
    total = time.time() - conn.info['query_start_time'].pop(-1)
    if settings.debug and total > 1.0:  # Log queries slower than 1 second
        logger.warning(f"Slow Query ({total:.3f}s): {statement[:100]}...")

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
    
    @staticmethod
    def get_table_sizes():
        """Get size of all tables"""
        with engine.connect() as conn:
            result = conn.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
                    pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
            """)
            return result.fetchall()

# Import all models to ensure they're registered with Base
def import_all_models():
    """Import all models to register them with SQLAlchemy"""
    try:
        from app.models import (
            user, query, visualization, supply_chain, analytics
        )
        logger.info("All models imported successfully")
    except ImportError as e:
        logger.warning(f"Some models could not be imported: {e}")

# Initialize models on module load
import_all_models()
