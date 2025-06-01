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
    if settings.DATABASE_URL:
        return settings.DATABASE_URL
    
    return (
        f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@"
        f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )

# Engine configuration based on environment
def get_engine_config():
    """Get engine configuration based on environment"""
    if settings.ENVIRONMENT == "production":
        return {
            "pool_size": 20,
            "max_overflow": 40,
            "pool_timeout": 30,
            "pool_recycle": 1800,  # 30 minutes
            "pool_pre_ping": True,
            "echo": False,
            "poolclass": QueuePool
        }
    elif settings.ENVIRONMENT == "testing":
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
            "echo": settings.DB_ECHO,
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

# Event listeners for performance monitoring
@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log slow queries in development"""
    conn.info.setdefault('query_start_time', []).append(time.time())
    if settings.LOG_SLOW_QUERIES:
        logger.debug(f"Start Query: {statement}")

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log query execution time"""
    total = time.time() - conn.info['query_start_time'].pop(-1)
    if settings.LOG_SLOW_QUERIES and total > settings.SLOW_QUERY_THRESHOLD:
        logger.warning(f"Slow Query ({total:.3f}s): {statement}")

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

# Async session support (optional, for future use)
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    
    async_engine = create_async_engine(
        get_database_url().replace('postgresql://', 'postgresql+asyncpg://'),
        **{k: v for k, v in get_engine_config().items() if k not in ['poolclass']}
    )
    
    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async def get_async_db() -> AsyncSession:
        """Async database dependency"""
        async with AsyncSessionLocal() as session:
            yield session
            
except ImportError:
    logger.info("Async database support not available")
    async_engine = None
    AsyncSessionLocal = None
    get_async_db = None

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
    def truncate_all():
        """Truncate all tables (use with caution!)"""
        with engine.begin() as conn:
            # Disable foreign key checks
            conn.execute("SET session_replication_role = 'replica';")
            
            # Get all table names
            tables = Base.metadata.sorted_tables
            
            # Truncate each table
            for table in tables:
                conn.execute(f"TRUNCATE TABLE {table.name} CASCADE;")
            
            # Re-enable foreign key checks
            conn.execute("SET session_replication_role = 'origin';")
            
        logger.info("All database tables truncated")
    
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
    
    @staticmethod
    def vacuum_analyze():
        """Run VACUUM ANALYZE on all tables"""
        with engine.connect() as conn:
            conn.execute("VACUUM ANALYZE;")
        logger.info("VACUUM ANALYZE completed")

# Import all models to ensure they're registered with Base
def import_all_models():
    """Import all models to register them with SQLAlchemy"""
    from app.models import (
        user, query, visualization, supply_chain, analytics
    )
    logger.info("All models imported successfully")

# Initialize models on module load
import_all_models()

