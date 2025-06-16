"""Database configuration with lazy imports to prevent circular dependencies"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Create Base first - no imports needed
Base = declarative_base()

# Lazy load settings to prevent circular import
def get_database_url():
    """Get database URL with lazy loading of settings"""
    try:
        from ..config import get_settings
        settings = get_settings()
        return settings.DATABASE_URL
    except ImportError:
        # Fallback for testing or when config not available
        return os.getenv("DATABASE_URL", "sqlite:///./test.db")

# Create engine with lazy-loaded URL
def create_db_engine():
    """Create database engine with proper configuration"""
    database_url = get_database_url()
    
    # Fix for SQLite
    if database_url.startswith("sqlite"):
        engine = create_engine(database_url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(database_url)
    
    return engine

# Create engine
engine = create_db_engine()

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

__all__ = ['Base', 'engine', 'SessionLocal', 'get_db']
