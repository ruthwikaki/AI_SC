"""
Base model configuration
All models should inherit from this Base
"""

# Import Base from database to ensure single source of truth
from app.db.database import Base

# Re-export for convenience
__all__ = ['Base']
