"""
Base model configuration for SQLAlchemy
"""
from sqlalchemy.ext.declarative import declarative_base

# Create the declarative base for SQLAlchemy
Base = declarative_base()

# Base mixin for SQLAlchemy models
class BaseModelMixin:
    """Base mixin for common model functionality"""
    
    @classmethod
    def create(cls, **kwargs):
        """Create a new instance"""
        instance = cls(**kwargs)
        return instance
    
    def update(self, **kwargs):
        """Update model attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

# Alias for backward compatibility
BaseMixin = BaseModelMixin