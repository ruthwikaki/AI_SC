"""
Base model for all SQLAlchemy models
Located at: /backend/app/models/base.py
"""
from sqlalchemy import Column, DateTime, String, Boolean, Integer
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declared_attr
from app.db.database import Base
import uuid

class BaseModel(Base):
    """Base model with common fields"""
    __abstract__ = True
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String, nullable=True)
    updated_by = Column(String, nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    deleted_by = Column(String, nullable=True)
    version = Column(Integer, default=1, nullable=False)
    
    @declared_attr
    def __tablename__(cls):
        """Generate table name from class name"""
        # Convert CamelCase to snake_case
        name = cls.__name__
        return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')
    
    def soft_delete(self, user_id: str = None):
        """Soft delete the record"""
        self.is_deleted = True
        self.deleted_at = func.now()
        self.deleted_by = user_id
    
    def restore(self):
        """Restore soft deleted record"""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update_from_dict(self, data: dict, user_id: str = None):
        """Update model from dictionary"""
        for key, value in data.items():
            if hasattr(self, key) and key not in ['id', 'created_at', 'created_by']:
                setattr(self, key, value)
        self.updated_by = user_id
        self.version += 1
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id})>"

class TimestampMixin:
    """Mixin for timestamp fields only"""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class AuditMixin(TimestampMixin):
    """Mixin for audit fields"""
    created_by = Column(String, nullable=True)
    updated_by = Column(String, nullable=True)
    version = Column(Integer, default=1, nullable=False)

class SoftDeleteMixin:
    """Mixin for soft delete functionality"""
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    deleted_by = Column(String, nullable=True)
    
    def soft_delete(self, user_id: str = None):
        self.is_deleted = True
        self.deleted_at = func.now()
        self.deleted_by = user_id
    
    def restore(self):
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None