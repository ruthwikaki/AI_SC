"""
User and authentication related models
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Column, String, Boolean, DateTime, ForeignKey, Text, Integer,
    UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.models.base import Base, BaseModelMixin


class User(Base, BaseModelMixin):
    """User model for authentication and authorization"""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    roles = relationship("UserRole", secondary="user_role_assignments", back_populates="users")
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    activities = relationship("UserActivity", back_populates="user", cascade="all, delete-orphan")
    created_queries = relationship("NaturalLanguageQuery", foreign_keys="NaturalLanguageQuery.created_by", back_populates="created_by_user")
    dashboards = relationship("Dashboard", foreign_keys="Dashboard.created_by", back_populates="created_by_user")
    
    def __repr__(self):
        return f"<User(username={self.username}, email={self.email})>"
    
    @property
    def full_name(self):
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username
    
    @property
    def is_locked(self):
        """Check if account is locked"""
        if self.locked_until:
            return datetime.utcnow() < self.locked_until
        return False


class UserSession(Base, BaseModelMixin):
    """User session tracking"""
    __tablename__ = 'user_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    token = Column(String(500), unique=True, nullable=False)
    refresh_token = Column(String(500), unique=True)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    refresh_expires_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_activity = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='sessions')
    
    def __repr__(self):
        return f"<UserSession(user_id={self.user_id}, active={self.is_active})>"
    
    @property
    def is_expired(self):
        """Check if session is expired"""
        from datetime import timezone
        return datetime.now(timezone.utc) > self.expires_at


class UserRole(Base, BaseModelMixin):
    """User roles for RBAC"""
    __tablename__ = 'user_roles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(50), unique=True, nullable=False)
    display_name = Column(String(100))
    description = Column(Text)
    is_system = Column(Boolean, default=False)  # System roles can't be deleted
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    permissions = relationship('UserPermission', secondary='role_permissions', back_populates='roles')
    users = relationship('User', secondary='user_role_assignments', back_populates='roles')
    
    def __repr__(self):
        return f"<UserRole(name={self.name})>"


class UserPermission(Base, BaseModelMixin):
    """Permissions for RBAC"""
    __tablename__ = 'user_permissions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), unique=True, nullable=False)
    resource = Column(String(50), nullable=False)  # e.g., 'dashboard', 'report', 'user'
    action = Column(String(50), nullable=False)    # e.g., 'create', 'read', 'update', 'delete'
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    roles = relationship('UserRole', secondary='role_permissions', back_populates='permissions')
    
    def __repr__(self):
        return f"<UserPermission(name={self.name}, resource={self.resource}, action={self.action})>"


class RolePermission(Base):
    """Association table for role-permission relationship"""
    __tablename__ = 'role_permissions'
    
    role_id = Column(UUID(as_uuid=True), ForeignKey('user_roles.id'), primary_key=True)
    permission_id = Column(UUID(as_uuid=True), ForeignKey('user_permissions.id'), primary_key=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class UserRoleAssignment(Base):
    """Association table for user-role relationship"""
    __tablename__ = 'user_role_assignments'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True)
    role_id = Column(UUID(as_uuid=True), ForeignKey('user_roles.id'), primary_key=True)
    assigned_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    assigned_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))


class UserProfile(Base, BaseModelMixin):
    """Extended user profile information"""
    __tablename__ = 'user_profiles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), unique=True, nullable=False)
    full_name = Column(String(200))
    phone = Column(String(20))
    department = Column(String(100))
    job_title = Column(String(100))
    location = Column(String(100))
    timezone = Column(String(50), default='UTC')
    language = Column(String(10), default='en')
    avatar_url = Column(String(500))
    bio = Column(Text)
    preferences = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='profile', uselist=False)
    
    def __repr__(self):
        return f"<UserProfile(user_id={self.user_id}, full_name={self.full_name})>"


class UserActivity(Base, BaseModelMixin):
    """User activity tracking"""
    __tablename__ = 'user_activities'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    activity_type = Column(String(50), nullable=False)  # 'login', 'query', 'export', etc.
    resource_type = Column(String(50))
    resource_id = Column(UUID(as_uuid=True))
    details = Column(JSONB, default={})
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='activities')
    
    def __repr__(self):
        return f"<UserActivity(user_id={self.user_id}, type={self.activity_type})>"

# Alias for backward compatibility
Role = UserRole
Permission = UserPermission
