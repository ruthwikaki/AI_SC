"""
User-related database models
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, ForeignKey,
    Table, Text, JSON, BigInteger, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.orm import relationship, backref

from app.models.base import Base

# Association tables
role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),
    Column('permission_id', UUID(as_uuid=True), ForeignKey('permissions.id', ondelete='CASCADE'), primary_key=True),
    Column('granted_at', DateTime(timezone=True), default=datetime.utcnow),
    Column('granted_by', UUID(as_uuid=True), ForeignKey('users.id'))
)

user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),
    Column('assigned_at', DateTime(timezone=True), default=datetime.utcnow),
    Column('assigned_by', UUID(as_uuid=True), ForeignKey('users.id'))
)

class User(Base):
    """User model for authentication and profile management"""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    role = Column(String(50), nullable=False, default='user')
    department = Column(String(100))
    phone = Column(String(50))
    avatar_url = Column(Text)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    email_verified_at = Column(DateTime(timezone=True))
    last_login = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))
    user_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    updated_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    sessions = relationship('UserSession', back_populates='user', cascade='all, delete-orphan')
    preferences = relationship('UserPreference', back_populates='user', uselist=False, cascade='all, delete-orphan')
    password_reset_tokens = relationship('PasswordResetToken', back_populates='user', cascade='all, delete-orphan')
    roles = relationship(
        "Role",
        secondary="user_roles",
        back_populates="users",
        primaryjoin="User.id==user_roles.c.user_id",
        secondaryjoin="Role.id==user_roles.c.role_id"
    )
    
    # Self-referential relationships
    created_by_user = relationship('User', foreign_keys=[created_by], remote_side=[id])
    updated_by_user = relationship('User', foreign_keys=[updated_by], remote_side=[id])
    
    # Other relationships
    queries = relationship('NaturalLanguageQuery', back_populates='user', cascade='all, delete-orphan')
    saved_queries = relationship('SavedQuery', back_populates='user', cascade='all, delete-orphan')
    created_charts = relationship('Chart', back_populates='created_by_user', cascade='all, delete-orphan')
    created_dashboards = relationship('Dashboard', back_populates='created_by_user', cascade='all, delete-orphan')
    audit_logs = relationship('AuditLog', back_populates='user', foreign_keys='AuditLog.user_id')
    
    # New Analytics and Reporting relationships
    analytics_metrics = relationship("AnalyticsMetric", back_populates="creator", foreign_keys="AnalyticsMetric.created_by")
    export_jobs = relationship("ExportJob", back_populates="user", foreign_keys="ExportJob.user_id")
    reports = relationship("Report", back_populates="user", foreign_keys="Report.user_id")
    scheduled_reports = relationship("ScheduledReport", back_populates="user", foreign_keys="ScheduledReport.user_id")
    notification_settings = relationship("NotificationSetting", back_populates="user", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, username={self.username})>"
    
    @property
    def full_name(self):
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has a specific role"""
        return any(role.name == role_name for role in self.roles)
    
    def has_permission(self, resource: str, action: str) -> bool:
        """Check if user has a specific permission"""
        for role in self.roles:
            for permission in role.permissions:
                if permission.resource == resource and permission.action == action:
                    return True
        return False


class UserSession(Base):
    """User session management for JWT tokens"""
    __tablename__ = 'user_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    token_hash = Column(String(255), nullable=False, unique=True, index=True)
    refresh_token_hash = Column(String(255))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    refresh_expires_at = Column(DateTime(timezone=True))
    ip_address = Column(INET)
    user_agent = Column(Text)
    device_id = Column(String(255))
    is_active = Column(Boolean, default=True)
    revoked_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='sessions')
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, expires_at={self.expires_at})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if session is valid"""
        return self.is_active and not self.is_expired and self.revoked_at is None


class PasswordResetToken(Base):
    """Password reset token management"""
    __tablename__ = 'password_reset_tokens'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    token_hash = Column(String(255), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used_at = Column(DateTime(timezone=True))
    ip_address = Column(INET)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='password_reset_tokens')
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if token is valid"""
        return not self.is_expired and self.used_at is None


class UserPreference(Base):
    """User preferences and settings"""
    __tablename__ = 'user_preferences'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, unique=True)
    theme = Column(String(50), default='light')
    language = Column(String(10), default='en')
    timezone = Column(String(50), default='UTC')
    date_format = Column(String(20), default='MM/DD/YYYY')
    number_format = Column(String(20), default='en-US')
    default_chart_type = Column(String(50), default='bar')
    dashboard_layout = Column(JSONB, default={})
    notification_preferences = Column(JSONB, default={})
    ui_preferences = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='preferences')
    
    def __repr__(self):
        return f"<UserPreference(user_id={self.user_id}, theme={self.theme})>"


class Role(Base):
    """Roles for role-based access control"""
    __tablename__ = 'roles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(50), unique=True, nullable=False)
    display_name = Column(String(100))
    description = Column(Text)
    is_system = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    permissions = relationship('Permission', secondary=role_permissions, back_populates='roles')
    users = relationship('User', secondary=user_roles, back_populates='roles')
    
    def __repr__(self):
        return f"<Role(name={self.name}, display_name={self.display_name})>"


class Permission(Base):
    """Permissions for fine-grained access control"""
    __tablename__ = 'permissions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    resource = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)
    display_name = Column(String(200))
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    roles = relationship(
        "Role",
        secondary="user_roles",
        back_populates="users",
        primaryjoin="User.id==user_roles.c.user_id",
        secondaryjoin="Role.id==user_roles.c.role_id"
    )
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('resource', 'action', name='uq_permission_resource_action'),
    )
    
    def __repr__(self):
        return f"<Permission(resource={self.resource}, action={self.action})>"


class AuditLog(Base):
    """Audit log for tracking system changes"""
    __tablename__ = 'audit_logs'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(255))
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    ip_address = Column(INET)
    user_agent = Column(Text)
    user_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship('User', back_populates='audit_logs', foreign_keys=[user_id])
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, resource_type={self.resource_type})>"