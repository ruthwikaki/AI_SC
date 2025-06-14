"""
User model for authentication and authorization
Located at: /backend/app/models/user.py
"""
from sqlalchemy import Column, String, Boolean, JSON, Text, ForeignKey, Table, Enum
from sqlalchemy.orm import relationship
import enum
from typing import Dict, Any

from app.models.base import BaseModel

# Association table for user roles
user_roles = Table(
    'user_roles',
    BaseModel.metadata,
    Column('user_id', String, ForeignKey('user.id'), primary_key=True),
    Column('role_id', String, ForeignKey('role.id'), primary_key=True)
)

# Association table for user permissions
user_permissions = Table(
    'user_permissions',
    BaseModel.metadata,
    Column('user_id', String, ForeignKey('user.id'), primary_key=True),
    Column('permission_id', String, ForeignKey('permission.id'), primary_key=True)
)

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"

class User(BaseModel):
    """User model with authentication and profile information"""
    __tablename__ = "user"
    
    # Authentication fields
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    
    # Profile fields
    full_name = Column(String, nullable=True)
    phone_number = Column(String, nullable=True)
    department = Column(String, nullable=True)
    job_title = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    
    # Status fields
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Role and permissions
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    permissions = relationship("Permission", secondary=user_permissions, back_populates="users")
    
    # Settings and preferences
    preferences = Column(JSON, default=dict, nullable=True)
    notification_settings = Column(JSON, default=dict, nullable=True)
    
    # Security fields
    last_login = Column(String, nullable=True)
    last_login_ip = Column(String, nullable=True)
    failed_login_attempts = Column(String, default=0, nullable=False)
    locked_until = Column(String, nullable=True)
    
    # API keys
    api_key = Column(String, unique=True, nullable=True, index=True)
    api_key_created_at = Column(String, nullable=True)
    
    # Relationships
    queries = relationship("NaturalLanguageQuery", back_populates="user", cascade="all, delete-orphan")
    dashboards = relationship("Dashboard", back_populates="user", cascade="all, delete-orphan")
    saved_charts = relationship("SavedChart", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    
    def has_permission(self, permission_name: str) -> bool:
        """Check if user has specific permission"""
        if self.is_superuser:
            return True
        
        # Check direct permissions
        for permission in self.permissions:
            if permission.name == permission_name:
                return True
        
        # Check role permissions
        for role in self.roles:
            for permission in role.permissions:
                if permission.name == permission_name:
                    return True
        
        return False
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role"""
        if self.is_superuser:
            return True
        
        if self.role.value == role_name:
            return True
        
        for role in self.roles:
            if role.name == role_name:
                return True
        
        return False
    
    def get_preferences(self) -> Dict[str, Any]:
        """Get user preferences with defaults"""
        defaults = {
            "theme": "light",
            "language": "en",
            "timezone": "UTC",
            "date_format": "YYYY-MM-DD",
            "chart_type": "line",
            "page_size": 20,
            "enable_notifications": True,
            "enable_tooltips": True,
            "compact_view": False
        }
        
        if self.preferences:
            defaults.update(self.preferences)
        
        return defaults
    
    def update_preferences(self, new_preferences: Dict[str, Any]):
        """Update user preferences"""
        current = self.get_preferences()
        current.update(new_preferences)
        self.preferences = current
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"

class Role(BaseModel):
    """Role model for RBAC"""
    __tablename__ = "role"
    
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    is_system = Column(Boolean, default=False, nullable=False)  # System roles cannot be deleted
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", secondary="role_permissions", back_populates="roles")

class Permission(BaseModel):
    """Permission model for fine-grained access control"""
    __tablename__ = "permission"
    
    name = Column(String, unique=True, nullable=False)
    resource = Column(String, nullable=False)  # e.g., "query", "dashboard", "user"
    action = Column(String, nullable=False)    # e.g., "create", "read", "update", "delete"
    description = Column(Text, nullable=True)
    
    # Relationships
    users = relationship("User", secondary=user_permissions, back_populates="permissions")
    roles = relationship("Role", secondary="role_permissions", back_populates="permissions")

# Association table for role permissions
role_permissions = Table(
    'role_permissions',
    BaseModel.metadata,
    Column('role_id', String, ForeignKey('role.id'), primary_key=True),
    Column('permission_id', String, ForeignKey('permission.id'), primary_key=True)
)

class AuditLog(BaseModel):
    """Audit log for tracking user actions"""
    __tablename__ = "audit_log"
    
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    action = Column(String, nullable=False)  # e.g., "login", "query", "export"
    resource_type = Column(String, nullable=True)  # e.g., "dashboard", "report"
    resource_id = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")