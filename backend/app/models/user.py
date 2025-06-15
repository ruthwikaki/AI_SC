"""


User and authentication related models


"""





from datetime import datetime


from typing import Optional


from uuid import uuid4





from sqlalchemy import (


    Table,


    Column, String, Boolean, DateTime, ForeignKey, Text, Integer,


    UniqueConstraint, Index, BigInteger)


from sqlalchemy.dialects.postgresql import UUID, JSONB


from sqlalchemy.orm import relationship





from app.models.base import Base





# Association tables for many-to-many relationships


role_permissions = Table(


    'role_permissions',


    Base.metadata,


    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),


    Column('permission_id', UUID(as_uuid=True), ForeignKey('permissions.id', ondelete='CASCADE'), primary_key=True),


    Column('granted_at', DateTime(timezone=True), default=datetime.utcnow)


,


    extend_existing=True


)





user_roles = Table(


    'user_roles',


    Base.metadata,


    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),


    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),


    Column('assigned_at', DateTime(timezone=True), default=datetime.utcnow)


,


    extend_existing=True


)








# Removed incorrect import








class User(Base):


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


    preferences = relationship('UserPreference', back_populates='user', uselist=False, cascade='all, delete-orphan')


    roles = relationship('Role', secondary='user_roles', back_populates='users')


    audit_logs = relationship('AuditLog', back_populates='user', foreign_keys='AuditLog.user_id')





    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")


    roles = relationship("Role", secondary=user_roles, back_populates="users")


    # profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan") # Removed - UserProfile 
    saved_queries = relationship("SavedQuery", back_populates="user", cascade="all, delete-orphan")
    saved_charts = relationship("SavedChart", back_populates="user", cascade="all, delete-orphan")
    query_records = relationship("Query", back_populates="user", cascade="all, delete-orphan")
    created_charts = relationship("Chart", foreign_keys="Chart.created_by", back_populates="created_by_user", cascade="all, delete-orphan")
        # activities = relationship  # Commented out to avoid conflict with audit_logs("AuditLog", back_populates="user", cascade="all, delete-orphan")


    queries = relationship("NaturalLanguageQuery", back_populates="user", cascade="all, delete-orphan")
    query_records = relationship("Query", back_populates="user", cascade="all, delete-orphan")


    created_dashboards = relationship("Dashboard", foreign_keys="Dashboard.created_by", back_populates="created_by_user", cascade="all, delete-orphan")


    


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








class UserSession(Base):


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








class PasswordResetToken(Base):


    """Password reset token model"""


    __tablename__ = 'password_reset_tokens'


    


    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)


    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)


    token = Column(String(500), unique=True, nullable=False)


    expires_at = Column(DateTime(timezone=True), nullable=False)


    used = Column(Boolean, default=False)


    used_at = Column(DateTime(timezone=True))


    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


    


    # Relationships


    user = relationship('User', backref='password_reset_tokens')


    


    def __repr__(self):


        return f"<PasswordResetToken(user_id={self.user_id}, used={self.used})>"


    


    @property


    def is_expired(self):


        """Check if token is expired"""


        from datetime import timezone


        return datetime.now(timezone.utc) > self.expires_at


    


    @property


    def is_valid(self):


        """Check if token is valid (not used and not expired)"""


        return not self.used and not self.is_expired








# Alias for backward compatibility








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


    permissions = relationship('Permission', secondary='role_permissions', back_populates='roles')


    users = relationship('User', secondary='user_roles', back_populates='roles')


    


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


    roles = relationship('Role', secondary='role_permissions', back_populates='permissions')


    


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


    ip_address = Column(String(45))  # Using String instead of INET for compatibility


    user_agent = Column(Text)


    audit_metadata = Column(JSONB, default={})


    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)


    


    # Relationships


    user = relationship('User', back_populates='audit_logs', foreign_keys=[user_id])
    

    


    def __repr__(self):


        return f"<AuditLog(id={self.id}, action={self.action}, resource_type={self.resource_type})>"




# Alias for compatibility
UserProfile = UserPreference
