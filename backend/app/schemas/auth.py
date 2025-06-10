"""
Authentication and user management schemas
"""

from typing import Optional, List, Dict, Any, Annotated
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field, field_validator, StringConstraints


# =====================================================
# User Schemas
# =====================================================

class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    username: Annotated[str, StringConstraints(min_length=3, max_length=100, pattern='^[a-zA-Z0-9_-]+$')]
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=50)
    role: str = Field(default='user', max_length=50)


class UserCreate(UserBase):
    """Schema for creating a new user"""
    password: Annotated[str, StringConstraints(min_length=8, max_length=100)]
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        """Validate password strength"""
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v


class UserUpdate(BaseModel):
    """Schema for updating user information"""
    email: Optional[EmailStr] = None
    username: Optional[Annotated[str, StringConstraints(min_length=3, max_length=100, pattern='^[a-zA-Z0-9_-]+$')]] = None
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=50)
    avatar_url: Optional[str] = None
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """User schema with database fields"""
    id: UUID
    is_active: bool = True
    is_verified: bool = False
    email_verified_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True  # Changed from orm_mode


class UserResponse(BaseModel):
    """User response schema"""
    id: UUID
    email: str
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    department: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    role: str
    is_active: bool
    is_verified: bool
    last_login: Optional[datetime] = None
    created_at: datetime
    roles: List['RoleResponse'] = []
    
    class Config:
        from_attributes = True  # Changed from orm_mode


# =====================================================
# Authentication Schemas
# =====================================================

class UserLogin(BaseModel):
    """Login request schema"""
    email: EmailStr
    password: str
    remember_me: bool = False


class TokenResponse(BaseModel):
    """JWT token response schema"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenData(BaseModel):
    """Token payload data"""
    user_id: UUID
    username: str
    role: str
    exp: datetime
    
    class Config:
        json_encoders = {
            UUID: str
        }


# =====================================================
# Password Reset Schemas
# =====================================================

class PasswordResetRequest(BaseModel):
    """Password reset request schema"""
    email: EmailStr


class PasswordReset(BaseModel):
    """Password reset schema"""
    token: str
    new_password: Annotated[str, StringConstraints(min_length=8, max_length=100)]
    
    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v):
        """Validate password strength"""
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v


class PasswordChangeRequest(BaseModel):
    """Password change request schema"""
    current_password: str
    new_password: Annotated[str, StringConstraints(min_length=8, max_length=100)]
    
    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v):
        """Validate password strength"""
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v


# =====================================================
# Role and Permission Schemas
# =====================================================

class PermissionBase(BaseModel):
    """Base permission schema"""
    resource: str = Field(..., max_length=100)
    action: str = Field(..., max_length=50)
    display_name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = None


class PermissionResponse(PermissionBase):
    """Permission response schema"""
    id: UUID
    is_active: bool
    
    class Config:
        from_attributes = True  # Changed from orm_mode


class RoleBase(BaseModel):
    """Base role schema"""
    name: Annotated[str, StringConstraints(min_length=3, max_length=50, pattern='^[a-zA-Z0-9_-]+$')]
    display_name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None


class RoleCreate(RoleBase):
    """Role creation schema"""
    permission_ids: List[UUID] = []


class RoleUpdate(BaseModel):
    """Role update schema"""
    display_name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    permission_ids: Optional[List[UUID]] = None
    is_active: Optional[bool] = None


class RoleResponse(RoleBase):
    """Role response schema"""
    id: UUID
    is_system: bool
    is_active: bool
    permissions: List[PermissionResponse] = []
    created_at: datetime
    
    class Config:
        from_attributes = True  # Changed from orm_mode


class RoleAssignment(BaseModel):
    """Role assignment schema"""
    user_id: UUID
    role_id: UUID
    expires_at: Optional[datetime] = None


# =====================================================
# User Preferences Schemas
# =====================================================

class UserPreferencesUpdate(BaseModel):
    """User preferences update schema"""
    theme: Optional[Annotated[str, StringConstraints(pattern='^(light|dark|auto)$')]] = None
    language: Optional[Annotated[str, StringConstraints(pattern='^[a-z]{2}(-[A-Z]{2})?$')]] = None
    timezone: Optional[str] = None
    date_format: Optional[str] = None
    number_format: Optional[str] = None
    default_chart_type: Optional[str] = None
    dashboard_layout: Optional[Dict[str, Any]] = None
    notification_preferences: Optional[Dict[str, Any]] = None
    ui_preferences: Optional[Dict[str, Any]] = None


class UserPreferencesResponse(UserPreferencesUpdate):
    """User preferences response schema"""
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True  # Changed from orm_mode


# =====================================================
# Notification Settings Schemas
# =====================================================

class NotificationSettingsUpdate(BaseModel):
    """Notification settings update schema"""
    email_enabled: Optional[bool] = None
    push_enabled: Optional[bool] = None
    sms_enabled: Optional[bool] = None
    notification_types: Optional[Dict[str, bool]] = None
    quiet_hours: Optional[Dict[str, str]] = None


class NotificationSettingsResponse(NotificationSettingsUpdate):
    """Notification settings response schema"""
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True  # Changed from orm_mode


# =====================================================
# Session Management Schemas
# =====================================================

class SessionInfo(BaseModel):
    """Session information schema"""
    id: UUID
    user_id: UUID
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_id: Optional[str] = None
    is_active: bool
    created_at: datetime
    expires_at: datetime
    
    class Config:
        from_attributes = True  # Changed from orm_mode


class ActiveSessionsResponse(BaseModel):
    """Active sessions response"""
    sessions: List[SessionInfo]
    total: int


# =====================================================
# Audit Log Schemas
# =====================================================

class AuditLogEntry(BaseModel):
    """Audit log entry schema"""
    id: int
    user_id: Optional[UUID] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime
    
    class Config:
        from_attributes = True  # Changed from orm_mode


class AuditLogFilter(BaseModel):
    """Audit log filter schema"""
    user_id: Optional[UUID] = None
    action: Optional[str] = None
    resource_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    skip: int = 0
    limit: int = 100


# =====================================================
# User Statistics Schemas
# =====================================================

class UserStatistics(BaseModel):
    """User statistics schema"""
    total_queries: int
    saved_queries: int
    dashboards: int
    charts_created: int
    last_activity: Optional[datetime] = None
    storage_used_mb: float = 0.0
    api_calls_today: int = 0
    api_calls_this_month: int = 0


class UserActivity(BaseModel):
    """User activity schema"""
    date: str
    queries_count: int
    charts_created: int
    dashboards_viewed: int
    analytics_run: int