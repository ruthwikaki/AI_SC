"""
Authentication schemas for request/response validation
Located at: /backend/app/schemas/auth.py
"""
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

# Token schemas
class Token(BaseModel):
    """Token response schema"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

class TokenData(BaseModel):
    """Token data extracted from JWT"""
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    permissions: Optional[List[str]] = []

# User authentication schemas
class UserLogin(BaseModel):
    """User login request schema"""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., min_length=6)
    remember_me: bool = False

class UserRegister(BaseModel):
    """User registration request schema"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, regex="^[a-zA-Z0-9_-]+$")
    password: str = Field(..., min_length=8)
    confirm_password: str
    full_name: Optional[str] = None
    department: Optional[str] = None
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength"""
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v

class PasswordChange(BaseModel):
    """Password change request schema"""
    current_password: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

class PasswordReset(BaseModel):
    """Password reset request schema"""
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema"""
    token: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

# User profile schemas
class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    department: Optional[str] = None
    job_title: Optional[str] = None
    phone_number: Optional[str] = None

class UserCreate(UserBase):
    """User creation schema (admin)"""
    password: str = Field(..., min_length=8)
    role: str = "viewer"
    is_active: bool = True
    is_verified: bool = False

class UserUpdate(BaseModel):
    """User update schema"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    department: Optional[str] = None
    job_title: Optional[str] = None
    phone_number: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    notification_settings: Optional[Dict[str, Any]] = None

class UserInDB(UserBase):
    """User in database schema"""
    id: str
    role: str
    is_active: bool
    is_verified: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class UserResponse(UserBase):
    """User response schema"""
    id: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    preferences: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True

# Permission and role schemas
class PermissionBase(BaseModel):
    """Base permission schema"""
    name: str
    resource: str
    action: str
    description: Optional[str] = None

class PermissionCreate(PermissionBase):
    """Permission creation schema"""
    pass

class PermissionResponse(PermissionBase):
    """Permission response schema"""
    id: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class RoleBase(BaseModel):
    """Base role schema"""
    name: str
    description: Optional[str] = None
    is_system: bool = False

class RoleCreate(RoleBase):
    """Role creation schema"""
    permissions: List[str] = []  # List of permission IDs

class RoleUpdate(BaseModel):
    """Role update schema"""
    name: Optional[str] = None
    description: Optional[str] = None
    permissions: Optional[List[str]] = None

class RoleResponse(RoleBase):
    """Role response schema"""
    id: str
    created_at: datetime
    permissions: List[PermissionResponse] = []
    
    class Config:
        from_attributes = True

# API Key schemas
class APIKeyCreate(BaseModel):
    """API key creation request"""
    name: str = Field(..., description="Name for the API key")
    expires_in_days: Optional[int] = Field(None, description="Expiration in days (None for no expiration)")

class APIKeyResponse(BaseModel):
    """API key response"""
    key: str = Field(..., description="The API key (only shown once)")
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    
class APIKeyList(BaseModel):
    """API key list item"""
    id: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool

# Session schemas
class SessionInfo(BaseModel):
    """Session information"""
    id: str
    user_id: str
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    is_current: bool = False