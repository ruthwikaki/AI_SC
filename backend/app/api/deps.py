"""
Common dependencies for FastAPI endpoints
Located at: /backend/app/api/deps.py
"""
from typing import Generator, Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import hashlib

from app.db.database import get_db
from app.config import get_settings
from app.models.user import User
from app.schemas.auth import TokenData
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Get settings
settings = get_settings()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.jwt_secret_key or settings.secret_key, 
        algorithm=settings.jwt_algorithm or settings.algorithm
    )
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    # If using passlib is available, use it
    try:
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.verify(plain_password, hashed_password)
    except ImportError:
        # Fallback to simple hash comparison
        return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def get_password_hash(password: str) -> str:
    """Hash a password"""
    try:
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.hash(password)
    except ImportError:
        # Fallback to simple SHA256
        return hashlib.sha256(password.encode()).hexdigest()

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token
    
    Args:
        token: JWT token from Authorization header
        db: Database session
    
    Returns:
        Current user object
    
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token, 
            settings.jwt_secret_key or settings.secret_key, 
            algorithms=[settings.jwt_algorithm or settings.algorithm]
        )
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise credentials_exception
        
        token_data = TokenData(user_id=user_id)
        
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = db.query(User).filter(User.id == token_data.user_id).first()
    
    if user is None:
        # Try by email as fallback
        user = db.query(User).filter(User.email == token_data.user_id).first()
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user
    
    Args:
        current_user: Current user from get_current_user
    
    Returns:
        Active user object
    
    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get current user with admin privileges
    
    Args:
        current_user: Current active user
    
    Returns:
        Admin user object
    
    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.is_superuser and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

def get_optional_current_user(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None
    
    Args:
        request: FastAPI request object
        db: Database session
    
    Returns:
        User object or None
    """
    # Try to get token from Authorization header
    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    token = authorization.split(" ")[1]
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.jwt_secret_key or settings.secret_key,
            algorithms=[settings.jwt_algorithm or settings.algorithm]
        )
        user_id = payload.get("sub")
        
        if not user_id:
            return None
        
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            user = db.query(User).filter(User.email == user_id).first()
        
        return user if user and user.is_active else None
        
    except JWTError:
        return None

# Database session dependency is already in database.py as get_db
# Re-export for convenience
from app.db.database import get_db

# Pagination dependencies
def get_pagination_params(
    skip: int = 0,
    limit: int = 100
) -> dict:
    """
    Common pagination parameters
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
    
    Returns:
        Dictionary with skip and limit
    """
    return {"skip": skip, "limit": min(limit, 1000)}  # Cap at 1000

# Sorting dependencies
def get_sort_params(
    sort_by: Optional[str] = None,
    sort_order: str = "asc"
) -> dict:
    """
    Common sorting parameters
    
    Args:
        sort_by: Field to sort by
        sort_order: Sort order (asc or desc)
    
    Returns:
        Dictionary with sort parameters
    """
    if sort_order not in ["asc", "desc"]:
        sort_order = "asc"
    
    return {
        "sort_by": sort_by,
        "sort_order": sort_order
    }

# Client context dependency
def get_client_id(request: Request) -> Optional[str]:
    """
    Get client ID from request
    
    Args:
        request: FastAPI request object
    
    Returns:
        Client ID or None
    """
    # Try header first
    client_id = request.headers.get("X-Client-ID")
    if client_id:
        return client_id
    
    # Try query parameter
    client_id = request.query_params.get("client_id")
    if client_id:
        return client_id
    
    # Try to get from user's default client
    return None

# API Key dependency (alternative to JWT)
async def get_api_key_user(
    api_key: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Validate API key and return associated user
    
    Args:
        api_key: API key from header
        db: Database session
    
    Returns:
        User associated with API key
    
    Raises:
        HTTPException: If API key is invalid
    """
    # First try as JWT token
    try:
        return await get_current_user(api_key, db)
    except HTTPException:
        pass
    
    # Then try as API key
    user = db.query(User).filter(User.api_key == api_key).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is inactive"
        )
    
    return user

# Request metadata
def get_request_metadata(request: Request) -> Dict[str, Any]:
    """
    Extract metadata from request
    
    Args:
        request: FastAPI request object
    
    Returns:
        Dictionary with request metadata
    """
    return {
        "ip_address": request.client.host if request.client else None,
        "user_agent": request.headers.get("User-Agent"),
        "method": request.method,
        "path": request.url.path,
        "timestamp": datetime.utcnow().isoformat()
    }

# Feature flags
def check_feature_flag(feature_name: str, user: Optional[User] = None) -> bool:
    """
    Check if a feature is enabled
    
    Args:
        feature_name: Name of the feature
        user: Optional user for user-specific features
    
    Returns:
        True if feature is enabled
    """
    # This is a placeholder - implement actual feature flag logic
    # Could check database, config file, or external service
    
    # For now, all features are enabled in development
    if settings.environment == "development":
        return True
    
    # Add specific feature checks here
    feature_flags = {
        "natural_language_queries": True,
        "advanced_analytics": True,
        "multi_tier_visualization": True,
        "export_functionality": True,
        "custom_dashboards": True
    }
    
    return feature_flags.get(feature_name, False)

# Rate limiting check (if not using middleware)
async def check_rate_limit(
    request: Request,
    user: Optional[User] = None
) -> bool:
    """
    Check if request is within rate limits
    
    Args:
        request: FastAPI request object
        user: Optional user for user-specific limits
    
    Returns:
        True if within limits
    
    Raises:
        HTTPException: If rate limit exceeded
    """
    # This is a placeholder - implement actual rate limiting
    # Could use Redis, memory cache, or database
    
    # For now, no rate limiting in development
    if settings.environment == "development":
        return True
    
    # Implement rate limiting logic here
    # raise HTTPException(
    #     status_code=status.HTTP_429_TOO_MANY_REQUESTS,
    #     detail="Rate limit exceeded"
    # )
    
    return True