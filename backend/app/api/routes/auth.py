# backend/app/api/routes/auth.py - WORKING VERSION

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import asyncio

from app.db.database import get_db
from app.db.interfaces.user_interface import UserInterface, User as DBUser
from app.security.rbac_manager import get_user_permissions
from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Router
router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={401: {"description": "Unauthorized"}},
)

# Models
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: datetime
    user_id: str
    role: str
    permissions: List[str]

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None

class User(BaseModel):
    id: str
    username: str
    email: str
    role: str
    is_active: bool = True
    client_id: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "user"
    client_id: Optional[str] = None

class UserInDB(User):
    password_hash: str

# Auth utilities
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt, expire

# Helper function to handle async calls
def run_async(coro):
    """Helper to run async functions in sync context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Dependency to get the current user from a JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        role: str = payload.get("role")
        
        if username is None:
            raise credentials_exception
        
        token_data = TokenData(username=username, user_id=user_id, role=role)
    except JWTError:
        logger.error("JWT token validation failed")
        raise credentials_exception
    
    # Get user from database with proper session
    user_interface = UserInterface(db)
    
    # Handle async method call
    user = user_interface.get_user_by_username(token_data.username)
    
    if user is None:
        logger.warning(f"User not found: {token_data.username}")
        raise credentials_exception
    
    return user

async def get_current_active_user(current_user: DBUser = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Routes
@router.post("/register", response_model=User)
async def register_user(user_create: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    user_interface = UserInterface(db)
    
    try:
        # Check if username already exists
        existing_user = user_interface.get_user_by_username(user_create.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists
        existing_email = user_interface.get_user_by_email(user_create.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password
        password_hash = get_password_hash(user_create.password)
        
        # Create user in database
        new_user = user_interface.create_user(
            username=user_create.username,
            email=user_create.email,
            hashed_password=password_hash,
            role=user_create.role,
            client_id=user_create.client_id
        )
        
        logger.info(f"New user registered: {user_create.username}")
        
        # Return user data in the expected format
        return User(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            role=new_user.role,
            is_active=new_user.is_active
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login to get an access token"""
    try:
        user_interface = UserInterface(db)
        
        # Log the login attempt
        logger.info(f"Login attempt for username: {form_data.username}")
        
        # Try to get user by username first, then by email
        user = user_interface.get_user_by_username(form_data.username)
        
        # If not found by username, try email
        if not user:
            logger.info(f"User not found by username, trying email: {form_data.username}")
            user = user_interface.get_user_by_email(form_data.username)
        
        # Debug logging
        if user:
            logger.info(f"User found: {user.username}, Active: {user.is_active}")
        else:
            logger.warning(f"User not found: {form_data.username}")
        
        # Check if user exists and password is correct
        if not user:
            logger.warning(f"Failed login - user not found: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify password
        password_valid = verify_password(form_data.password, user.hashed_password)
        logger.info(f"Password verification for {form_data.username}: {'Valid' if password_valid else 'Invalid'}")
        
        if not password_valid:
            logger.warning(f"Failed login - invalid password for user: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if user is active
        if not user.is_active:
            logger.warning(f"Failed login - inactive user: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token, expires_at = create_access_token(
            data={"sub": user.username, "user_id": user.id, "role": user.role},
            expires_delta=access_token_expires
        )
        
        # Get user permissions
        try:
            permissions = get_user_permissions(user.role)
        except Exception as e:
            logger.warning(f"Could not get permissions for role {user.role}: {e}")
            permissions = []
        
        logger.info(f"User logged in successfully: {form_data.username}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_at=expires_at,
            user_id=user.id,
            role=user.role,
            permissions=permissions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """Logout the user"""
    logger.info(f"User logged out: {current_user.username}")
    return {"detail": "Successfully logged out"}

@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get the current user's information"""
    return User(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active
    )

@router.put("/me", response_model=User)
async def update_user(
    user_update: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update the current user's information"""
    try:
        user_interface = UserInterface(db)
        
        # Only allow certain fields to be updated
        allowed_fields = {"email", "password"}
        update_data = {k: v for k, v in user_update.items() if k in allowed_fields}
        
        # If password is being updated, hash it
        if "password" in update_data:
            update_data["password_hash"] = get_password_hash(update_data.pop("password"))
        
        # Update user in database
        updated_user = user_interface.update_user(current_user.id, update_data)
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User updated: {current_user.username}")
        
        return User(
            id=updated_user.id,
            username=updated_user.username,
            email=updated_user.email,
            role=updated_user.role,
            is_active=updated_user.is_active
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User update failed"
        )

# Test endpoint for debugging
@router.post("/test-login")
async def test_login(email: str, password: str, db: Session = Depends(get_db)):
    """Test endpoint to debug login issues"""
    user_interface = UserInterface(db)
    
    # Try to find user
    user = user_interface.get_user_by_email(email)
    if not user:
        user = user_interface.get_user_by_username(email)
    
    result = {
        "user_found": user is not None,
        "user_details": None,
        "password_check": False,
        "password_hash_sample": None
    }
    
    if user:
        result["user_details"] = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "is_active": user.is_active
        }
        result["password_hash_sample"] = user.hashed_password[:50] + "..."
        
        # Test password
        try:
            result["password_check"] = verify_password(password, user.hashed_password)
        except Exception as e:
            result["password_check"] = f"Error: {str(e)}"
    
    # Also test creating a new hash
    result["test_hash"] = get_password_hash(password)[:50] + "..."
    
    return result