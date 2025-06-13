from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel
from passlib.context import CryptContext
import traceback

from app.db.database import get_db
from app.db.interfaces.user_interface import UserInterface, User as DBUser
from app.security.rbac_manager import get_user_permissions
from app.utils.logger import get_logger
from app.config import get_settings
from sqlalchemy.orm import Session

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

# Auth utilities
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt, expire

# Define get_current_user BEFORE it's used
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
        
        if username is None:
            raise credentials_exception
            
    except JWTError:
        logger.error("JWT token validation failed")
        raise credentials_exception
    
    # Get user from database - WITH AWAIT
    user_interface = UserInterface(db)
    user = await user_interface.get_user_by_username(username)
    
    if user is None:
        logger.warning(f"User not found: {username}")
        raise credentials_exception
    
    return user

async def get_current_active_user(current_user: DBUser = Depends(get_current_user)):
    if not hasattr(current_user, 'is_active') or not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Routes
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login to get an access token"""
    logger.info(f"Login attempt for user: {form_data.username}")
    
    try:
        # Initialize UserInterface with db session
        user_interface = UserInterface(db)
        logger.debug("UserInterface initialized")
        
        # Get user - WITH AWAIT
        logger.debug("Looking up user...")
        user = await user_interface.get_user_by_username(form_data.username)
        logger.debug(f"User lookup result: {user}")
        
        # Check if user exists
        if not user:
            logger.warning(f"User not found: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if user has password
        if not hasattr(user, 'hashed_password') or not user.hashed_password:
            logger.error(f"User {form_data.username} has no password hash")
            logger.debug(f"User object: {vars(user)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        logger.debug(f"Verifying password for user: {form_data.username}")
        logger.debug(f"Password hash exists: {bool(user.hashed_password)}")
        
        # Verify password
        if not verify_password(form_data.password, user.hashed_password):
            logger.warning(f"Invalid password for user: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        logger.debug("Password verified successfully")
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token, expires_at = create_access_token(
            data={"sub": user.username, "user_id": user.id, "role": user.role},
            expires_delta=access_token_expires
        )
        
        logger.debug("Access token created")
        
        # Get user permissions
        try:
            permissions = get_user_permissions(user.role)
            logger.debug(f"Permissions for role {user.role}: {permissions}")
        except Exception as e:
            logger.warning(f"Error getting permissions: {e}")
            permissions = []  # Default empty permissions
        
        logger.info(f"User logged in successfully: {form_data.username}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_at": expires_at,
            "user_id": user.id,
            "role": user.role,
            "permissions": permissions
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full error
        logger.error(f"Login error for {form_data.username}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a generic error to avoid leaking information
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login"
        )

@router.post("/register", response_model=User)
async def register_user(user_create: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    user_interface = UserInterface(db)
    
    # Check if username already exists - WITH AWAIT
    existing_user = await user_interface.get_user_by_username(user_create.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists - WITH AWAIT
    existing_email = await user_interface.get_user_by_email(user_create.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password
    hashed_password = get_password_hash(user_create.password)
    
    # Create user in database - WITH AWAIT
    new_user = await user_interface.create_user(
        username=user_create.username,
        email=user_create.email,
        hashed_password=hashed_password,
        role=user_create.role,
        client_id=user_create.client_id
    )
    
    if not new_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )
    
    logger.info(f"New user registered: {user_create.username}")
    
    return User(
        id=new_user.id,
        username=new_user.username,
        email=new_user.email,
        role=new_user.role,
        is_active=new_user.is_active,
        client_id=getattr(new_user, 'client_id', None)
    )

@router.get("/me", response_model=User)
async def read_users_me(current_user: DBUser = Depends(get_current_active_user)):
    """Get the current user's information"""
    return User(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        client_id=getattr(current_user, 'client_id', None)
    )

@router.put("/me", response_model=User)
async def update_user(
    user_update: dict,
    current_user: DBUser = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update the current user's information"""
    user_interface = UserInterface(db)
    
    # Only allow certain fields to be updated
    allowed_fields = {"email", "password"}
    update_data = {k: v for k, v in user_update.items() if k in allowed_fields}
    
    # If password is being updated, hash it
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
    
    # Update user in database - WITH AWAIT
    updated_user = await user_interface.update_user(current_user.id, update_data)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )
    
    logger.info(f"User updated: {current_user.username}")
    
    return User(
        id=updated_user.id,
        username=updated_user.username,
        email=updated_user.email,
        role=updated_user.role,
        is_active=updated_user.is_active,
        client_id=getattr(updated_user, 'client_id', None)
    )

@router.post("/logout")
async def logout(current_user: DBUser = Depends(get_current_active_user)):
    """Logout the user"""
    logger.info(f"User logged out: {current_user.username}")
    return {"detail": "Successfully logged out"}
