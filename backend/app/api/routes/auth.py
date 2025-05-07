from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel

from app.db.interfaces.user_interface import UserInterface
from app.security.encryption import get_password_hash, verify_password
from app.security.rbac_manager import get_user_permissions
from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Router
router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={401: {"description": "Unauthorized"}},
)

# JWT Config (should be in environment variables in production)
SECRET_KEY = "REPLACE_WITH_SECURE_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

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
    hashed_password: str

# Auth utilities
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, expire

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency to get the current user from a JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        role: str = payload.get("role")
        
        if username is None:
            raise credentials_exception
        
        token_data = TokenData(username=username, user_id=user_id, role=role)
    except JWTError:
        logger.error("JWT token validation failed")
        raise credentials_exception
    
    # Get user from database
    user_interface = UserInterface()
    user = await user_interface.get_user_by_username(token_data.username)
    
    if user is None:
        logger.warning(f"User not found: {token_data.username}")
        raise credentials_exception
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Dependency to check if the user is active"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Routes
@router.post("/register", response_model=User)
async def register_user(user_create: UserCreate):
    """Register a new user"""
    user_interface = UserInterface()
    
    # Check if username already exists
    existing_user = await user_interface.get_user_by_username(user_create.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = await user_interface.get_user_by_email(user_create.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password
    hashed_password = get_password_hash(user_create.password)
    
    # Create user in database
    new_user = await user_interface.create_user(
        username=user_create.username,
        email=user_create.email,
        hashed_password=hashed_password,
        role=user_create.role,
        client_id=user_create.client_id
    )
    
    logger.info(f"New user registered: {user_create.username}")
    return new_user

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login to get an access token"""
    user_interface = UserInterface()
    user = await user_interface.get_user_by_username(form_data.username)
    
    # Check if user exists and password is correct
    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token, expires_at = create_access_token(
        data={"sub": user.username, "user_id": user.id, "role": user.role}, 
        expires_delta=access_token_expires
    )
    
    # Get user permissions
    permissions = get_user_permissions(user.role)
    
    logger.info(f"User logged in: {form_data.username}")
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "expires_at": expires_at,
        "user_id": user.id,
        "role": user.role,
        "permissions": permissions
    }

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """Logout the user 
    
    In a stateless JWT setup, we don't actually invalidate the token on the server.
    Instead, the client should discard the token. This endpoint provides a standardized
    place for clients to do that as part of their flow.
    
    For a more secure setup, you could implement a token blacklist.
    """
    logger.info(f"User logged out: {current_user.username}")
    return {"detail": "Successfully logged out"}

@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get the current user's information"""
    return current_user

@router.put("/me", response_model=User)
async def update_user(
    user_update: dict,
    current_user: User = Depends(get_current_active_user)
):
    """Update the current user's information"""
    user_interface = UserInterface()
    
    # Only allow certain fields to be updated
    allowed_fields = {"email", "password"}
    update_data = {k: v for k, v in user_update.items() if k in allowed_fields}
    
    # If password is being updated, hash it
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
    
    # Update user in database
    updated_user = await user_interface.update_user(current_user.id, update_data)
    
    logger.info(f"User updated: {current_user.username}")
    return updated_user