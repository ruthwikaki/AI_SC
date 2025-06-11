from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db.interfaces.user_interface import UserInterface
from app.db.database import get_db
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
    hashed_password: str

# Auth utilities
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

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

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
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
    
    # Get user from database
    db = next(get_db())
    user_interface = UserInterface(db)
    
    # Note: UserInterface doesn't have get_user_by_username method based on the provided code
    # Using get_user_by_id instead since we have the user_id from the token
    user_dict = await user_interface.get_user_by_id(token_data.user_id)
    
    if user_dict is None:
        logger.warning(f"User not found: {token_data.username}")
        raise credentials_exception
    
    # Convert dict to User model
    return User(**user_dict)

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Routes
@router.post("/register", response_model=User)
async def register_user(user_create: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    user_interface = UserInterface(db)
    
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
    # Note: UserInterface doesn't have create_user method, so this needs to be implemented
    # For now, using raw SQL as a placeholder
    import uuid
    
    user_id = str(uuid.uuid4())
    try:
        db.execute(
            text("""
            INSERT INTO users (id, username, email, password_hash, role, is_active, created_at)
            VALUES (:id, :username, :email, :password_hash, :role, :is_active, :created_at)
            """),
            {
                "id": user_id,
                "username": user_create.username,
                "email": user_create.email,
                "password_hash": hashed_password,
                "role": user_create.role,
                "is_active": True,
                "created_at": datetime.now()
            }
        )
        db.commit()
        
        # Return the created user
        new_user = User(
            id=user_id,
            username=user_create.username,
            email=user_create.email,
            role=user_create.role,
            is_active=True,
            client_id=user_create.client_id
        )
        
        logger.info(f"New user registered: {user_create.username}")
        return new_user
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating user"
        )

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login to get an access token"""
    user_interface = UserInterface(db)
    
    # Get user by username/email
    user_dict = await user_interface.get_user_by_email(form_data.username)
    if not user_dict:
        # Try to get by username if email lookup fails
        # Since UserInterface doesn't have get_user_by_username, we'll use email as username
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get password hash from database
    result = db.execute(
        text("SELECT password_hash FROM users WHERE id = :user_id"),
        {"user_id": user_dict['id']}
    ).first()
    
    if not result or not verify_password(form_data.password, result.password_hash):
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token, expires_at = create_access_token(
        data={
            "sub": user_dict['username'],
            "user_id": user_dict['id'],
            "role": user_dict['role']
        },
        expires_delta=access_token_expires
    )
    
    # Get user permissions
    permissions = get_user_permissions(user_dict['role'])
    
    logger.info(f"User logged in: {form_data.username}")
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": expires_at,
        "user_id": user_dict['id'],
        "role": user_dict['role'],
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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update the current user's information"""
    user_interface = UserInterface(db)
    
    # Only allow certain fields to be updated
    allowed_fields = {"email", "password"}
    update_data = {k: v for k, v in user_update.items() if k in allowed_fields}
    
    # If password is being updated, hash it
    if "password" in update_data:
        update_data["password_hash"] = get_password_hash(update_data.pop("password"))
    
    # Update user in database
    # Note: UserInterface doesn't have update_user method, so using raw SQL
    if update_data:
        set_clause = ", ".join([f"{k} = :{k}" for k in update_data.keys()])
        update_data["user_id"] = current_user.id
        update_data["updated_at"] = datetime.now()
        
        try:
            db.execute(
                text(f"""
                UPDATE users 
                SET {set_clause}, updated_at = :updated_at
                WHERE id = :user_id
                """),
                update_data
            )
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating user"
            )
    
    # Get updated user
    updated_user_dict = await user_interface.get_user_by_id(current_user.id)
    
    logger.info(f"User updated: {current_user.username}")
    return User(**updated_user_dict)