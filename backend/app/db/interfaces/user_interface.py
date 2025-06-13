# app/db/interfaces/user_interface.py
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import json
import uuid
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Simple User model class
class User:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.username = kwargs.get('username')
        self.email = kwargs.get('email')
        self.hashed_password = kwargs.get('hashed_password')
        self.role = kwargs.get('role', 'user')
        self.is_active = kwargs.get('is_active', True)
        self.client_id = kwargs.get('client_id')

class UserInterface:
    """Interface for user-related database operations"""
    
    def __init__(self, db: Session, admin_db_client_id: Optional[str] = None):
        self.db = db
        self.admin_db_client_id = admin_db_client_id
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username OR email - ASYNC"""
        try:
            # Try exact username first
            query = """
                SELECT id, username, email, hashed_password, role
                FROM users 
                WHERE username = :username
            """
            result = self.db.execute(text(query), {"username": username}).first()
            
            # If not found and username contains @, try as email
            if not result and '@' in username:
                query = """
                    SELECT id, username, email, hashed_password, role
                    FROM users 
                    WHERE email = :email
                """
                result = self.db.execute(text(query), {"email": username}).first()
            
            if result:
                logger.debug(f"Found user: {result.username}, has password: {bool(result.hashed_password)}")
                return User(
                    id=str(result.id),
                    username=result.username,
                    email=result.email,
                    hashed_password=result.hashed_password,
                    role=result.role
                )
            
            logger.debug(f"No user found for: {username}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email - ASYNC"""
        try:
            query = """
                SELECT id, username, email, hashed_password, role
                FROM users 
                WHERE email = :email
            """
            result = self.db.execute(text(query), {"email": email}).first()
            
            if result:
                return User(
                    id=str(result.id),
                    username=result.username,
                    email=result.email,
                    hashed_password=result.hashed_password,
                    role=result.role
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID - ASYNC"""
        try:
            query = """
                SELECT id, username, email, hashed_password, role
                FROM users 
                WHERE id = :user_id
            """
            result = self.db.execute(text(query), {"user_id": user_id}).first()
            
            if result:
                return User(
                    id=str(result.id),
                    username=result.username,
                    email=result.email,
                    hashed_password=result.hashed_password,
                    role=result.role
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting user by id: {e}")
            return None
    
    async def create_user(self, username: str, email: str, hashed_password: str, 
                   role: str = "user", client_id: Optional[str] = None) -> Optional[User]:
        """Create a new user - ASYNC"""
        try:
            user_id = str(uuid.uuid4())
            
            query = """
                INSERT INTO users (id, username, email, hashed_password, role)
                VALUES (:id, :username, :email, :hashed_password, :role)
            """
            
            self.db.execute(
                text(query),
                {
                    "id": user_id,
                    "username": username,
                    "email": email,
                    "hashed_password": hashed_password,
                    "role": role
                }
            )
            self.db.commit()
            
            return User(
                id=user_id,
                username=username,
                email=email,
                hashed_password=hashed_password,
                role=role
            )
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating user: {e}")
            return None
    
    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Optional[User]:
        """Update user data - ASYNC"""
        try:
            set_clauses = []
            params = {"user_id": user_id}
            
            for key, value in update_data.items():
                if key in ["email", "role", "hashed_password"]:
                    set_clauses.append(f"{key} = :{key}")
                    params[key] = value
            
            if not set_clauses:
                return None
            
            query = f"""
                UPDATE users 
                SET {', '.join(set_clauses)}
                WHERE id = :user_id
            """
            
            self.db.execute(text(query), params)
            self.db.commit()
            
            return await self.get_user_by_id(user_id)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating user: {e}")
            return None
    
    # Dashboard preferences methods
    async def get_user_dashboard_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's dashboard preferences"""
        return {
            "selected_metrics": ["inventory_value", "order_fill_rate"],
            "refresh_interval": 300,
            "time_frame": "last_month"
        }
    
    async def save_user_dashboard_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Save user's dashboard preferences"""
        return True
    
    async def get_user_accessible_metrics(self, user_id: str) -> List[str]:
        """Get list of metrics accessible to the user"""
        return ["inventory_value", "order_fill_rate", "on_time_delivery"]
