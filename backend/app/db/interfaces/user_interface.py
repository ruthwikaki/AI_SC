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

class User:
    """User model class to match the database schema"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class UserInterface:
    """Interface for user-related database operations"""
    
    def __init__(self, db: Optional[Session] = None, admin_db_client_id: Optional[str] = None):
        # Import here to avoid circular imports
        if db is None:
            from app.db.database import get_db
            self.db = next(get_db())
        else:
            self.db = db
        self.admin_db_client_id = admin_db_client_id
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username - required for authentication"""
        try:
            result = self.db.execute(
                text("SELECT * FROM users WHERE username = :username"),
                {"username": username}
            ).first()
            
            if result:
                return User(
                    id=str(result.id),
                    username=result.username,
                    email=result.email,
                    hashed_password=result.password_hash,  # Map password_hash to hashed_password
                    first_name=getattr(result, 'first_name', ''),
                    last_name=getattr(result, 'last_name', ''),
                    role=getattr(result, 'role', 'user'),
                    is_active=getattr(result, 'is_active', True),
                    created_at=getattr(result, 'created_at', None),
                    updated_at=getattr(result, 'updated_at', None)
                )
            return None
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email - required for registration check"""
        try:
            result = self.db.execute(
                text("SELECT * FROM users WHERE email = :email"),
                {"email": email}
            ).first()
            
            if result:
                return User(
                    id=str(result.id),
                    username=result.username,
                    email=result.email,
                    hashed_password=result.password_hash,  # Map password_hash to hashed_password
                    first_name=getattr(result, 'first_name', ''),
                    last_name=getattr(result, 'last_name', ''),
                    role=getattr(result, 'role', 'user'),
                    is_active=getattr(result, 'is_active', True),
                    created_at=getattr(result, 'created_at', None),
                    updated_at=getattr(result, 'updated_at', None)
                )
            return None
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    def create_user(self, username: str, email: str, hashed_password: str, 
                         role: str = "user", client_id: Optional[str] = None) -> User:
        """Create a new user - required for registration"""
        try:
            user_id = str(uuid.uuid4())
            
            user_data = {
                "id": user_id,
                "username": username,
                "email": email,
                "password_hash": hashed_password,  # Map hashed_password to password_hash
                "role": role,
                "is_active": True,
                "created_at": datetime.now()
            }
            
            self.db.execute(
                text("""
                INSERT INTO users (id, username, email, password_hash, role, is_active, created_at)
                VALUES (:id, :username, :email, :password_hash, :role, :is_active, :created_at)
                """),
                user_data
            )
            
            self.db.commit()
            logger.info(f"Created new user: {username}")
            
            return User(
                id=user_id,
                username=username,
                email=email,
                hashed_password=hashed_password,
                role=role,
                is_active=True,
                created_at=datetime.now()
            )
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating user: {e}")
            raise e
    
    def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Optional[User]:
        """Update user data"""
        try:
            # Build dynamic update query
            set_clauses = []
            params = {"user_id": user_id, "updated_at": datetime.now()}
            
            for key, value in update_data.items():
                # Map hashed_password to password_hash for database
                db_key = "password_hash" if key == "hashed_password" else key
                set_clauses.append(f"{db_key} = :{key}")
                params[key] = value
            
            if not set_clauses:
                return self.get_user_by_id(user_id)
            
            query = f"""
            UPDATE users 
            SET {', '.join(set_clauses)}, updated_at = :updated_at
            WHERE id = :user_id
            """
            
            self.db.execute(text(query), params)
            self.db.commit()
            
            logger.info(f"Updated user: {user_id}")
            return self.get_user_by_id(user_id)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating user: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            result = self.db.execute(
                text("SELECT * FROM users WHERE id = :user_id"),
                {"user_id": user_id}
            ).first()
            
            if result:
                return User(
                    id=str(result.id),
                    username=result.username,
                    email=result.email,
                    hashed_password=result.password_hash,  # Map password_hash to hashed_password
                    first_name=getattr(result, 'first_name', ''),
                    last_name=getattr(result, 'last_name', ''),
                    role=getattr(result, 'role', 'user'),
                    is_active=getattr(result, 'is_active', True),
                    created_at=getattr(result, 'created_at', None),
                    updated_at=getattr(result, 'updated_at', None)
                )
            return None
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    # Existing methods for dashboard functionality
    def get_user_dashboard_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's dashboard preferences"""
        try:
            result = self.db.execute(
                text("""
                SELECT ui_preferences 
                FROM user_preferences
                WHERE user_id = :user_id
                """),
                {"user_id": user_id}
            ).first()
            
            if result and result.ui_preferences:
                return result.ui_preferences
            
            # Return default preferences
            return {
                "selected_metrics": [
                    "inventory_value",
                    "order_fill_rate", 
                    "on_time_delivery",
                    "supplier_performance"
                ],
                "refresh_interval": 300,
                "time_frame": "last_month",
                "chart_preferences": {
                    "show_trends": True,
                    "show_abc_analysis": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard preferences: {str(e)}")
            return {
                "selected_metrics": [
                    "inventory_value",
                    "order_fill_rate",
                    "on_time_delivery",
                    "supplier_performance"
                ],
                "refresh_interval": 300,
                "time_frame": "last_month"
            }
    
    def save_user_dashboard_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Save user's dashboard preferences"""
        try:
            # Check if user preferences exist
            check_result = self.db.execute(
                text("SELECT id FROM user_preferences WHERE user_id = :user_id"),
                {"user_id": user_id}
            ).first()
            
            if check_result:
                # Update existing
                self.db.execute(
                    text("""
                    UPDATE user_preferences
                    SET ui_preferences = :preferences, updated_at = :updated_at
                    WHERE user_id = :user_id
                    """),
                    {
                        "user_id": user_id,
                        "preferences": json.dumps(preferences),
                        "updated_at": datetime.now()
                    }
                )
            else:
                # Insert new
                self.db.execute(
                    text("""
                    INSERT INTO user_preferences (
                        id, user_id, ui_preferences, created_at
                    )
                    VALUES (
                        :id, :user_id, :preferences, :created_at
                    )
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "preferences": json.dumps(preferences),
                        "created_at": datetime.now()
                    }
                )
            
            self.db.commit()
            logger.info(f"Saved dashboard preferences for user: {user_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error saving dashboard preferences: {str(e)}")
            return False
    
    def get_user_accessible_metrics(self, user_id: str) -> List[str]:
        """Get list of metrics accessible to the user based on their role"""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                return []
            
            role = user.role
            
            # Define metrics by role
            role_metrics = {
                "admin": [
                    "inventory_value",
                    "order_fill_rate",
                    "on_time_delivery", 
                    "supplier_performance",
                    "total_revenue",
                    "cost_savings",
                    "cash_cycle",
                    "network_efficiency"
                ],
                "manager": [
                    "inventory_value",
                    "order_fill_rate",
                    "on_time_delivery",
                    "supplier_performance",
                    "total_revenue"
                ],
                "user": [
                    "inventory_value",
                    "order_fill_rate",
                    "on_time_delivery"
                ],
                "viewer": [
                    "inventory_value",
                    "order_fill_rate"
                ]
            }
            
            return role_metrics.get(role, role_metrics["user"])
            
        except Exception as e:
            logger.error(f"Error getting accessible metrics: {str(e)}")
            return ["inventory_value", "order_fill_rate"]