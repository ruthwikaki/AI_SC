# app/db/interfaces/user_interface.py
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import json
import uuid
import logging
from app.models.user import User as UserModel
from app.db.database import get_db

# Setup logger
logger = logging.getLogger(__name__)

class User:
    """User data class for auth compatibility"""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.username = kwargs.get('username')
        self.email = kwargs.get('email')
        self.hashed_password = kwargs.get('hashed_password') or kwargs.get('password_hash')
        self.role = kwargs.get('role', 'user')
        self.is_active = kwargs.get('is_active', True)
        self.is_verified = kwargs.get('is_verified', False)
        self.first_name = kwargs.get('first_name')
        self.last_name = kwargs.get('last_name')
        
    def dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active
        }

class UserInterface:
    """Interface for user-related database operations"""
    
    def __init__(self, db: Session = None, admin_db_client_id: Optional[str] = None):
        self.db = db or next(get_db())
        self.admin_db_client_id = admin_db_client_id
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            result = self.db.query(UserModel).filter(UserModel.username == username).first()
            if result:
                return User(
                    id=str(result.id),
                    username=result.username,
                    email=result.email,
                    hashed_password=result.password_hash,
                    role=result.role,
                    is_active=result.is_active,
                    is_verified=result.is_verified,
                    first_name=result.first_name,
                    last_name=result.last_name
                )
            return None
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            result = self.db.query(UserModel).filter(UserModel.email == email).first()
            if result:
                return User(
                    id=str(result.id),
                    username=result.username,
                    email=result.email,
                    hashed_password=result.password_hash,
                    role=result.role,
                    is_active=result.is_active,
                    is_verified=result.is_verified,
                    first_name=result.first_name,
                    last_name=result.last_name
                )
            return None
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    async def create_user(self, username: str, email: str, hashed_password: str, 
                         role: str = "user", client_id: Optional[str] = None) -> User:
        """Create a new user"""
        try:
            db_user = UserModel(
                username=username,
                email=email,
                password_hash=hashed_password,
                role=role,
                is_active=True
            )
            self.db.add(db_user)
            self.db.commit()
            self.db.refresh(db_user)
            
            return User(
                id=str(db_user.id),
                username=db_user.username,
                email=db_user.email,
                role=db_user.role,
                is_active=db_user.is_active
            )
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            self.db.rollback()
            raise
    
    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Optional[User]:
        """Update user information"""
        try:
            db_user = self.db.query(UserModel).filter(UserModel.id == user_id).first()
            if not db_user:
                return None
            
            # Map hashed_password to password_hash if present
            if 'hashed_password' in update_data:
                update_data['password_hash'] = update_data.pop('hashed_password')
            
            for key, value in update_data.items():
                if hasattr(db_user, key):
                    setattr(db_user, key, value)
            
            db_user.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(db_user)
            
            return User(
                id=str(db_user.id),
                username=db_user.username,
                email=db_user.email,
                role=db_user.role,
                is_active=db_user.is_active
            )
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            self.db.rollback()
            return None
    
    # Keep existing methods for dashboard preferences
    async def get_user_dashboard_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's dashboard preferences."""
        try:
            result = self.db.execute(
                text("""
                SELECT preference_value 
                FROM user_preferences
                WHERE user_id = :user_id AND preference_key = 'dashboard_metrics'
                """),
                {"user_id": user_id}
            ).first()
            
            if result and result.preference_value:
                pref_value = result.preference_value
                if isinstance(pref_value, str):
                    return json.loads(pref_value)
                return pref_value
            
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
    
    async def save_user_dashboard_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, Any]
    ) -> bool:
        """Save user's dashboard preferences."""
        try:
            check_result = self.db.execute(
                text("""
                SELECT id FROM user_preferences
                WHERE user_id = :user_id AND preference_key = 'dashboard_metrics'
                """),
                {"user_id": user_id}
            ).first()
            
            if check_result:
                self.db.execute(
                    text("""
                    UPDATE user_preferences
                    SET preference_value = :value, updated_at = :updated_at
                    WHERE user_id = :user_id AND preference_key = 'dashboard_metrics'
                    """),
                    {
                        "user_id": user_id,
                        "value": json.dumps(preferences),
                        "updated_at": datetime.now()
                    }
                )
            else:
                self.db.execute(
                    text("""
                    INSERT INTO user_preferences (
                        id, user_id, preference_key, preference_value, created_at
                    )
                    VALUES (
                        :id, :user_id, :key, :value, :created_at
                    )
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "key": "dashboard_metrics",
                        "value": json.dumps(preferences),
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
    
    async def get_user_accessible_metrics(self, user_id: str) -> List[str]:
        """Get list of metrics accessible to the user based on their role."""
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return []
            
            role = user.get('role', 'user')
            
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
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            result = self.db.query(UserModel).filter(UserModel.id == user_id).first()
            
            if result:
                return {
                    "id": str(result.id),
                    "username": result.username,
                    "email": result.email,
                    "first_name": getattr(result, 'first_name', ''),
                    "last_name": getattr(result, 'last_name', ''),
                    "role": getattr(result, 'role', 'user'),
                    "is_active": getattr(result, 'is_active', True)
                }
            return None
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Alias for get_user_by_id for backward compatibility"""
        return await self.get_user_by_id(user_id)

# Helper function if needed for connector pattern
async def get_connector_for_client(client_id: str):
    """Placeholder for connector pattern - not used in direct approach"""
    return None

# Import this User class in auth.py as DBUser
