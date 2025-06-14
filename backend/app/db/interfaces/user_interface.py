# app/db/interfaces/user_interface.py
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import json
import uuid
import logging
from app.models.user import User
from app.db.database import get_db

# Setup logger
logger = logging.getLogger(__name__)

class UserInterface:
    """Interface for user-related database operations"""
    
    def __init__(self, db: Session = None, admin_db_client_id: Optional[str] = None):
        self.db = db
        self.admin_db_client_id = admin_db_client_id
        if not self.db:
            # This will be set when used with dependency injection
            self.db = None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            if not self.db:
                return None
            user = self.db.query(User).filter(User.username == username).first()
            return user
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            if not self.db:
                return None
            user = self.db.query(User).filter(User.email == email).first()
            return user
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    async def create_user(
        self, 
        username: str, 
        email: str, 
        password_hash: str, 
        role: str = "user",
        client_id: Optional[str] = None
    ) -> User:
        """Create a new user"""
        try:
            if not self.db:
                raise Exception("Database session not available")
            
            new_user = User(
                id=uuid.uuid4(),
                username=username,
                email=email,
                password_hash=password_hash,
                role=role,
                is_active=True,
                created_at=datetime.utcnow()
            )
            
            self.db.add(new_user)
            self.db.commit()
            self.db.refresh(new_user)
            
            logger.info(f"Created new user: {username}")
            return new_user
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating user: {e}")
            raise
    
    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Optional[User]:
        """Update user information"""
        try:
            if not self.db:
                return None
            
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                return None
            
            for key, value in update_data.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            
            user.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(user)
            
            return user
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating user: {e}")
            return None
    
    async def get_user_dashboard_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's dashboard preferences.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with dashboard preferences
        """
        try:
            # Direct query approach for simplicity
            result = self.db.execute(
                text("""
                SELECT preference_value 
                FROM user_preferences
                WHERE user_id = :user_id AND preference_key = 'dashboard_metrics'
                """),
                {"user_id": user_id}
            ).first()
            
            if result and result.preference_value:
                # Parse JSON preference value
                pref_value = result.preference_value
                if isinstance(pref_value, str):
                    return json.loads(pref_value)
                return pref_value
            
            # Return default preferences
            return {
                "selected_metrics": [
                    "inventory_value",
                    "order_fill_rate",
                    "on_time_delivery",
                    "supplier_performance"
                ],
                "refresh_interval": 300,  # 5 minutes
                "time_frame": "last_month",
                "chart_preferences": {
                    "show_trends": True,
                    "show_abc_analysis": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard preferences: {str(e)}")
            # Return defaults on error
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
        """
        Save user's dashboard preferences.
        
        Args:
            user_id: User ID
            preferences: Dashboard preferences dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Check if preferences exist
            check_result = self.db.execute(
                text("""
                SELECT id FROM user_preferences
                WHERE user_id = :user_id AND preference_key = 'dashboard_metrics'
                """),
                {"user_id": user_id}
            ).first()
            
            if check_result:
                # Update existing
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
                # Insert new
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
        """
        Get list of metrics accessible to the user based on their role.
        
        Args:
            user_id: User ID
            
        Returns:
            List of accessible metric keys
        """
        try:
            # Get user to check role
            user = await self.get_user_by_id(user_id)
            if not user:
                return []
            
            role = user.role if hasattr(user, 'role') else 'user'
            
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
            return ["inventory_value", "order_fill_rate"]  # Minimum defaults
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            if not self.db:
                return None
            user = self.db.query(User).filter(User.id == user_id).first()
            return user
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Alias for get_user_by_id for backward compatibility"""
        return await self.get_user_by_id(user_id)

# Helper function if needed for connector pattern
async def get_connector_for_client(client_id: str):
    """Placeholder for connector pattern - not used in direct approach"""
    return None
