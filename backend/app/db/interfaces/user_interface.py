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
    def __init__(self, id, username, email, hashed_password, role, is_active, client_id=None):
        self.id = id
        self.username = username
        self.email = email
        self.hashed_password = hashed_password
        self.role = role
        self.is_active = is_active
        self.client_id = client_id

class UserInterface:
    """Interface for user-related database operations"""
    
    def __init__(self, db: Session, admin_db_client_id: Optional[str] = None):
        self.db = db
        self.admin_db_client_id = admin_db_client_id
    
    def _extract_password_hash(self, result) -> Optional[str]:
        """Extract password hash from result, handling different column names"""
        if not result:
            return None
            
        # Try different possible column names
        password_columns = ['hashed_password', 'password_hash', 'password', 'encrypted_password']
        
        for col in password_columns:
            if hasattr(result, col):
                return getattr(result, col)
        
        # If result is a dict-like object
        try:
            for col in password_columns:
                if col in result:
                    return result[col]
        except:
            pass
            
        logger.warning("No password column found in result")
        return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username - SYNCHRONOUS"""
        try:
            result = self.db.execute(
                text("SELECT * FROM users WHERE username = :username"),
                {"username": username}
            ).first()
            
            if result:
                password_hash = self._extract_password_hash(result)
                
                return User(
                    id=str(result.id),
                    username=result.username,
                    email=result.email,
                    hashed_password=password_hash,
                    role=getattr(result, 'role', 'user'),
                    is_active=getattr(result, 'is_active', True),
                    client_id=getattr(result, 'client_id', None)
                )
            return None
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email - SYNCHRONOUS"""
        try:
            result = self.db.execute(
                text("SELECT * FROM users WHERE email = :email"),
                {"email": email}
            ).first()
            
            if result:
                password_hash = self._extract_password_hash(result)
                
                return User(
                    id=str(result.id),
                    username=result.username,
                    email=result.email,
                    hashed_password=password_hash,
                    role=getattr(result, 'role', 'user'),
                    is_active=getattr(result, 'is_active', True),
                    client_id=getattr(result, 'client_id', None)
                )
            return None
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID - SYNCHRONOUS"""
        try:
            result = self.db.execute(
                text("SELECT * FROM users WHERE id = :user_id"),
                {"user_id": user_id}
            ).first()
            
            if result:
                password_hash = self._extract_password_hash(result)
                
                return User(
                    id=str(result.id),
                    username=result.username,
                    email=result.email,
                    hashed_password=password_hash,
                    role=getattr(result, 'role', 'user'),
                    is_active=getattr(result, 'is_active', True),
                    client_id=getattr(result, 'client_id', None)
                )
            return None
        except Exception as e:
            logger.error(f"Error getting user by id: {e}")
            return None
    
    def create_user(self, username: str, email: str, hashed_password: str, 
                   role: str = "user", client_id: Optional[str] = None) -> Optional[User]:
        """Create a new user - SYNCHRONOUS"""
        try:
            user_id = str(uuid.uuid4())
            
            # First, check which password column exists
            columns_result = self.db.execute(
                text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'users' 
                AND column_name IN ('hashed_password', 'password_hash', 'password')
                """)
            ).first()
            
            password_column = 'hashed_password'  # default
            if columns_result:
                password_column = columns_result[0]
            
            self.db.execute(
                text(f"""
                INSERT INTO users (id, username, email, {password_column}, role, is_active, client_id, created_at)
                VALUES (:id, :username, :email, :hashed_password, :role, :is_active, :client_id, :created_at)
                """),
                {
                    "id": user_id,
                    "username": username,
                    "email": email,
                    "hashed_password": hashed_password,
                    "role": role,
                    "is_active": True,
                    "client_id": client_id,
                    "created_at": datetime.now()
                }
            )
            self.db.commit()
            
            return User(
                id=user_id,
                username=username,
                email=email,
                hashed_password=hashed_password,
                role=role,
                is_active=True,
                client_id=client_id
            )
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating user: {e}")
            return None
    
    def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Optional[User]:
        """Update user data - SYNCHRONOUS"""
        try:
            # Build update query dynamically
            set_clauses = []
            params = {"user_id": user_id}
            
            # Map password_hash to the correct column name
            if "password_hash" in update_data:
                # Check which password column exists
                columns_result = self.db.execute(
                    text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'users' 
                    AND column_name IN ('hashed_password', 'password_hash', 'password')
                    """)
                ).first()
                
                password_column = 'hashed_password'  # default
                if columns_result:
                    password_column = columns_result[0]
                
                set_clauses.append(f"{password_column} = :password_hash")
                params["password_hash"] = update_data["password_hash"]
                update_data.pop("password_hash")
            
            # Handle other fields
            for key, value in update_data.items():
                if key in ["email", "role", "is_active"]:
                    set_clauses.append(f"{key} = :{key}")
                    params[key] = value
            
            if not set_clauses:
                return None
            
            query = f"""
                UPDATE users 
                SET {', '.join(set_clauses)}, updated_at = :updated_at
                WHERE id = :user_id
            """
            
            params["updated_at"] = datetime.now()
            
            self.db.execute(text(query), params)
            self.db.commit()
            
            return self.get_user_by_id(user_id)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating user: {e}")
            return None
    
    # Async methods for other operations
    async def get_user_dashboard_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's dashboard preferences - ASYNC"""
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
    
    async def save_user_dashboard_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, Any]
    ) -> bool:
        """Save user's dashboard preferences - ASYNC"""
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
        """Get list of metrics accessible to the user based on their role - ASYNC"""
        try:
            user = self.get_user_by_id(user_id)  # Use sync method
            if not user:
                return []
            
            role = user.role
            
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