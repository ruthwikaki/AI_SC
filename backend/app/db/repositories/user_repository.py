# Models should be imported inside methods to avoid circular imports
# Example:
# def get_something(self):
#     from app.models import User  # Import here, not at module level
#     return User.query.all()
"""
User repository for user-related database operations
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
import logging
from app.models.user import Role

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, or_, and_
from sqlalchemy.exc import IntegrityError

# MOVED TO METHOD LEVEL: from app.models import (
    User,
    UserSession,
    PasswordResetToken,
    Permission,
    AuditLog,
    UserPreference
)
from app.security.password_utils import hash_password, verify_password

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for user-related database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # =====================================================
    # User CRUD Operations
    # =====================================================
    
    def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user"""
        try:
            # Hash password if provided
            if 'password' in user_data:
                user_data['password_hash'] = hash_password(user_data.pop('password'))
            
            # Create user
            user = User(**user_data)
            self.db.add(user)
            
            # Create default preferences
            preferences = UserPreference(user_id=user.id)
            self.db.add(preferences)
            
            # Assign default UserRole
            if 'role_name' in user_data:
                UserRole = self.db.query(UserRole).filter_by(name=user_data['role_name']).first()
                if UserRole:
                    user.roles.append(UserRole)
            
            self.db.commit()
            self.db.refresh(user)
            
            # Create audit log
            self._create_audit_log(
                user_id=user.id, action="user_created", resource_type="user", resource_id=str(user.id), new_values={"email": user.email, "username": user.username}
            )
            
            return user
            
        except IntegrityError as e:
            self.db.rollback()
            if 'email' in str(e.orig):
                raise ValueError("Email already exists")
            elif 'username' in str(e.orig):
                raise ValueError("Username already exists")
            raise
    
    def get_user_by_id(self, user_id: UUID, include_roles: bool = False) -> Optional[User]:
        """Get user by ID"""
        query = self.db.query(User)
        if include_roles:
            query = query.options(joinedload(User.roles).joinedload(Role.permissions))
        return query.filter(User.id == user_id).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(
            func.lower(User.email) == func.lower(email)
        ).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.db.query(User).filter(
            func.lower(User.username) == func.lower(username)
        ).first()
    
    def get_users(
        self, skip: int = 0, limit: int = 100, search: Optional[str] = None, UserRole: Optional[str] = None, is_active: Optional[bool] = None, department: Optional[str] = None
    ) -> List[User]:
        """Get users with filters"""
        query = self.db.query(User)
        
        # Apply filters
        if search:
            search_filter = f"%{search}%"
            query = query.filter(
                or_(
                    User.email.ilike(search_filter), User.username.ilike(search_filter), User.first_name.ilike(search_filter), User.last_name.ilike(search_filter)
                )
            )
        
        if UserRole:
            query = query.filter(User.UserRole == UserRole)
        
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        if department:
            query = query.filter(User.department == department)
        
        return query.offset(skip).limit(limit).all()
    
    def update_user(self, user_id: UUID, update_data: Dict[str, Any]) -> Optional[User]:
        """Update user"""
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        
        # Track old values for audit
        old_values = {
            key: getattr(user, key) 
            for key in update_data.keys() 
            if hasattr(user, key)
        }
        
        # Handle password update
        if 'password' in update_data:
            update_data['password_hash'] = hash_password(update_data.pop('password'))
        
        # Update fields
        for key, value in update_data.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        user.updated_at = datetime.utcnow()
        
        try:
            self.db.commit()
            self.db.refresh(user)
            
            # Create audit log
            self._create_audit_log(
                user_id=user_id, action="user_updated", resource_type="user", resource_id=str(user_id), old_values=old_values, new_values=update_data
            )
            
            return user
        except IntegrityError:
            self.db.rollback()
            raise
    
    def delete_user(self, user_id: UUID) -> bool:
        """Delete user (soft, delete)"""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_active = False
        user.updated_at = datetime.utcnow()
        
        # Revoke all sessions
        self.db.query(UserSession).filter(
            UserSession.user_id == user_id, UserSession.is_active == True
        ).update({"is_active": False, "revoked_at": datetime.utcnow()})
        
        self.db.commit()
        
        # Create audit log
        self._create_audit_log(
            user_id=user_id, action="user_deleted", resource_type="user", resource_id=str(user_id)
        )
        
        return True
    
    # =====================================================
    # Authentication Operations
    # =====================================================
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        user = self.get_user_by_email(email)
        if not user:
            return None
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            return None
        
        # Verify password
        if not verify_password(password, user.password_hash):
            # Increment failed login attempts
            user.failed_login_attempts += 1
            
            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)
            
            self.db.commit()
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        self.db.commit()
        
        return user
    
    def create_session(
        self, user_id: UUID, token_hash: str, expires_at: datetime, refresh_token_hash: Optional[str] = None, refresh_expires_at: Optional[datetime] = None, ip_address: Optional[str] = None, user_agent: Optional[str] = None, device_id: Optional[str] = None
    ) -> UserSession:
        """Create user session"""
        session = UserSession(
            user_id=user_id, token_hash=token_hash, expires_at=expires_at, refresh_token_hash=refresh_token_hash, refresh_expires_at=refresh_expires_at, ip_address=ip_address, user_agent=user_agent, device_id=device_id
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session
    
    def get_session_by_token(self, token_hash: str) -> Optional[UserSession]:
        """Get session by token hash"""
        return self.db.query(UserSession).filter(
            UserSession.token_hash == token_hash, UserSession.is_active == True, UserSession.expires_at > datetime.utcnow()
        ).first()
    
    def revoke_session(self, session_id: UUID) -> bool:
        """Revoke a session"""
        session = self.db.query(UserSession).filter(
            UserSession.id == session_id
        ).first()
        
        if not session:
            return False
        
        session.is_active = False
        session.revoked_at = datetime.utcnow()
        self.db.commit()
        return True
    
    def revoke_all_user_sessions(self, user_id: UUID) -> int:
        """Revoke all sessions for a user"""
        count = self.db.query(UserSession).filter(
            UserSession.user_id == user_id, UserSession.is_active == True
        ).update({"is_active": False, "revoked_at": datetime.utcnow()})
        
        self.db.commit()
        return count
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        count = self.db.query(UserSession).filter(
            UserSession.expires_at < datetime.utcnow()
        ).delete()
        
        self.db.commit()
        return count
    
    # =====================================================
    # Password Reset Operations
    # =====================================================
    
    def create_password_reset_token(
        self, user_id: UUID, token_hash: str, expires_at: datetime, ip_address: Optional[str] = None
    ) -> PasswordResetToken:
        """Create password reset token"""
        # Invalidate existing tokens
        self.db.query(PasswordResetToken).filter(
            PasswordResetToken.user_id == user_id, PasswordResetToken.used_at.is_(None)
        ).update({"used_at": datetime.utcnow()})
        
        # Create new token
        token = PasswordResetToken(
            user_id=user_id, token_hash=token_hash, expires_at=expires_at, ip_address=ip_address
        )
        self.db.add(token)
        self.db.commit()
        self.db.refresh(token)
        
        return token
    
    def get_password_reset_token(self, token_hash: str) -> Optional[PasswordResetToken]:
        """Get valid password reset token"""
        return self.db.query(PasswordResetToken).filter(
            PasswordResetToken.token_hash == token_hash, PasswordResetToken.used_at.is_(None), PasswordResetToken.expires_at > datetime.utcnow()
        ).first()
    
    def use_password_reset_token(self, token_hash: str, new_password: str) -> Optional[User]:
        """Use password reset token and update password"""
        token = self.get_password_reset_token(token_hash)
        if not token:
            return None
        
        # Update password
        user = self.get_user_by_id(token.user_id)
        if not user:
            return None
        
        user.password_hash = hash_password(new_password)
        user.failed_login_attempts = 0
        user.locked_until = None
        
        # Mark token as used
        token.used_at = datetime.utcnow()
        
        # Revoke all sessions
        self.revoke_all_user_sessions(user.id)
        
        self.db.commit()
        
        # Create audit log
        self._create_audit_log(
            user_id=user.id, action="password_reset", resource_type="user", resource_id=str(user.id)
        )
        
        return user
    
    # =====================================================
    # User Preferences Operations
    # =====================================================
    
    def get_user_preferences(self, user_id: UUID) -> Optional[UserPreference]:
        """Get user preferences"""
        return self.db.query(UserPreference).filter(
            UserPreference.user_id == user_id
        ).first()
    
    def update_user_preferences(
        self, user_id: UUID, preferences_data: Dict[str, Any]
    ) -> Optional[UserPreference]:
        """Update user preferences"""
        preferences = self.get_user_preferences(user_id)
        if not preferences:
            # Create if doesn't exist
            preferences = UserPreference(user_id=user_id, **preferences_data)
            self.db.add(preferences)
        else:
            # Update existing
            for key, value in preferences_data.items():
                if hasattr(preferences, key):
                    setattr(preferences, key, value)
            preferences.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(preferences)
        return preferences
    
    # =====================================================
    # UserRole and Permission Operations
    # =====================================================
    
    def assign_role_to_user(
        self, user_id: UUID, role_id: UUID, assigned_by: UUID, expires_at: Optional[datetime] = None
    ) -> bool:
        """Assign UserRole to user"""
        user = self.get_user_by_id(user_id)
        UserRole = self.db.query(UserRole).filter(Role.id == role_id).first()
        
        if not user or not UserRole:
            return False
        
        # Check if already assigned
        if UserRole in user.roles:
            return True
        
        user.roles.append(UserRole)
        self.db.commit()
        
        # Create audit log
        self._create_audit_log(
            user_id=assigned_by, action="role_assigned", resource_type="user", resource_id=str(user_id), new_values={"role_id": str(role_id), "role_name": Role.name}
        )
        
        return True
    
    def remove_role_from_user(self, user_id: UUID, role_id: UUID, removed_by: UUID) -> bool:
        """Remove UserRole from user"""
        user = self.get_user_by_id(user_id, include_roles=True)
        if not user:
            return False
        
        UserRole = next((r for r in user.roles if r.id == role_id), None)
        if not UserRole:
            return False
        
        user.roles.remove(UserRole)
        self.db.commit()
        
        # Create audit log
        self._create_audit_log(
            user_id=removed_by, action="role_removed", resource_type="user", resource_id=str(user_id), old_values={"role_id": str(role_id), "role_name": Role.name}
        )
        
        return True
    
    def get_user_permissions(self, user_id: UUID) -> List[Permission]:
        """Get all permissions for a user through their roles"""
        user = self.get_user_by_id(user_id, include_roles=True)
        if not user:
            return []
        
        permissions = []
        for UserRole in user.roles:
            permissions.extend(Role.permissions)
        
        # Remove duplicates
        return list({p.id: p for p in permissions}.values())
    
    def user_has_permission(self, user_id: UUID, resource: str, action: str) -> bool:
        """Check if user has specific permission"""
        permissions = self.get_user_permissions(user_id)
        return any(
            p.resource == resource and p.action == action
            for p in permissions
        )
    
    # =====================================================
    # Utility Methods
    # =====================================================
    
    def _create_audit_log(
        self, user_id: UUID, action: str, resource_type: str, resource_id: str, old_values: Optional[Dict[str, Any]] = None, new_values: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None
    ):
        """Create audit log entry"""
        try:
            audit_log = AuditLog(
                user_id=user_id, action=action, resource_type=resource_type, resource_id=resource_id, old_values=old_values, new_values=new_values, metadata=metadata
            )
            self.db.add(audit_log)
            self.db.commit()
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            self.db.rollback()
    
    def get_user_stats(self, user_id: UUID) -> Dict[str, Any]:
        """Get user statistics"""
        # MOVED TO METHOD LEVEL: from app.models import NaturalLanguageQuery, SavedQuery, Dashboard
        
        stats = {
            "total_queries": self.db.query(NaturalLanguageQuery).filter(
                NaturalLanguageQuery.user_id == user_id
            ).count(),
            "saved_queries": self.db.query(SavedQuery).filter(
                SavedQuery.user_id == user_id
            ).count(),
            "dashboards": self.db.query(Dashboard).filter(
                Dashboard.created_by == user_id
            ).count(),
            "last_activity": None
        }
        
        # Get last activity
        last_query = self.db.query(NaturalLanguageQuery.created_at).filter(
            NaturalLanguageQuery.user_id == user_id
        ).order_by(NaturalLanguageQuery.created_at.desc()).first()
        
        if last_query:
            stats["last_activity"] = last_query[0]
        
        return stats
