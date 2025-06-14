from typing import List, Dict, Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Default permissions by role
ROLE_PERMISSIONS = {
    "admin": [
        "users:read", "users:write", "users:delete",
        "queries:read", "queries:write", "queries:delete",
        "analytics:read", "analytics:write", "analytics:delete",
        "database:read", "database:write", "database:delete",
        "visualizations:read", "visualizations:write", "visualizations:delete",
        "reports:read", "reports:write", "reports:delete",
        "settings:read", "settings:write",
        "audit:read", "system:manage"
    ],
    "analyst": [
        "queries:read", "queries:write",
        "analytics:read", "analytics:write",
        "database:read",
        "visualizations:read", "visualizations:write",
        "reports:read", "reports:write"
    ],
    "viewer": [
        "queries:read",
        "analytics:read",
        "database:read",
        "visualizations:read",
        "reports:read"
    ],
    "user": [
        "queries:read",
        "analytics:read",
        "visualizations:read"
    ]
}

def get_user_permissions(role: str) -> List[str]:
    """Get permissions for a given role"""
    return ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["user"])

def check_permission(user_role: str, required_permission: str) -> bool:
    """Check if a role has a specific permission"""
    permissions = get_user_permissions(user_role)
    return required_permission in permissions

def get_role_permissions(role: str) -> List[str]:
    """Get permissions for a specific role (alias for get_user_permissions)"""
    return get_user_permissions(role)

def update_role_permissions(role: str, permissions: List[str]) -> bool:
    """Update permissions for a role (in-memory only for now)"""
    try:
        if role in ROLE_PERMISSIONS:
            ROLE_PERMISSIONS[role] = permissions
            logger.info(f"Updated permissions for role {role}")
            return True
        else:
            logger.warning(f"Role {role} not found")
            return False
    except Exception as e:
        logger.error(f"Error updating role permissions: {e}")
        return False

def check_permissions(user_role: str, required_permissions: List[str]) -> bool:
    """Check if a role has all required permissions"""
    permissions = get_user_permissions(user_role)
    return all(perm in permissions for perm in required_permissions)

def get_role_hierarchy() -> Dict[str, int]:
    """Get role hierarchy with priority levels"""
    return {
        "admin": 100,
        "analyst": 50,
        "viewer": 20,
        "user": 10
    }

def is_role_higher_or_equal(user_role: str, required_role: str) -> bool:
    """Check if user role is higher or equal to required role"""
    hierarchy = get_role_hierarchy()
    user_level = hierarchy.get(user_role, 0)
    required_level = hierarchy.get(required_role, 100)
    return user_level >= required_level

class RBACManager:
    """Role-Based Access Control Manager"""
    
    @staticmethod
    def check_resource_access(user_role: str, resource: str, action: str) -> bool:
        """Check if a role can perform an action on a resource"""
        permission = f"{resource}:{action}"
        return check_permission(user_role, permission)
    
    @staticmethod
    def get_accessible_resources(user_role: str) -> Dict[str, List[str]]:
        """Get all accessible resources and actions for a role"""
        permissions = get_user_permissions(user_role)
        resources = {}
        
        for permission in permissions:
            if ":" in permission:
                resource, action = permission.split(":", 1)
                if resource not in resources:
                    resources[resource] = []
                resources[resource].append(action)
        
        return resources
    
    @staticmethod
    def validate_role(role: str) -> bool:
        """Validate if a role exists"""
        return role in ROLE_PERMISSIONS

# Create global instance
rbac_manager = RBACManager()