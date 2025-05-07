"""
Role-Based Access Control (RBAC) system.

This module provides functions for role-based permission checking, role management,
and permission evaluation.
"""

from typing import Dict, List, Any, Optional, Set, Union
import re
from fastapi import HTTPException, status

from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Permission cache - Role to permissions mapping
_role_permissions: Dict[str, Set[str]] = {}

# System roles with predefined permissions
SYSTEM_ROLES = {
    "admin": {
        "description": "System administrator with full access",
        "permissions": ["*"],  # Wildcard means all permissions
    },
    "client_admin": {
        "description": "Client administrator with limited admin access",
        "permissions": [
            "admin:client:*", 
            "admin:users:*", 
            "queries:*", 
            "visualizations:*", 
            "analytics:*", 
            "database:*",
        ],
    },
    "analyst": {
        "description": "Data analyst with query and visualization access",
        "permissions": [
            "queries:*", 
            "visualizations:*", 
            "analytics:*", 
            "database:read",
        ],
    },
    "viewer": {
        "description": "Read-only access to dashboards and visualizations",
        "permissions": [
            "queries:execute", 
            "visualizations:view",
        ],
    },
}

# Initialize roles
def initialize_roles() -> None:
    """
    Initialize the role permissions from predefined system roles.
    """
    global _role_permissions
    
    # Add system roles
    for role, role_info in SYSTEM_ROLES.items():
        _role_permissions[role] = set(role_info["permissions"])
    
    logger.info("RBAC roles initialized")

# Permission checking
def check_permission(role: str, permission: str) -> bool:
    """
    Check if a role has a specific permission.
    
    Args:
        role: Role name
        permission: Permission to check
        
    Returns:
        True if role has permission, False otherwise
        
    Raises:
        HTTPException if role doesn't have the permission
    """
    if has_permission(role, permission):
        return True
    
    logger.warning(f"Permission denied: {role} does not have {permission}")
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You do not have permission to perform this action"
    )

def has_permission(role: str, permission: str) -> bool:
    """
    Check if a role has a specific permission without raising an exception.
    
    Args:
        role: Role name
        permission: Permission to check
        
    Returns:
        True if role has permission, False otherwise
    """
    # Make sure roles are initialized
    if not _role_permissions:
        initialize_roles()
    
    # Special case for admin role (has all permissions)
    if role == "admin":
        return True
    
    # Get role permissions
    role_perms = _role_permissions.get(role, set())
    
    # Check exact permission
    if permission in role_perms:
        return True
    
    # Check wildcard permissions
    for role_perm in role_perms:
        # If the role has a wildcard permission that matches
        if _match_wildcard_permission(role_perm, permission):
            return True
    
    return False

def _match_wildcard_permission(role_perm: str, required_perm: str) -> bool:
    """
    Check if a role permission with wildcards matches a required permission.
    
    Args:
        role_perm: Role permission (may contain wildcards)
        required_perm: Required permission to check
        
    Returns:
        True if the role permission matches the required permission
    """
    # Literal wildcard "*" matches everything
    if role_perm == "*":
        return True
    
    # Convert to regex pattern (e.g., "admin:*" becomes "^admin:.*$")
    pattern = "^" + role_perm.replace("*", ".*") + "$"
    return bool(re.match(pattern, required_perm))

# Role management
def get_user_permissions(role: str) -> List[str]:
    """
    Get all permissions for a role.
    
    Args:
        role: Role name
        
    Returns:
        List of permission strings
    """
    # Make sure roles are initialized
    if not _role_permissions:
        initialize_roles()
    
    # For admin, return all defined permissions
    if role == "admin":
        # Collect all unique permissions from all roles
        all_perms = set()
        for perms in _role_permissions.values():
            all_perms.update(perms)
        return sorted(list(all_perms))
    
    # Get permissions for the role
    return sorted(list(_role_permissions.get(role, set())))

def get_role_permissions(role: str) -> List[str]:
    """
    Get the permissions assigned to a role.
    
    Args:
        role: Role name
        
    Returns:
        List of permission strings
    """
    # Make sure roles are initialized
    if not _role_permissions:
        initialize_roles()
    
    return sorted(list(_role_permissions.get(role, set())))

async def update_role_permissions(role: str, permissions: List[str]) -> bool:
    """
    Update the permissions for a role.
    
    Args:
        role: Role name
        permissions: New list of permissions
        
    Returns:
        True if successful, False otherwise
    """
    global _role_permissions
    
    try:
        # Check if role exists
        if role not in _role_permissions and role not in SYSTEM_ROLES:
            # Create new role
            _role_permissions[role] = set(permissions)
            logger.info(f"Created new role: {role}")
        else:
            # Update existing role
            _role_permissions[role] = set(permissions)
            logger.info(f"Updated permissions for role: {role}")
        
        return True
    except Exception as e:
        logger.error(f"Error updating role permissions: {str(e)}")
        return False

async def delete_role(role: str) -> bool:
    """
    Delete a role.
    
    Args:
        role: Role name
        
    Returns:
        True if successful, False otherwise
    """
    global _role_permissions
    
    try:
        # Check if role exists and is not a system role
        if role in SYSTEM_ROLES:
            logger.warning(f"Cannot delete system role: {role}")
            return False
        
        if role in _role_permissions:
            del _role_permissions[role]
            logger.info(f"Deleted role: {role}")
            return True
        
        logger.warning(f"Role not found: {role}")
        return False
    except Exception as e:
        logger.error(f"Error deleting role: {str(e)}")
        return False

# Permission evaluation
def evaluate_complex_permission(
    role: str, 
    resource_type: str, 
    action: str, 
    resource_owner_id: Optional[str] = None,
    user_id: Optional[str] = None,
    client_id: Optional[str] = None,
    resource_client_id: Optional[str] = None
) -> bool:
    """
    Evaluate a complex permission scenario.
    
    Args:
        role: User's role
        resource_type: Type of resource (e.g., "dashboard", "query")
        action: Action being performed (e.g., "view", "edit", "delete")
        resource_owner_id: ID of the resource owner
        user_id: ID of the user making the request
        client_id: Client ID of the user making the request
        resource_client_id: Client ID of the resource
        
    Returns:
        True if permission is granted, False otherwise
    """
    # Make sure roles are initialized
    if not _role_permissions:
        initialize_roles()
    
    # Admin always has permission
    if role == "admin":
        return True
    
    # Define the permission string
    permission = f"{resource_type}:{action}"
    
    # Check basic permission
    has_basic_perm = has_permission(role, permission)
    
    # If no basic permission, deny
    if not has_basic_perm:
        return False
    
    # Owner can always access their own resources
    if resource_owner_id and user_id and resource_owner_id == user_id:
        return True
    
    # Client-specific checks
    if client_id and resource_client_id:
        # Users can only access resources from their own client
        # (except admins, who have already been approved)
        if client_id != resource_client_id:
            return False
    
    # Resource-specific checks
    if resource_type == "dashboard" and action == "edit":
        # Only owner or client_admin can edit dashboards
        if resource_owner_id and user_id and resource_owner_id != user_id:
            return role == "client_admin"
    
    # Default to the basic permission check result
    return has_basic_perm