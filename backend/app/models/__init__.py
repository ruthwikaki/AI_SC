"""
Models package - with careful import order to avoid circular dependencies
"""

# Import base first
from .base import Base, BaseModelMixin

# Import system models
try:
    from .system import SystemSetting
except ImportError:
    SystemSetting = None

# Import all user models
from .user import (
    User, UserSession, UserRole, UserPermission, 
    RolePermission, UserRoleAssignment, UserProfile, UserActivity
)

# Import core business models
try:
    from .supply_chain import *
except ImportError as e:
    print(f"Warning: Could not import supply_chain models: {e}")

# Import query models
try:
    from .query import *
except ImportError as e:
    print(f"Warning: Could not import query models: {e}")

# Import visualization models
try:
    from .visualization import *
except ImportError as e:
    print(f"Warning: Could not import visualization models: {e}")

# Import analytics models
try:
    from .analytics import *
except ImportError as e:
    print(f"Warning: Could not import analytics models: {e}")

# Import extended models last
try:
    from .extended_models import *
except ImportError as e:
    print(f"Warning: Could not import extended_models: {e}")

# APIKey might be in extended_models
try:
    from .extended_models import APIKey
except ImportError:
    APIKey = None