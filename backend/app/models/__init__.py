"""
Models package - with all necessary exports
"""

# Import base first
from .base import Base

# Import system models
try:
    from .system import SystemSetting
except ImportError:
    SystemSetting = None

# Import all user models and aliases
from .user import (
    User, UserSession, UserRole, UserPermission, 
    RolePermission, UserRoleAssignment, UserProfile, UserActivity,
    PasswordResetToken, EmailVerificationToken,
    Role, Permission  # Aliases
)

# Import core business models
try:
    from .supply_chain import (
        Supplier, Product, Inventory, Order, OrderItem, 
        Shipment, Customer, Warehouse, Material,
        InventoryHistory, ABCAnalysisResult, SafetyStockCalculation
    )
except ImportError as e:
    print(f"Warning: Could not import supply_chain models: {e}")

# Import query models
try:
    from .query import NaturalLanguageQuery, SQLQuery, QueryExecution, SavedQuery, QueryResultCache, QueryTemplate, QuerySuggestion
except ImportError as e:
    print(f"Warning: Could not import query models: {e}")

# Import visualization models
try:
    from .visualization import Chart, Dashboard, ChartData, DashboardWidget, VisualizationTemplate
except ImportError as e:
    print(f"Warning: Could not import visualization models: {e}")

# Import analytics models
try:
    from .analytics import *
except ImportError as e:
    print(f"Warning: Could not import analytics models: {e}")

# Import extended models
try:
    from .extended_models import (
        ForecastModel, ExtendedAnalyticsMetric, ExportJob,
        ReportTemplate, Report, ScheduledReport,
        NotificationSetting, WidgetType, DataSource, APIKey, AuditLog
    )
    # Create alias
    AnalyticsMetric = ExtendedAnalyticsMetric
except ImportError as e:
    print(f"Warning: Could not import extended_models: {e}")
    APIKey = None
    Report = None
    AnalyticsMetric = None

# Ensure all names are available
__all__ = [
    # Base
    'Base',
    # System
    'SystemSetting',
    # User models
    'User', 'UserSession', 'UserRole', 'UserPermission',
    'RolePermission', 'UserRoleAssignment', 'UserProfile', 'UserActivity',
    'PasswordResetToken', 'EmailVerificationToken',
    'Role', 'Permission',  # Aliases
    # Supply chain
    'Supplier', 'Product', 'Material', 'Inventory', 'Order', 'OrderItem',
    'Shipment', 'Customer', 'Warehouse', 'InventoryHistory', 
    'ABCAnalysisResult', 'SafetyStockCalculation',
    # Query
    'NaturalLanguageQuery', 'SQLQuery', 'QueryExecution', 'SavedQuery', 
    'QueryResultCache', 'QueryTemplate', 'QuerySuggestion',
    # Visualization
    'Chart', 'Dashboard', 'ChartData', 'DashboardWidget', 'VisualizationTemplate',
    # Extended
    'ForecastModel', 'ExtendedAnalyticsMetric', 'AnalyticsMetric',
    'ExportJob', 'ReportTemplate', 'Report', 'ScheduledReport',
    'NotificationSetting', 'WidgetType', 'DataSource', 'APIKey', 'AuditLog'
]