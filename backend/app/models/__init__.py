"""
Database Models Package
Exports all database models for easy importing

IMPORTANT: Import order matters for SQLAlchemy relationships
"""

# Import Base first
from .base import Base

# Import user models (no dependencies)
from .user import User, UserSession, PasswordResetToken, UserPreference, Role, Permission, AuditLog

# Import query models (depends on User)
from .query import NaturalLanguageQuery, SavedQuery

# Import visualization models (depends on User and Query)
from .visualization import ChartType, Chart, SavedChart, Dashboard, DashboardChart

# Import supply chain models
from .supply_chain import Supplier, Product, Material, Inventory, Order, OrderItem, Shipment

# Import analytics models
from .analytics import (
    AnalyticsResult, ScheduledAnalytic, AnalyticsTemplate,
    Report, ReportSchedule,
    SupplierPerformanceMetric, InventoryMetric,
    DeliveryPerformance, RiskAssessment, ComplianceCheck,
    ABCAnalysisResult, ForecastResult, SafetyStockCalculation
)

# Import extended models if they exist
try:
    from .extended_models import (
        ForecastModel,
        ExtendedAnalyticsMetric,
        ExportJob,
        ReportTemplate,
        ExtendedReport,
        ScheduledReport,
        NotificationSetting,
        WidgetType
    )
except ImportError:
    pass

# Import system models
try:
    from .system import SystemSetting
except ImportError:
    pass

# List all exported models
__all__ = [
    # Base
    'Base',
    
    # User models
    'User', 'UserSession', 'PasswordResetToken', 'UserPreference',
    'Role', 'Permission', 'AuditLog',
    
    # Query models
    'NaturalLanguageQuery', 'SavedQuery',
    
    # Visualization models
    'ChartType', 'Chart', 'SavedChart', 'Dashboard', 'DashboardChart',
    
    # Supply chain models
    'Supplier', 'Product', 'Material', 'Inventory', 'Order', 'OrderItem', 'Shipment',
    
    # Analytics models
    'AnalyticsResult', 'ScheduledAnalytic', 'AnalyticsTemplate',
    'Report', 'ReportSchedule',
    'SupplierPerformanceMetric', 'InventoryMetric',
    'DeliveryPerformance', 'RiskAssessment', 'ComplianceCheck',
    'ABCAnalysisResult', 'ForecastResult', 'SafetyStockCalculation',
]

# Add extended models to __all__ if they were imported
import sys
current_module = sys.modules[__name__]
extended_models = [
    'ForecastModel', 'ExtendedAnalyticsMetric', 'ExportJob',
    'ReportTemplate', 'ExtendedReport', 'ScheduledReport',
    'NotificationSetting', 'WidgetType', 'SystemSetting'
]

for model in extended_models:
    if hasattr(current_module, model):
        __all__.append(model)
