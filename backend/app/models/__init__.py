"""
Database Models Package
Exports all database models for easy importing

IMPORTANT: Import order matters for SQLAlchemy relationships
"""

# Import Base first
from .base import Base

# Import user models - uncomment Role, Permission, AuditLog
from .user import (
    User, UserSession, PasswordResetToken, UserPreference,
    Role, Permission, AuditLog
)

# Import query models
from .query import (
    NaturalLanguageQuery, SavedQuery, QueryResultCache,
    QueryTemplate, QuerySuggestion
)

# Import visualization models
from .visualization import (
    ChartType, Chart, SavedChart, Dashboard, DashboardChart
)

# Import supply chain models
from .supply_chain import (
    Supplier, Product, Material, Inventory,
    Order, OrderItem, Shipment
)

# Try to import additional supply chain models if they exist
try:
    from .supply_chain import (
        SupplierTier, SupplierRelationship,
        ProductMaterial, InventoryHistory, ShipmentItem
    )
except ImportError:
    pass

# Import analytics models
from .analytics import (
    AnalyticsResult, ScheduledAnalytic, AnalyticsTemplate,
    Report, ReportSchedule,
    SupplierPerformanceMetric, InventoryMetric,
    DeliveryPerformance, RiskAssessment, ComplianceCheck,
    ABCAnalysisResult, ForecastResult, SafetyStockCalculation
)

# Try to import network models
try:
    from .analytics import (
        SupplyChainNetwork, NetworkNode, NetworkEdge,
        BottleneckAnalysis, RiskPropagationScenario, DisruptionImpact
    )
except ImportError:
    pass

# Import extended models with correct relative imports
try:
    from .extended_models import (
        ForecastModel,
        ExtendedAnalyticsMetric,  # Note: Changed from AnalyticsMetric
        ExportJob,
        ReportTemplate,
        ExtendedReport,  # Note: Changed from Report to avoid conflict
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

# Build __all__ dynamically based on what actually imported
__all__ = ['Base']

# Add all imported models to __all__
import sys
current_module = sys.modules[__name__]

# List of all possible models
all_models = [
    # User models
    'User', 'UserSession', 'PasswordResetToken', 'UserPreference',
    'Role', 'Permission', 'AuditLog',
    
    # Query models
    'NaturalLanguageQuery', 'SavedQuery', 'QueryResultCache',
    'QueryTemplate', 'QuerySuggestion',
    
    # Visualization models
    'ChartType', 'Chart', 'SavedChart', 'Dashboard', 'DashboardChart',
    
    # Supply chain models
    'Supplier', 'Product', 'Material', 'Inventory',
    'Order', 'OrderItem', 'Shipment',
    'SupplierTier', 'SupplierRelationship',
    'ProductMaterial', 'InventoryHistory', 'ShipmentItem',
    
    # Analytics models
    'AnalyticsResult', 'ScheduledAnalytic', 'AnalyticsTemplate',
    'Report', 'ReportSchedule',
    'SupplierPerformanceMetric', 'InventoryMetric',
    'DeliveryPerformance', 'RiskAssessment', 'ComplianceCheck',
    'ABCAnalysisResult', 'ForecastResult', 'SafetyStockCalculation',
    'SupplyChainNetwork', 'NetworkNode', 'NetworkEdge',
    'BottleneckAnalysis', 'RiskPropagationScenario', 'DisruptionImpact',
    
    # Extended models
    'ForecastModel', 'ExtendedAnalyticsMetric', 'ExportJob',
    'ReportTemplate', 'ExtendedReport', 'ScheduledReport',
    'NotificationSetting', 'WidgetType',
    
    # System models
    'SystemSetting'
]

# Only add to __all__ if the model was successfully imported
for model_name in all_models:
    if hasattr(current_module, model_name):
        __all__.append(model_name)