"""
Database Models Package
Exports all database models for easy importing
"""

from .user import (
    User, UserSession, PasswordResetToken, UserPreference,
    Role, Permission, AuditLog
)

from .query import (
    NaturalLanguageQuery, SavedQuery, QueryResultCache,
    QueryTemplate, QuerySuggestion
)

from .visualization import (
    ChartType, Chart, SavedChart, Dashboard, DashboardChart
)

from .supply_chain import (
    Supplier, SupplierTier, SupplierRelationship,
    Product, Material, ProductMaterial,
    Inventory, InventoryHistory,
    Order, OrderItem, Shipment, ShipmentItem
)

from .analytics import (
    AnalyticsResult, ScheduledAnalytic, AnalyticsTemplate,
    Report, ReportSchedule,
    SupplierPerformanceMetric, InventoryMetric,
    DeliveryPerformance, RiskAssessment, ComplianceCheck,
    ABCAnalysisResult, ForecastResult, SafetyStockCalculation,
    SupplyChainNetwork, NetworkNode, NetworkEdge,
    BottleneckAnalysis, RiskPropagationScenario, DisruptionImpact
)

# Try to import extended models
try:
    from .extended_models import (
        ForecastModel,
        AnalyticsMetric,
        ExportJob,
        ReportTemplate,
        ScheduledReport,
        SystemSetting,
        NotificationSetting,
        WidgetType
    )
    HAS_EXTENDED_MODELS = True
except ImportError:
    HAS_EXTENDED_MODELS = False

# Re-export all models
__all__ = [
    # User models
    'User', 'UserSession', 'PasswordResetToken', 'UserPreference',
    'Role', 'Permission', 'AuditLog',
    
    # Query models
    'NaturalLanguageQuery', 'SavedQuery', 'QueryResultCache',
    'QueryTemplate', 'QuerySuggestion',
    
    # Visualization models
    'ChartType', 'Chart', 'SavedChart', 'Dashboard', 'DashboardChart',
    
    # Supply chain models
    'Supplier', 'SupplierTier', 'SupplierRelationship',
    'Product', 'Material', 'ProductMaterial',
    'Inventory', 'InventoryHistory',
    'Order', 'OrderItem', 'Shipment', 'ShipmentItem',
    
    # Analytics models
    'AnalyticsResult', 'ScheduledAnalytic', 'AnalyticsTemplate',
    'Report', 'ReportSchedule',
    'SupplierPerformanceMetric', 'InventoryMetric',
    'DeliveryPerformance', 'RiskAssessment', 'ComplianceCheck',
    'ABCAnalysisResult', 'ForecastResult', 'SafetyStockCalculation',
    'SupplyChainNetwork', 'NetworkNode', 'NetworkEdge',
    'BottleneckAnalysis', 'RiskPropagationScenario', 'DisruptionImpact',
]

if HAS_EXTENDED_MODELS:
    __all__.extend([
        'ForecastModel', 'AnalyticsMetric', 'ExportJob',
        'ReportTemplate', 'ScheduledReport',
        'SystemSetting', 'NotificationSetting', 'WidgetType'
    ])
