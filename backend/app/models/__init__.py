"""
Database Models Package
Exports all database models for easy importing
"""

from .base import Base

from .user import (
    User, UserSession, PasswordResetToken, UserPreference,
    Role, Permission, RolePermission, UserRole, AuditLog
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

# Add these imports to your existing backend/app/models/__init__.py:
from .extended_models import (
    ForecastModel,
    AnalyticsMetric,
    ExportJob,
    ReportTemplate,
    ScheduledReport,
    SystemSetting,
    NotificationSetting,
    WidgetType,
    DataSource,
    APIKey,
    AuditLog as ExtendedAuditLog
)

# Re-export all models
__all__ = [
    # Base
    'Base',
    
    # User models
    'User', 'UserSession', 'PasswordResetToken', 'UserPreference',
    'Role', 'Permission', 'RolePermission', 'UserRole', 'AuditLog',
    
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
    
    # Extended models
    'ForecastModel', 'AnalyticsMetric', 'ExportJob',
    'ReportTemplate', 'ScheduledReport',
    'SystemSetting', 'NotificationSetting', 'WidgetType',
    'DataSource', 'APIKey', 'ExtendedAuditLog'
]