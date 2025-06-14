"""
Database Models Package
Exports all database models for easy importing
"""

from .user import (
    User, UserSession, PasswordResetToken, 
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
    ReportSchedule,
    SupplierPerformanceMetric, InventoryMetric,
    DeliveryPerformance, RiskAssessment, ComplianceCheck,
    ABCAnalysisResult, ForecastResult, SafetyStockCalculation,
    SupplyChainNetwork, NetworkNode, NetworkEdge,
    BottleneckAnalysis, RiskPropagationScenario, DisruptionImpact
)

# Add these imports to your existing backend/app/models/__init__.py:
from app.models.extended_models import (
    ForecastModel,
    AnalyticsMetric,
    ExportJob,
    ReportTemplate,
    ScheduledReport,
    SystemSetting,
    NotificationSetting,
    WidgetType
)

# Re-export all models
__all__ = [
    # User models
    'User', 'UserSession', 'PasswordResetToken', 
      
    
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
    'SystemSetting', 'NotificationSetting', 'WidgetType'
]