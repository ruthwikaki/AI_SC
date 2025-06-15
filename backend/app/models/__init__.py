"""
Database Models Package
Exports all database models for easy importing
"""

# Import Base first
from .base import Base

# User models - now including Role, Permission, AuditLog
from .user import (
    User, UserSession, PasswordResetToken, UserPreference,
    Role, Permission, AuditLog
)

# Add UserProfile alias
from .user import UserPreference as UserProfile

# Query models
from .query import (
    Query, QueryResult, QueryHistory,
    NaturalLanguageQuery, SavedQuery, QueryResultCache,
    QueryTemplate, QuerySuggestion, SQLQuery, QueryExecution
)

# Visualization models
from .visualization import (
    ChartType, Chart, SavedChart, Dashboard, DashboardChart
)

# Supply chain models
from .supply_chain import (
    Supplier, SupplierTier, SupplierRelationship,
    Product, Material, ProductMaterial,
    Inventory, InventoryHistory,
    Order, OrderItem, Shipment, ShipmentItem,
    Warehouse
)

# Analytics models
from .analytics import (
    AnalyticsResult, ScheduledAnalytic, AnalyticsTemplate,
    Report, ReportSchedule,
    SupplierPerformanceMetric, InventoryMetric,
    DeliveryPerformance, RiskAssessment, ComplianceCheck,
    ABCAnalysisResult, ForecastResult, SafetyStockCalculation,
    SupplyChainNetwork, NetworkNode, NetworkEdge,
    BottleneckAnalysis, RiskPropagationScenario, DisruptionImpact
)

# Extended models
from .extended_models import (
    ForecastModel,
    ExtendedAnalyticsMetric,
    ExportJob,
    ReportTemplate,
    ExtendedReport,
    ScheduledReport,
    SystemSetting,
    NotificationSetting,
    WidgetType,
    DataSource,
    APIKey
)

# Re-export all models
__all__ = [
    # Base
    'Base',
    
    # User models
    'User', 'UserSession', 'PasswordResetToken', 'UserPreference', 'UserProfile',
    'Role', 'Permission', 'AuditLog',
    
    # Query models
    'Query', 'QueryResult', 'QueryHistory',
    'NaturalLanguageQuery', 'SavedQuery', 'QueryResultCache',
    'QueryTemplate', 'QuerySuggestion', 'SQLQuery', 'QueryExecution',
    
    # Visualization models
    'ChartType', 'Chart', 'SavedChart', 'Dashboard', 'DashboardChart',
    
    # Supply chain models
    'Supplier', 'SupplierTier', 'SupplierRelationship',
    'Product', 'Material', 'ProductMaterial',
    'Inventory', 'InventoryHistory',
    'Order', 'OrderItem', 'Shipment', 'ShipmentItem',
    'Warehouse',
    
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
    'SystemSetting', 'NotificationSetting', 'WidgetType',
    'DataSource', 'APIKey'
]
