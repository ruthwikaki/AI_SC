"""
Database Models Package
Exports all database models for easy importing
"""

from .user import (
    User, UserSession, PasswordResetToken, UserPreference, UserProfile, 
    Role, Permission, AuditLog
)

from .query import (
    Query,QueryResult,QueryHistory,  # Basic models
    NaturalLanguageQuery,SQLQuery,QueryExecution,  # SQL generation models
    SavedQuery,QueryTemplate,QuerySuggestion,QueryResultCache  # Template and cache models
)
    NaturalLanguageQuery, SavedQuery, QueryResultCache,
    QueryTemplate, QuerySuggestion
)

from .visualization import (
    ChartType, Chart, SavedChart, Dashboard, DashboardChart
)

from .supply_chain import (
    Inventory,
    InventoryHistory,
    Material,
    Order,
    OrderItem,
    Product,
    ProductMaterial,
    Shipment,
    ShipmentItem,
    Supplier,
    SupplierRelationship,
    SupplierTier,
    Warehouse
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

# Import extended models
from .extended_models import (
    ForecastModel,
    ExtendedAnalyticsMetric,
    ExportJob,
    ScheduledReport,
    ReportTemplate,
    ExtendedReport,
    NotificationSetting,
    WidgetType,
    DataSource,
    APIKey,
    SystemSetting
)

# Re-export all models
__all__ = [
    "ABCAnalysisResult",
    "APIKey",
    "AnalyticsResult",
    "AnalyticsTemplate",
    "BottleneckAnalysis",
    "Chart",
    "ChartType",
    "ComplianceCheck",
    "Dashboard",
    "DashboardChart",
    "DataSource",
    "DeliveryPerformance",
    "DisruptionImpact",
    "ExportJob",
    "ExtendedAnalyticsMetric",
    "ForecastModel",
    "ForecastResult",
    "Inventory",
    "InventoryHistory",
    "InventoryMetric",
    "Material",
    "NaturalLanguageQuery",
    "NetworkEdge",
    "NetworkNode",
    "NotificationSetting",
    "Order",
    "OrderItem",
    "PasswordResetToken",
    "Product",
    "ProductMaterial",
    "Query",
    "QueryExecution",
    "QueryHistory",
    "QueryResult",
    "QueryResultCache",
    "QuerySuggestion",
    "QueryTemplate",
    "Report",
    "ReportSchedule",
    "ReportTemplate",
    "RiskAssessment",
    "RiskPropagationScenario",
    "SQLQuery",
    "SafetyStockCalculation",
    "SavedChart",
    "SavedQuery",
    "ScheduledAnalytic",
    "ScheduledReport",
    "Shipment",
    "ShipmentItem",
    "Supplier",
    "SupplierPerformanceMetric",
    "SupplierRelationship",
    "SupplierTier",
    "SupplyChainNetwork",
    "SystemSetting",
    "User",
    "UserPreference",
    "UserProfile",
    "UserSession",
    "Warehouse",
    "WidgetType"
]