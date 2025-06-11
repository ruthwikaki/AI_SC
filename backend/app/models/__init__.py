"""
Models package initialization
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
    ChartType, Chart, SavedChart, Dashboard, DashboardChart, DashboardWidget
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
from .extended_models import (
    ForecastModel,
    ExtendedAnalyticsMetric,  # Renamed from AnalyticsMetric
    ExportJob,
    ReportTemplate,
    ExtendedReport,  # Renamed from Report
    ScheduledReport,
    SystemSetting,
    NotificationSetting,
    WidgetType,
    SyncHistory)

__all__ = [
    # User models
    "User", "UserSession", "PasswordResetToken", "UserPreference",
    "Role", "Permission", "AuditLog",
    
    # Query models
    "NaturalLanguageQuery", "SavedQuery", "QueryResultCache",
    "QueryTemplate", "QuerySuggestion",
    
    # Visualization models
    "ChartType", "Chart", "SavedChart", "Dashboard", "DashboardChart", "DashboardWidget",
    
    # Supply chain models
    "Supplier", "SupplierTier", "SupplierRelationship",
    "Product", "Material", "ProductMaterial",
    "Inventory", "InventoryHistory",
    "Order", "OrderItem", "Shipment", "ShipmentItem",
    
    # Analytics models
    "AnalyticsResult", "ScheduledAnalytic", "AnalyticsTemplate",
    "Report", "ReportSchedule",
    "SupplierPerformanceMetric", "InventoryMetric",
    "DeliveryPerformance", "RiskAssessment", "ComplianceCheck",
    "ABCAnalysisResult", "ForecastResult", "SafetyStockCalculation",
    "SupplyChainNetwork", "NetworkNode", "NetworkEdge",
    "BottleneckAnalysis", "RiskPropagationScenario", "DisruptionImpact",
    
    # Extended models
    "ForecastModel", "ExtendedAnalyticsMetric", "ExportJob",
    "ReportTemplate", "ExtendedReport", "ScheduledReport",
    "SystemSetting", "NotificationSetting", "WidgetType"
]