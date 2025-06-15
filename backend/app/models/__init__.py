"""
Database Models Package - Comprehensive
"""

from .base import Base

# Import all models with error handling
_all_models = []

# User models
try:
    from .user import (
        User, UserSession, PasswordResetToken, UserPreference,
        Role, Permission, AuditLog
    )
    _all_models.extend(["User", "UserSession", "PasswordResetToken", 
                       "UserPreference", "Role", "Permission", "AuditLog"])
except ImportError as e:
    print(f"Warning: Could not import user models: {e}")

# Query models
try:
    from .query import (
        NaturalLanguageQuery, SavedQuery, QueryResultCache,
        QueryTemplate, QuerySuggestion
    )
    _all_models.extend(["Query", "NaturalLanguageQuery", "SavedQuery", "QueryResultCache",
                       "QueryTemplate", "QuerySuggestion"])
except ImportError:
    pass

# Supply chain models
try:
    from .supply_chain import (Supplier, Product, Warehouse, Inventory, Customer, Order, OrderItem, Shipment, SupplierTier, SupplierRelationship, Material, ProductMaterial, InventoryHistory, ShipmentItem)
    _all_models.extend(["Supplier", "Product", "Warehouse", "Inventory", "Customer", "Order", "OrderItem", "Shipment", "SupplierTier", "SupplierRelationship", "Material", "ProductMaterial", "InventoryHistory", "ShipmentItem"])
except ImportError:
    pass

# Analytics models
try:
    from .analytics import (ABCAnalysisResult, AnalyticsMetric, AnalyticsResult, AnalyticsSummary, AnalyticsTemplate, BottleneckAnalysis, ComplianceCheck, DeliveryPerformance, DisruptionImpact, ForecastResult, InventoryMetric, KPIDefinition, NetworkEdge, NetworkNode, Report, ReportSchedule, RiskAssessment, RiskPropagationScenario, SafetyStockCalculation, ScheduledAnalytic, SupplierPerformanceMetric, SupplyChainNetwork)
    _all_models.extend(["ABCAnalysisResult", "AnalyticsMetric", "AnalyticsResult", "AnalyticsSummary", "AnalyticsTemplate", "BottleneckAnalysis", "ComplianceCheck", "DeliveryPerformance", "DisruptionImpact", "ForecastResult", "InventoryMetric", "KPIDefinition", "NetworkEdge", "NetworkNode", "Report", "ReportSchedule", "RiskAssessment", "RiskPropagationScenario", "SafetyStockCalculation", "ScheduledAnalytic", "SupplierPerformanceMetric", "SupplyChainNetwork"])
except ImportError:
    pass

# Visualization models
try:
    from .visualization import (
        ChartType, Chart, SavedChart, Dashboard, DashboardChart
    )
    _all_models.extend(["ChartType", "Chart", "SavedChart", "Dashboard", "DashboardChart"])
except ImportError:
    pass

# Extended models
try:
    from .extended_models import (
        ExtendedAnalyticsMetric, ExtendedReport, ForecastModel,
        ExportJob, ReportTemplate, ScheduledReport,
        SystemSetting, NotificationSetting, WidgetType,
        DataSource, APIKey
    )
    _all_models.extend(["ExtendedAnalyticsMetric", "ExtendedReport", "ForecastModel",
                       "ExportJob", "ReportTemplate", "ScheduledReport",
                       "SystemSetting", "NotificationSetting", "WidgetType",
                       "DataSource", "APIKey"])
except ImportError:
    pass

# Aliases
UserProfile = UserPreference

# Export all successfully imported models
__all__ = ["Base"] + _all_models + ["UserProfile"]