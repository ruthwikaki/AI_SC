"""
Database Models Package
Exports all database models for easy importing
"""

# Import Base first
from .base import Base

# User and auth models
from .user import (
    User, UserSession, PasswordResetToken, UserPreference,
    Role, Permission, AuditLog
)

# Query models - import what exists
from .query import NaturalLanguageQuery, SavedQuery

# Try to import QueryResult if it exists
try:
    from .query import QueryResult
except ImportError:
    pass

# Try to import optional query models
try:
    from .query import QueryResultCache, QueryTemplate, QuerySuggestion
except ImportError:
    # These might not exist in query.py
    pass

# Try to import QueryVisualization
try:
    from .query import QueryVisualization
except ImportError:
    pass

# Visualization models
from .visualization import (
    ChartType, Chart, SavedChart, Dashboard, DashboardChart
)

# Try to import DashboardWidget
try:
    from .visualization import DashboardWidget
except ImportError:
    pass

# Supply chain models
from .supply_chain import (
    Supplier, Product, Material, Inventory,
    Order, OrderItem, Shipment
)

# Try to import additional supply chain models
try:
    from .supply_chain import (
        SupplierTier, SupplierRelationship, ShipmentItem,
        ProductMaterial, InventoryHistory
    )
except ImportError:
    pass

# Try to import Warehouse and SupplierProduct
try:
    from .supply_chain import Warehouse, SupplierProduct
except ImportError:
    pass

# Analytics models
from .analytics import (
    AnalyticsResult, ScheduledAnalytic, AnalyticsTemplate,
    Report, ReportSchedule,
    SupplierPerformanceMetric, InventoryMetric,
    DeliveryPerformance, RiskAssessment, ComplianceCheck,
    ABCAnalysisResult, ForecastResult, SafetyStockCalculation
)

# Try to import network analytics models
try:
    from .analytics import (
        SupplyChainNetwork, NetworkNode, NetworkEdge,
        BottleneckAnalysis, RiskPropagationScenario, DisruptionImpact
    )
except ImportError:
    pass

# Extended models - with corrected names
try:
    from .extended_models import (
        ForecastModel,
        ExtendedAnalyticsMetric,  # Renamed from AnalyticsMetric
        ExportJob,
        ReportTemplate,
        ExtendedReport,  # Renamed from Report to avoid conflict
        ScheduledReport,
        NotificationSetting,
        WidgetType
    )
except ImportError:
    pass

# System models
try:
    from .system import (
        SystemSetting,  # Import from system.py, not extended_models.py
        DatabaseConnection,
        QueryExecutionLog,
        SchemaCache,
        SystemConfig,
        ClientConfig,
        IntegrationLog
    )
except ImportError:
    # If system.py doesn't have all models, at least try to get what exists
    try:
        from .system import SystemSetting
    except ImportError:
        pass

# Build __all__ dynamically based on what's actually imported
__all__ = []

# Get all names in current module
import sys
current_module = sys.modules[__name__]

for name in dir(current_module):
    obj = getattr(current_module, name, None)
    # Only add if it's a model (has __tablename__) or is Base
    if obj is not None and (name == 'Base' or (hasattr(obj, '__tablename__') and not name.startswith('_'))):
        __all__.append(name)

# Ensure core models are in __all__ if they were imported
core_exports = [
    'Base', 'User', 'Role', 'Permission', 'AuditLog',
    'NaturalLanguageQuery', 'SavedQuery',
    'Chart', 'Dashboard', 'Supplier', 'Product', 'Order',
    'AnalyticsResult', 'Report', 'SystemSetting'
]

for export in core_exports:
    if hasattr(current_module, export) and export not in __all__:
        __all__.append(export)

# Sort for consistency
__all__.sort()