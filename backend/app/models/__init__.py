"""
Models package initialization
Import order is important - base models must be imported before models that reference them
"""

# Base model - must be first
from .base import Base

# User models - no foreign keys to other app models
from .user import (
    User,
    UserSession,
    PasswordResetToken,
    UserPreference,
    Role,
    Permission,
    AuditLog
)

# System models - may reference User
from .system import (
    SystemSetting,
    DatabaseConnection,
    QueryOptimization,
    # Add other system models here
)

# Query models - references User
from .query import (
    NaturalLanguageQuery,
    SQLQuery,
    QueryExecution,
    SavedQuery,
    QueryTemplate,
    QuerySuggestion,
    QueryResultCache,
    Query,
    QueryResult,
    QueryHistory,
    QueryVisualization
)

# Analytics models - may reference Query and User
from .analytics import (
    AnalyticsMetric,
    AnalyticsReport,
    # Add other analytics models here
)

# Visualization models - references Query and User
from .visualization import (
    ChartType,
    Chart,
    SavedChart,
    Dashboard,
    DashboardChart,
    ChartData,
    DashboardWidget,
    VisualizationTemplate
)

# Supply chain models - may reference User
from .supply_chain import (
    Product,
    Supplier,
    SupplierProduct,
    InventoryLevel,
    Customer,
    CustomerOrder,
    DeliveryOrder,
    Transfer,
    ProductMovement,
    # Add other supply chain models here
)

# Extended models - may reference multiple other models
try:
    from .extended_models import (
        ExtendedAnalyticsMetric,
        ForecastModel,
        ExtendedReport,
        DataQualityCheck,
        AlertRule,
        # Add other extended models here
    )
except ImportError:
    pass  # Extended models are optional

# Build __all__ dynamically
import sys
current_module = sys.modules[__name__]
__all__ = [
    name for name in dir(current_module)
    if not name.startswith('_') and name != 'Base'
]

# Add Base to __all__
__all__.insert(0, 'Base')
