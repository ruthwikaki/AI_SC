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

# Query models
from .query import (
    NaturalLanguageQuery, SavedQuery, QueryResult
)

# Try to import optional query models
try:
    from .query import QueryResultCache, QueryTemplate, QuerySuggestion, QueryVisualization
except ImportError:
    # Create placeholders if they don't exist
    class QueryResultCache(Base):
        __tablename__ = 'query_result_cache'
        from sqlalchemy import Column, Text, Integer, DateTime
        from sqlalchemy.dialects.postgresql import UUID
        from uuid import uuid4
        from datetime import datetime as dt
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
        query_hash = Column(Text, nullable=False)
        result = Column(Text)
        ttl_seconds = Column(Integer, default=3600)
        created_at = Column(DateTime(timezone=True), default=dt.utcnow)
    
    class QueryTemplate(Base):
        __tablename__ = 'query_templates'
        from sqlalchemy import Column, String, Text, Boolean, DateTime
        from sqlalchemy.dialects.postgresql import UUID
        from uuid import uuid4
        from datetime import datetime as dt
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
        name = Column(String(255), nullable=False)
        template = Column(Text, nullable=False)
        category = Column(String(100))
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime(timezone=True), default=dt.utcnow)
    
    class QuerySuggestion(Base):
        __tablename__ = 'query_suggestions'
        from sqlalchemy import Column, String, Text, Integer, DateTime
        from sqlalchemy.dialects.postgresql import UUID
        from uuid import uuid4
        from datetime import datetime as dt
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
        suggestion = Column(Text, nullable=False)
        category = Column(String(100))
        usage_count = Column(Integer, default=0)
        created_at = Column(DateTime(timezone=True), default=dt.utcnow)

# Visualization models
from .visualization import (
    ChartType, Chart, SavedChart, Dashboard, DashboardChart
)

try:
    from .visualization import DashboardWidget
except ImportError:
    pass

# Supply chain models
from .supply_chain import (
    Supplier, Product, Material, Inventory,
    Order, OrderItem, Shipment,
    ProductMaterial, InventoryHistory
)

# Try to import additional supply chain models
try:
    from .supply_chain import (
        SupplierTier, SupplierRelationship, ShipmentItem,
        SupplierProduct, Warehouse
    )
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
        ExtendedAnalyticsMetric,  # Note: renamed from AnalyticsMetric
        ExportJob,
        ReportTemplate,
        ExtendedReport,  # Note: renamed from Report
        ScheduledReport,
        NotificationSetting,
        WidgetType
    )
except ImportError as e:
    print(f"Warning: Could not import extended models: {e}")

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
    # If system.py doesn't have all models, at least get SystemSetting
    from .system import SystemSetting

# Build __all__ dynamically based on what's actually imported
__all__ = []
for name, obj in list(globals().items()):
    if not name.startswith('_') and hasattr(obj, '__tablename__'):
        __all__.append(name)

# Ensure we have the most commonly used exports
core_exports = [
    'Base', 'User', 'Role', 'Permission', 'AuditLog',
    'NaturalLanguageQuery', 'SavedQuery',
    'Chart', 'Dashboard', 'Supplier', 'Product', 'Order',
    'AnalyticsResult', 'Report'
]

for export in core_exports:
    if export in globals() and export not in __all__:
        __all__.append(export)
