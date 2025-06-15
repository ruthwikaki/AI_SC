"""
Database models for AI Supply Chain Management
"""

# Import Base first
from .base import Base

# Import association tables (not as classes, just to ensure they're created)
from .user import user_roles, role_permissions

# User and auth models
from .user import (
    User, Role, Permission, AuditLog,
    UserSession, PasswordResetToken, UserPreference
)

# Query models
try:
    from .query import (
    NaturalLanguageQuery, SavedQuery,
    QueryResult, QueryVisualization
)
except ImportError as e:
    print(f"Warning: Some query models could not be imported: {e}")
    from .query import NaturalLanguageQuery, SavedQuery
try:
    from .query import Query, QueryResultCache, QueryTemplate, QuerySuggestion
except ImportError:
    pass

# Supply chain models
from .supply_chain import (
    Supplier, Product, Material, Inventory, 
    Order, OrderItem, Shipment, ShipmentItem,
    ProductMaterial, InventoryHistory
)
# Import optional supply chain models
try:
    from .supply_chain import (
        SupplierTier, SupplierRelationship, SupplierProduct,
        Warehouse
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
try:
    from .analytics import (
        SupplyChainNetwork, NetworkNode, NetworkEdge,
        BottleneckAnalysis, RiskPropagationScenario, DisruptionImpact
    )
except ImportError:
    pass

# Extended models
try:
    from .extended_models import (
        ForecastModel, ExtendedAnalyticsMetric, ExportJob,
        ReportTemplate, ExtendedReport, ScheduledReport,
        SystemSetting, NotificationSetting, WidgetType
    )
except ImportError:
    pass

# Visualization models
from .visualization import (
    ChartType, Chart, SavedChart, Dashboard,
    DashboardChart, DashboardWidget
)

# System models (if they exist)
try:
    from .system import (
        DatabaseConnection, QueryExecutionLog, SchemaCache,
        SystemConfig, ClientConfig, IntegrationLog
    )
except ImportError:
    pass

# Create placeholders for models that don't exist yet
if 'QueryResultCache' not in globals():
    class QueryResultCache(Base):
        """Placeholder for Query Result Cache"""
        __tablename__ = 'query_result_cache'
        
        from sqlalchemy import Column, UUID, DateTime, Text, Integer
        from uuid import uuid4
        from datetime import datetime
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
        query_hash = Column(Text, nullable=False)
        result = Column(Text)
        ttl_seconds = Column(Integer, default=3600)
        created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

if 'QueryTemplate' not in globals():
    class QueryTemplate(Base):
        """Placeholder for Query Template"""
        __tablename__ = 'query_templates'
        
        from sqlalchemy import Column, UUID, String, Text, Boolean, DateTime
        from uuid import uuid4
        from datetime import datetime
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
        name = Column(String(255), nullable=False)
        template = Column(Text, nullable=False)
        category = Column(String(100))
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

if 'QuerySuggestion' not in globals():
    class QuerySuggestion(Base):
        """Placeholder for Query Suggestion"""
        __tablename__ = 'query_suggestions'
        
        from sqlalchemy import Column, UUID, String, Text, Integer, DateTime
        from uuid import uuid4
        from datetime import datetime
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
        suggestion = Column(Text, nullable=False)
        category = Column(String(100))
        usage_count = Column(Integer, default=0)
        created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

# Build __all__ dynamically
__all__ = [
    # Base
    'Base',
    
    # User and auth
    'User', 'Role', 'Permission', 'AuditLog',
    'UserSession', 'PasswordResetToken', 'UserPreference',
    
    # Query
    'NaturalLanguageQuery', 'SavedQuery', 'QueryResult', 'QueryVisualization',
    'QueryResultCache', 'QueryTemplate', 'QuerySuggestion',
    
    # Supply chain
    'Supplier', 'Product', 'Material', 'Inventory',
    'Order', 'OrderItem', 'Shipment', 'ShipmentItem',
    'ProductMaterial', 'InventoryHistory',
    'SupplierTier', 'SupplierRelationship', 'SupplierProduct', 'Warehouse',
    
    # Analytics
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
    
    # Visualization
    'ChartType', 'Chart', 'SavedChart', 'Dashboard',
    'DashboardChart', 'DashboardWidget',
]

# Filter out any that don't actually exist
__all__ = [name for name in __all__ if name in globals()]
