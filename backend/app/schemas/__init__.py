"""
Pydantic schemas for API request/response validation
"""

from .auth import (
    UserBase, UserCreate, UserUpdate, UserInDB, UserResponse,
    UserLogin, TokenData,
    PasswordReset, PasswordReset,
    RoleBase, RoleCreate, RoleResponse,
    PermissionBase, PermissionResponse
)

from .query import (
    NaturalLanguageQueryRequest, NaturalLanguageQueryResponse,
    QueryExecutionResult, QuerySuggestion,
    SavedQueryCreate, SavedQueryUpdate, SavedQueryResponse,
    QueryTemplateResponse, QueryHistoryItem,
    QueryCacheConfig
)

from .visualization import (
    ChartTypeResponse, ChartDataPoint,
    ChartCreate, ChartUpdate, ChartResponse,
    ChartDataUpdate, ChartConfigUpdate,
    SavedChartCreate, SavedChartResponse,
    DashboardCreate, DashboardUpdate, DashboardResponse,
    DashboardChartPosition, DashboardChartAdd,
    DashboardLayoutUpdate
)

from .analytics import (
    AnalyticsRequest, AnalyticsResponse,
    InventoryAnalyticsRequest, InventoryAnalyticsResponse,
    SupplierAnalyticsRequest, SupplierAnalyticsResponse,
    LogisticsAnalyticsRequest, LogisticsAnalyticsResponse,
    ABCAnalysisRequest, ABCAnalysisResponse,
    ForecastRequest, ForecastResponse,
    SafetyStockRequest, SafetyStockResponse,
    NetworkAnalysisRequest, NetworkAnalysisResponse,
    RiskScenarioRequest, ReportResponse)

from .supply_chain import (
    SupplierBase, SupplierCreate, SupplierUpdate, SupplierResponse,
    SupplierTierUpdate, SupplierRelationshipCreate,
    ProductBase, ProductCreate, ProductUpdate, ProductResponse,
    MaterialBase, MaterialCreate, MaterialUpdate, MaterialResponse,
    InventoryBase, InventoryUpdate, InventoryResponse,
    InventoryAdjustment, InventoryReservation,
    OrderBase, OrderCreate, OrderUpdate, OrderResponse,
    OrderItemBase, OrderItemCreate, OrderItemResponse,
    ShipmentBase, ShipmentCreate, ShipmentUpdate, ShipmentResponse,
    SupplierPerformanceUpdate, ComplianceCheckCreate
)

__all__ = [
    ]