"""
Analytics schemas for various supply chain analytics operations
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# =====================================================
# Enums
# =====================================================

class AnalyticsType(str, Enum):
    """Types of analytics"""
    INVENTORY_OPTIMIZATION = "inventory_optimization"
    ABC_ANALYSIS = "abc_analysis"
    SAFETY_STOCK = "safety_stock"
    DEMAND_FORECAST = "demand_forecast"
    SUPPLIER_PERFORMANCE = "supplier_performance"
    LOGISTICS_OPTIMIZATION = "logistics_optimization"
    RISK_ASSESSMENT = "risk_assessment"
    NETWORK_ANALYSIS = "network_analysis"
    CUSTOM = "custom"


class TimeFrame(str, Enum):
    """Time frame for analytics"""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"


class ForecastMethod(str, Enum):
    """Forecasting methods"""
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ARIMA = "arima"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    MACHINE_LEARNING = "machine_learning"
    AUTO = "auto"


class RiskLevel(str, Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =====================================================
# Base Analytics Schemas
# =====================================================

class AnalyticsRequest(BaseModel):
    """Base analytics request"""
    analytics_type: AnalyticsType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    time_frame: Optional[TimeFrame] = TimeFrame.MONTH
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('end_date')
    @classmethod
    def validate_dates(cls, v, info):
        """Validate date range"""
        if v and 'start_date' in info.data and info.data['start_date']:
            if v < info.data['start_date']:
                raise ValueError('End date must be after start date')
        return v


class AnalyticsResponse(BaseModel):
    """Base analytics response"""
    id: UUID
    analytics_type: str
    parameters: Dict[str, Any]
    result_data: Dict[str, Any]
    summary: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = []
    execution_time_ms: int
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True  # Changed from orm_mode


# =====================================================
# Inventory Analytics Schemas
# =====================================================

class InventoryAnalyticsRequest(BaseModel):
    """Inventory analytics request"""
    location_codes: Optional[List[str]] = None
    category: Optional[str] = None
    include_zero_stock: bool = True
    time_frame: TimeFrame = TimeFrame.MONTH
    metrics: List[str] = Field(default_factory=lambda: [
        "turnover_ratio", "days_of_inventory", "stockout_rate", "overstock_items"
    ])


class InventoryMetrics(BaseModel):
    """Inventory metrics"""
    total_value: float
    total_items: int
    turnover_ratio: float
    days_of_inventory: float
    stockout_rate: float
    overstock_items: int
    dead_stock_value: float
    carrying_cost: float


class InventoryAnalyticsResponse(BaseModel):
    """Inventory analytics response"""
    metrics: InventoryMetrics
    low_stock_items: List[Dict[str, Any]]
    overstock_items: List[Dict[str, Any]]
    trends: Dict[str, List[Dict[str, Any]]]
    recommendations: List[str]
    generated_at: datetime


# =====================================================
# ABC Analysis Schemas
# =====================================================

class ABCAnalysisRequest(BaseModel):
    """ABC analysis request"""
    analysis_type: str = Field(default="value", pattern="^(value|quantity|frequency)$")  # Changed from regex
    location_code: Optional[str] = None
    categories: Optional[List[str]] = None
    thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "A": 0.8,  # 80% cumulative
        "B": 0.95,  # 95% cumulative
        "C": 1.0   # 100% cumulative
    })
    include_recommendations: bool = True


class ABCAnalysisItem(BaseModel):
    """ABC analysis item result"""
    item_id: UUID
    item_name: str
    item_code: str
    category: str  # A, B, or C
    annual_value: float
    annual_quantity: float
    percentage_of_total: float
    cumulative_percentage: float


class ABCAnalysisResponse(BaseModel):
    """ABC analysis response"""
    analysis_date: date
    total_items: int
    total_value: float
    category_distribution: Dict[str, Dict[str, Any]]
    items: List[ABCAnalysisItem]
    recommendations: Dict[str, List[str]]
    visualization_data: Dict[str, Any]


# =====================================================
# Safety Stock Schemas
# =====================================================

class SafetyStockRequest(BaseModel):
    """Safety stock calculation request"""
    item_ids: Optional[List[UUID]] = None
    location_codes: Optional[List[str]] = None
    service_level: float = Field(default=95.0, ge=50.0, le=99.9)
    lead_time_days: Optional[int] = Field(None, ge=1)
    include_seasonality: bool = True
    calculation_method: str = Field(default="basic", pattern="^(basic|advanced|dynamic)$")  # Changed from regex


class SafetyStockItem(BaseModel):
    """Safety stock calculation for an item"""
    item_id: UUID
    item_name: str
    location_code: str
    current_stock: float
    calculated_safety_stock: float
    reorder_point: float
    service_level: float
    lead_time_days: int
    demand_variability: float
    recommendation: str


class SafetyStockResponse(BaseModel):
    """Safety stock response"""
    calculation_date: date
    items: List[SafetyStockItem]
    total_additional_stock_needed: float
    total_additional_value: float
    summary_by_location: Dict[str, Dict[str, Any]]
    recommendations: List[str]


# =====================================================
# Forecast Schemas
# =====================================================

class ForecastRequest(BaseModel):
    """Demand forecast request"""
    item_ids: Optional[List[UUID]] = None
    categories: Optional[List[str]] = None
    forecast_horizon_days: int = Field(default=90, ge=7, le=365)
    forecast_method: ForecastMethod = ForecastMethod.AUTO
    include_seasonality: bool = True
    include_trend: bool = True
    confidence_level: float = Field(default=95.0, ge=50.0, le=99.9)


class ForecastItem(BaseModel):
    """Forecast for a single item"""
    item_id: UUID
    item_name: str
    forecast_values: List[Dict[str, Any]]  # date, value, lower_bound, upper_bound
    accuracy_metrics: Dict[str, float]
    trend: str  # "increasing", "decreasing", "stable"
    seasonality_detected: bool


class ForecastResponse(BaseModel):
    """Forecast response"""
    forecast_start_date: date
    forecast_end_date: date
    method_used: str
    items: List[ForecastItem]
    aggregate_forecast: Dict[str, Any]
    accuracy_summary: Dict[str, float]
    recommendations: List[str]


# =====================================================
# Supplier Analytics Schemas
# =====================================================

class SupplierAnalyticsRequest(BaseModel):
    """Supplier analytics request"""
    supplier_ids: Optional[List[UUID]] = None
    categories: Optional[List[str]] = None
    metrics: List[str] = Field(default_factory=lambda: [
        "on_time_delivery", "quality_score", "price_competitiveness", "risk_score"
    ])
    time_frame: TimeFrame = TimeFrame.QUARTER
    include_benchmarking: bool = True


class SupplierMetrics(BaseModel):
    """Supplier performance metrics"""
    supplier_id: UUID
    supplier_name: str
    on_time_delivery_rate: float
    quality_score: float
    price_competitiveness: float
    response_time_hours: float
    order_accuracy_rate: float
    overall_score: float
    trend: str


class SupplierAnalyticsResponse(BaseModel):
    """Supplier analytics response"""
    period_start: date
    period_end: date
    suppliers_analyzed: int
    top_performers: List[SupplierMetrics]
    bottom_performers: List[SupplierMetrics]
    category_benchmarks: Dict[str, Dict[str, float]]
    risk_alerts: List[Dict[str, Any]]
    recommendations: List[str]


# =====================================================
# Logistics Analytics Schemas
# =====================================================

class LogisticsAnalyticsRequest(BaseModel):
    """Logistics analytics request"""
    carrier_names: Optional[List[str]] = None
    route_codes: Optional[List[str]] = None
    analysis_types: List[str] = Field(default_factory=lambda: [
        "delivery_performance", "cost_analysis", "route_optimization"
    ])
    time_frame: TimeFrame = TimeFrame.MONTH
    include_predictions: bool = True


class RouteMetrics(BaseModel):
    """Route performance metrics"""
    route_code: str
    total_shipments: int
    on_time_rate: float
    average_transit_time_hours: float
    cost_per_shipment: float
    utilization_rate: float


class LogisticsAnalyticsResponse(BaseModel):
    """Logistics analytics response"""
    delivery_performance: Dict[str, Any]
    cost_analysis: Dict[str, Any]
    route_metrics: List[RouteMetrics]
    optimization_opportunities: List[Dict[str, Any]]
    predicted_delays: List[Dict[str, Any]]
    recommendations: List[str]


# =====================================================
# Network Analysis Schemas
# =====================================================

class NetworkAnalysisRequest(BaseModel):
    """Supply chain network analysis request"""
    network_id: Optional[UUID] = None
    analysis_types: List[str] = Field(default_factory=lambda: [
        "bottlenecks", "critical_paths", "risk_propagation"
    ])
    disruption_scenarios: Optional[List[Dict[str, Any]]] = None
    include_alternatives: bool = True


class NetworkNode(BaseModel):
    """Network node information"""
    node_id: UUID
    entity_type: str
    entity_name: str
    tier_level: int
    criticality_score: float
    risk_score: float
    connections_count: int


class BottleneckInfo(BaseModel):
    """Bottleneck information"""
    bottleneck_id: UUID
    type: str
    location: str
    severity_score: float
    affected_flow_percentage: float
    estimated_impact: Dict[str, Any]
    mitigation_options: List[Dict[str, Any]]


class NetworkAnalysisResponse(BaseModel):
    """Network analysis response"""
    network_summary: Dict[str, Any]
    critical_nodes: List[NetworkNode]
    bottlenecks: List[BottleneckInfo]
    risk_propagation_paths: List[Dict[str, Any]]
    alternative_routes: List[Dict[str, Any]]
    resilience_score: float
    recommendations: List[str]


# =====================================================
# Risk Analysis Schemas
# =====================================================

class RiskScenarioRequest(BaseModel):
    """Risk scenario analysis request"""
    scenario_name: str
    scenario_type: str
    disruption_sources: List[Dict[str, Any]]
    disruption_duration_days: int = Field(ge=1, le=365)
    impact_categories: List[str] = Field(default_factory=lambda: [
        "financial", "operational", "reputational"
    ])
    include_mitigation: bool = True


class RiskScenarioResponse(BaseModel):
    """Risk scenario response"""
    scenario_id: UUID
    total_impact_score: float
    financial_impact: float
    operational_impact: Dict[str, Any]
    affected_entities: List[Dict[str, Any]]
    recovery_time_estimate_days: int
    mitigation_strategies: List[Dict[str, Any]]
    simulation_confidence: float


# =====================================================
# Report Generation Schemas
# =====================================================

class ReportGenerationRequest(BaseModel):
    """Report generation request"""
    report_type: str
    report_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    format: str = Field(default="pdf", pattern="^(pdf|excel|html|pptx)$")  # Changed from regex
    include_sections: List[str] = Field(default_factory=lambda: [
        "executive_summary", "detailed_analysis", "recommendations"
    ])
    recipients: Optional[List[str]] = None


class ReportResponse(BaseModel):
    """Report response"""
    id: UUID
    name: str
    report_type: str
    format: str
    status: str
    file_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    generated_at: datetime
    expires_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True  # Changed from orm_mode


# =====================================================
# Scheduled Analytics Schemas
# =====================================================

class ScheduledAnalyticsCreate(BaseModel):
    """Create scheduled analytics"""
    name: str = Field(..., min_length=1, max_length=255)
    analytics_type: AnalyticsType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    schedule_cron: str
    timezone: str = "UTC"
    notification_emails: List[str] = []
    notification_webhook: Optional[str] = None
    is_active: bool = True


class ScheduledAnalyticsResponse(BaseModel):
    """Scheduled analytics response"""
    id: UUID
    name: str
    analytics_type: str
    parameters: Dict[str, Any]
    schedule_cron: str
    timezone: str
    is_active: bool
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True  # Changed from orm_mode