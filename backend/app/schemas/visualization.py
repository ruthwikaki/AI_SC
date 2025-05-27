"""
Visualization schemas for charts and dashboards
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, validator
from enum import Enum


# =====================================================
# Enums
# =====================================================

class ChartTypeEnum(str, Enum):
    """Available chart types"""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    HEATMAP = "heatmap"
    SANKEY = "sankey"
    NETWORK = "network"
    SCATTER = "scatter"
    AREA = "area"
    DONUT = "donut"
    RADAR = "radar"
    TREEMAP = "treemap"
    GAUGE = "gauge"


class DashboardTheme(str, Enum):
    """Dashboard themes"""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    COMPACT = "compact"
    SPACIOUS = "spacious"


# =====================================================
# Chart Type Schemas
# =====================================================

class ChartTypeResponse(BaseModel):
    """Chart type information"""
    id: UUID
    name: str
    display_name: str
    description: Optional[str] = None
    component_name: str
    default_config: Dict[str, Any] = Field(default_factory=dict)
    supported_data_types: List[str] = []
    min_data_points: Optional[int] = None
    max_data_points: Optional[int] = None
    is_active: bool
    preview_image: Optional[str] = None
    
    class Config:
        orm_mode = True


# =====================================================
# Chart Data Schemas
# =====================================================

class ChartDataPoint(BaseModel):
    """Single data point for charts"""
    label: Optional[str] = None
    value: Union[float, int, str]
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChartDataSeries(BaseModel):
    """Data series for multi-series charts"""
    name: str
    data: List[ChartDataPoint]
    color: Optional[str] = None
    type: Optional[str] = None  # For mixed charts


class ChartData(BaseModel):
    """Complete chart data structure"""
    series: List[ChartDataSeries] = []
    categories: Optional[List[str]] = None
    totals: Optional[Dict[str, float]] = None


# =====================================================
# Chart CRUD Schemas
# =====================================================

class ChartBase(BaseModel):
    """Base chart schema"""
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    is_public: bool = False


class ChartCreate(ChartBase):
    """Create chart request"""
    chart_type_id: UUID
    data_source: Dict[str, Any]
    config: Dict[str, Any] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)
    query_id: Optional[UUID] = None
    
    @validator('data_source')
    def validate_data_source(cls, v):
        """Validate data source has required fields"""
        if 'type' not in v:
            raise ValueError('Data source must specify type')
        return v


class ChartUpdate(BaseModel):
    """Update chart request"""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None
    is_public: Optional[bool] = None


class ChartDataUpdate(BaseModel):
    """Update chart data"""
    data_source: Dict[str, Any]
    refresh_now: bool = True


class ChartConfigUpdate(BaseModel):
    """Update chart configuration"""
    config: Dict[str, Any]
    merge: bool = True  # If true, merge with existing config


class ChartResponse(ChartBase):
    """Chart response schema"""
    id: UUID
    chart_type: ChartTypeResponse
    data_source: Dict[str, Any]
    config: Dict[str, Any] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)
    query_id: Optional[UUID] = None
    created_by: UUID
    created_at: datetime
    updated_at: datetime
    saved_count: int = 0
    last_accessed_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True


# =====================================================
# Saved Chart Schemas
# =====================================================

class SavedChartCreate(BaseModel):
    """Save chart to user collection"""
    chart_id: UUID
    name: Optional[str] = None
    is_favorite: bool = False
    tags: List[str] = Field(default_factory=list)


class SavedChartResponse(BaseModel):
    """Saved chart response"""
    id: UUID
    user_id: UUID
    chart: ChartResponse
    name: Optional[str] = None
    is_favorite: bool
    tags: List[str] = []
    saved_at: datetime
    
    class Config:
        orm_mode = True


# =====================================================
# Dashboard Schemas
# =====================================================

class DashboardBase(BaseModel):
    """Base dashboard schema"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    theme: DashboardTheme = DashboardTheme.DEFAULT
    is_public: bool = False
    tags: List[str] = Field(default_factory=list)


class DashboardCreate(DashboardBase):
    """Create dashboard request"""
    layout_config: Dict[str, Any] = Field(default_factory=lambda: {
        "grid": {"cols": 12, "rowHeight": 100},
        "breakpoints": {"lg": 1200, "md": 996, "sm": 768, "xs": 480},
        "layouts": {}
    })
    refresh_interval: Optional[int] = Field(None, ge=0)  # seconds, 0 means no auto-refresh


class DashboardUpdate(BaseModel):
    """Update dashboard request"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    layout_config: Optional[Dict[str, Any]] = None
    theme: Optional[DashboardTheme] = None
    refresh_interval: Optional[int] = Field(None, ge=0)
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None


class DashboardResponse(DashboardBase):
    """Dashboard response schema"""
    id: UUID
    layout_config: Dict[str, Any]
    refresh_interval: Optional[int] = None
    is_default: bool
    created_by: UUID
    created_at: datetime
    updated_at: datetime
    charts: List['DashboardChartResponse'] = []
    
    class Config:
        orm_mode = True
        use_enum_values = True


# =====================================================
# Dashboard Chart Schemas
# =====================================================

class DashboardChartPosition(BaseModel):
    """Chart position in dashboard grid"""
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    w: int = Field(..., ge=1, le=12)
    h: int = Field(..., ge=1)
    
    @validator('w')
    def validate_width(cls, v):
        """Validate width doesn't exceed grid columns"""
        if v > 12:
            raise ValueError('Width cannot exceed 12 grid columns')
        return v


class DashboardChartAdd(BaseModel):
    """Add chart to dashboard request"""
    chart_id: UUID
    position: DashboardChartPosition
    config_overrides: Dict[str, Any] = Field(default_factory=dict)


class DashboardChartUpdate(BaseModel):
    """Update chart in dashboard"""
    position: Optional[DashboardChartPosition] = None
    config_overrides: Optional[Dict[str, Any]] = None


class DashboardChartResponse(BaseModel):
    """Dashboard chart response"""
    id: UUID
    dashboard_id: UUID
    chart: ChartResponse
    position: Dict[str, int]
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    display_order: int
    
    class Config:
        orm_mode = True


class DashboardLayoutUpdate(BaseModel):
    """Update multiple chart positions"""
    chart_positions: List[Dict[str, Any]]
    
    @validator('chart_positions')
    def validate_positions(cls, v):
        """Validate each position has required fields"""
        for pos in v:
            if 'chart_id' not in pos or 'position' not in pos:
                raise ValueError('Each position must have chart_id and position')
        return v


# =====================================================
# Chart Export Schemas
# =====================================================

class ChartExportRequest(BaseModel):
    """Chart export request"""
    format: str = Field(..., regex='^(png|svg|pdf|csv|xlsx)$')
    width: Optional[int] = Field(None, ge=100, le=4000)
    height: Optional[int] = Field(None, ge=100, le=4000)
    include_data: bool = True
    include_config: bool = False


class DashboardExportRequest(BaseModel):
    """Dashboard export request"""
    format: str = Field(..., regex='^(pdf|png|html)$')
    include_data: bool = True
    paper_size: str = Field(default='a4', regex='^(a4|letter|legal)$')
    orientation: str = Field(default='portrait', regex='^(portrait|landscape)$')


# =====================================================
# Visualization Analytics Schemas
# =====================================================

class ChartAnalytics(BaseModel):
    """Chart usage analytics"""
    chart_id: UUID
    view_count: int
    save_count: int
    export_count: int
    average_view_duration_seconds: float
    last_viewed_at: Optional[datetime] = None
    popular_filters: List[Dict[str, Any]] = []


class DashboardAnalytics(BaseModel):
    """Dashboard usage analytics"""
    dashboard_id: UUID
    view_count: int
    unique_viewers: int
    average_view_duration_seconds: float
    chart_interaction_count: int
    last_viewed_at: Optional[datetime] = None
    popular_time_ranges: List[str] = []


# =====================================================
# Visualization Recommendation Schemas
# =====================================================

class VisualizationRecommendationRequest(BaseModel):
    """Request for visualization recommendations"""
    data_sample: List[Dict[str, Any]]
    data_types: Dict[str, str]  # column_name: data_type
    row_count: int
    intent: Optional[str] = None
    preferred_types: Optional[List[ChartTypeEnum]] = None


class VisualizationRecommendation(BaseModel):
    """Visualization recommendation"""
    chart_type: ChartTypeEnum
    confidence_score: float = Field(ge=0.0, le=1.0)
    reason: str
    config_suggestions: Dict[str, Any] = Field(default_factory=dict)
    data_mapping: Dict[str, str] = Field(default_factory=dict)


# Forward reference update
DashboardResponse.update_forward_refs()