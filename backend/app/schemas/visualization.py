"""
Visualization schemas for charts and dashboards
"""

from typing import Optional, List, Dict, Any, Union, Annotated
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, StringConstraints
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


class ExportStatusEnum(str, Enum):
    """Export job status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportTypeEnum(str, Enum):
    """Export types"""
    CHART = "chart"
    DASHBOARD = "dashboard"
    DATA = "data"
    REPORT = "report"


class WidgetTypeEnum(str, Enum):
    """Widget types"""
    CHART = "chart"
    TEXT = "text"
    METRIC = "metric"
    TABLE = "table"
    FILTER = "filter"
    IMAGE = "image"
    CUSTOM = "custom"


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
        from_attributes = True


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
    
    @field_validator('data_source')
    @classmethod
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
        from_attributes = True


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
        from_attributes = True


# =====================================================
# Widget Schemas (NEW)
# =====================================================

class WidgetBase(BaseModel):
    """Base widget schema"""
    type: Union[WidgetTypeEnum, str]
    title: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    position: Dict[str, Any] = Field(default_factory=dict)


class WidgetCreateRequest(WidgetBase):
    """Create widget request"""
    chart_id: Optional[UUID] = None  # For chart widgets
    data_source: Optional[Dict[str, Any]] = None  # For data-driven widgets


class WidgetUpdateRequest(BaseModel):
    """Update widget request"""
    type: Optional[Union[WidgetTypeEnum, str]] = None
    title: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    position: Optional[Dict[str, Any]] = None


class WidgetResponse(WidgetBase):
    """Widget response schema"""
    id: UUID
    dashboard_id: UUID
    chart_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# =====================================================
# Dashboard Schemas (UPDATED)
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
    widgets: Optional[List[WidgetCreateRequest]] = None  # NEW: Create widgets with dashboard


# Alternative simplified create request (NEW)
class DashboardCreateRequest(BaseModel):
    """Simplified dashboard create request"""
    name: str
    description: Optional[str] = None
    is_public: bool = False
    layout_config: Optional[Dict[str, Any]] = None
    widgets: Optional[List[WidgetCreateRequest]] = None


class DashboardUpdate(BaseModel):
    """Update dashboard request"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    layout_config: Optional[Dict[str, Any]] = None
    theme: Optional[DashboardTheme] = None
    refresh_interval: Optional[int] = Field(None, ge=0)
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None


# Alternative simplified update request (NEW)
class DashboardUpdateRequest(BaseModel):
    """Simplified dashboard update request"""
    name: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None
    layout_config: Optional[Dict[str, Any]] = None


class DashboardResponse(DashboardBase):
    """Dashboard response schema"""
    id: UUID
    layout_config: Dict[str, Any]
    refresh_interval: Optional[int] = None
    is_default: bool
    created_by: UUID
    created_at: datetime
    updated_at: datetime
    charts: List['DashboardChartResponse'] = []  # Legacy: for backward compatibility
    widgets: List[WidgetResponse] = []  # NEW: Modern widget system
    widgets_count: int = 0  # NEW: Widget count
    
    class Config:
        from_attributes = True
        use_enum_values = True


# =====================================================
# Dashboard Chart Schemas (Legacy - kept for backward compatibility)
# =====================================================

class DashboardChartPosition(BaseModel):
    """Chart position in dashboard grid"""
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    w: int = Field(..., ge=1, le=12)
    h: int = Field(..., ge=1)
    
    @field_validator('w')
    @classmethod
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
        from_attributes = True


class DashboardLayoutUpdate(BaseModel):
    """Update multiple chart positions"""
    chart_positions: List[Dict[str, Any]]
    
    @field_validator('chart_positions')
    @classmethod
    def validate_positions(cls, v):
        """Validate each position has required fields"""
        for pos in v:
            if 'chart_id' not in pos or 'position' not in pos:
                raise ValueError('Each position must have chart_id and position')
        return v


# =====================================================
# Export Schemas (ENHANCED)
# =====================================================

class ChartExportRequest(BaseModel):
    """Chart export request"""
    format: str = Field(..., pattern='^(png|svg|pdf|csv|xlsx)$')
    width: Optional[int] = Field(None, ge=100, le=4000)
    height: Optional[int] = Field(None, ge=100, le=4000)
    include_data: bool = True
    include_config: bool = False


class DashboardExportRequest(BaseModel):
    """Dashboard export request"""
    format: str = Field(..., pattern='^(pdf|png|html)$')
    paper_size: str = Field(default='a4', pattern='^(a4|letter|legal)$')
    orientation: str = Field(default='portrait', pattern='^(portrait|landscape)$')


# Generic export request (NEW)
class ExportRequest(BaseModel):
    """Generic export request"""
    export_type: Union[ExportTypeEnum, str]
    format: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('format')
    @classmethod
    def validate_format(cls, v, info):
        """Validate format based on export type"""
        export_type = info.data.get('export_type')
        valid_formats = {
            ExportTypeEnum.CHART: ['png', 'svg', 'pdf', 'csv', 'xlsx'],
            ExportTypeEnum.DASHBOARD: ['pdf', 'png', 'html'],
            ExportTypeEnum.DATA: ['csv', 'xlsx', 'json'],
            ExportTypeEnum.REPORT: ['pdf', 'docx', 'html']
        }
        
        if isinstance(export_type, str):
            export_type = ExportTypeEnum(export_type)
            
        if export_type in valid_formats:
            if v not in valid_formats[export_type]:
                raise ValueError(f"Invalid format '{v}' for export type '{export_type}'")
        
        return v


# Export job response (NEW)
class ExportJobResponse(BaseModel):
    """Export job status and result"""
    id: UUID
    status: Union[ExportStatusEnum, str]
    export_type: Union[ExportTypeEnum, str]
    format: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    file_size: Optional[int] = None
    download_url: Optional[str] = None  # Added for completeness
    expires_at: Optional[datetime] = None  # Added for completeness
    error_message: Optional[str] = None
    progress: Optional[float] = Field(None, ge=0.0, le=100.0)  # Added progress tracking
    
    class Config:
        from_attributes = True
        use_enum_values = True


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
    widget_interaction_count: int  # NEW: Track widget interactions
    last_viewed_at: Optional[datetime] = None
    popular_time_ranges: List[str] = []
    widget_performance: Dict[str, Any] = Field(default_factory=dict)  # NEW: Widget-specific metrics


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


# =====================================================
# Batch Operations (NEW)
# =====================================================

class BatchWidgetCreate(BaseModel):
    """Batch create widgets"""
    dashboard_id: UUID
    widgets: List[WidgetCreateRequest]


class BatchWidgetUpdate(BaseModel):
    """Batch update widgets"""
    updates: List[Dict[str, Any]]  # Each dict must have 'id' and update fields
    
    @field_validator('updates')
    @classmethod
    def validate_updates(cls, v):
        """Validate each update has widget id"""
        for update in v:
            if 'id' not in update:
                raise ValueError('Each update must have widget id')
        return v


class BatchExportRequest(BaseModel):
    """Batch export request"""
    export_requests: List[ExportRequest]
    notify_on_completion: bool = True
    combine_results: bool = False


# Forward reference updates
DashboardResponse.model_rebuild()
WidgetResponse.model_rebuild()