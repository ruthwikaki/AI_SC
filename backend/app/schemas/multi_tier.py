# backend/app/schemas/multi_tier.py
# Add these schema definitions to support the new routes

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Multi-Tier Schemas
class NetworkNode(BaseModel):
    id: str
    label: str
    type: str
    tier: int
    risk_score: Optional[float] = 0.0
    metadata: Optional[Dict[str, Any]] = {}

class NetworkEdge(BaseModel):
    source: str
    target: str
    weight: Optional[float] = 1.0
    type: Optional[str] = "supply"

class NetworkGraphResponse(BaseModel):
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    bottlenecks: List[Dict[str, Any]]
    metrics: Dict[str, Any]

class RiskAnalysisResponse(BaseModel):
    risk_scores: Dict[str, float]
    propagation_paths: List[List[str]]
    impact_metrics: Dict[str, Any]
    recommendations: List[str]
    analysis_timestamp: datetime

class ScenarioSimulationRequest(BaseModel):
    disrupted_suppliers: List[int]
    severity: float = Field(ge=0.0, le=1.0)
    duration_days: int = Field(ge=1)
    disruption_type: str
    target_recovery_time: Optional[int] = None

class ScenarioSimulationResponse(BaseModel):
    scenario_id: str
    affected_suppliers: List[Dict[str, Any]]
    supply_impact: Dict[str, float]
    financial_impact: Dict[str, float]
    recovery_strategies: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]

class SupplierTierResponse(BaseModel):
    supplier_id: int
    supplier_name: str
    tier_level: int
    tier_score: float
    connections_count: int
    risk_level: str

# Reports Schemas
class ReportCreateRequest(BaseModel):
    name: Optional[str] = None
    template_id: int
    parameters: Dict[str, Any] = {}

class ReportResponse(BaseModel):
    id: int
    name: str
    status: str
    template_name: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = {}

class ReportTemplateResponse(BaseModel):
    id: int
    name: str
    description: str
    category: str
    parameters: Dict[str, Any]
    preview_image: Optional[str] = None
    estimated_time: Optional[int] = None

class ScheduledReportRequest(BaseModel):
    name: str
    template_id: int
    schedule_type: str  # daily, weekly, monthly
    schedule_config: Dict[str, Any]
    parameters: Dict[str, Any] = {}
    recipients: List[str] = []

# Settings Schemas
class UserPreferencesUpdate(BaseModel):
    preferences: Dict[str, Any]

class NotificationSettingsUpdate(BaseModel):
    email_enabled: Optional[bool] = None
    push_enabled: Optional[bool] = None
    sms_enabled: Optional[bool] = None
    notification_types: Optional[Dict[str, bool]] = None
    quiet_hours: Optional[Dict[str, Any]] = None

# Dashboard Schemas
class WidgetPosition(BaseModel):
    x: int
    y: int
    w: int
    h: int

class WidgetCreateRequest(BaseModel):
    type: str
    title: str
    config: Optional[Dict[str, Any]] = {}
    position: Optional[WidgetPosition] = None

class WidgetResponse(BaseModel):
    id: int
    type: str
    title: str
    config: Dict[str, Any]
    position: Dict[str, Any]
    created_at: datetime

class DashboardCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    is_public: bool = False
    layout_config: Optional[Dict[str, Any]] = {}
    widgets: Optional[List[WidgetCreateRequest]] = []

class DashboardUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None
    layout_config: Optional[Dict[str, Any]] = None

class DashboardResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    is_public: bool
    is_default: bool
    layout_config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    widgets_count: int

# Export Schemas
class ExportRequest(BaseModel):
    export_type: str  # inventory, orders, suppliers, custom
    format: str  # csv, xlsx, json, pdf
    parameters: Dict[str, Any] = {}

class ExportJobResponse(BaseModel):
    id: int
    status: str
    export_type: str
    format: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    file_size: Optional[int] = None
    error_message: Optional[str] = None