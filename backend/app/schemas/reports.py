"""
Schemas for reports endpoints
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ReportFormat(str, Enum):
    """Available report formats"""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    HTML = "html"


class ReportStatus(str, Enum):
    """Report generation status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReportType(str, Enum):
    """Types of reports available"""
    INVENTORY = "inventory"
    SUPPLIER_PERFORMANCE = "supplier_performance"
    ORDER_SUMMARY = "order_summary"
    LOGISTICS = "logistics"
    FINANCIAL = "financial"
    CUSTOM = "custom"


class ReportRequest(BaseModel):
    """Request model for generating a report"""
    report_type: ReportType
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    format: ReportFormat = ReportFormat.PDF
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    filters: Optional[Dict[str, Any]] = None
    include_charts: bool = True
    include_summary: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "report_type": "inventory",
                "name": "Monthly Inventory Report",
                "description": "Inventory levels and movements for the month",
                "parameters": {
                    "warehouse_id": "wh-001",
                    "include_zero_stock": False
                },
                "format": "pdf",
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-01-31T23:59:59",
                "include_charts": True,
                "include_summary": True
            }
        }


class ReportResponse(BaseModel):
    """Response model for a generated report"""
    id: str
    name: str
    report_type: ReportType
    status: ReportStatus
    format: ReportFormat
    file_url: Optional[str] = None
    file_size: Optional[int] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    created_by: str
    error_message: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        from_attributes = True


class ReportTemplateRequest(BaseModel):
    """Request model for creating a report template"""
    name: str
    description: Optional[str] = None
    report_type: ReportType
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    default_format: ReportFormat = ReportFormat.PDF
    is_active: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Weekly Supplier Performance",
                "description": "Standard weekly supplier performance metrics",
                "report_type": "supplier_performance",
                "default_parameters": {
                    "metrics": ["on_time_delivery", "quality_score", "cost_variance"],
                    "threshold": 0.95
                },
                "default_format": "excel",
                "is_active": True
            }
        }


class ReportTemplateResponse(BaseModel):
    """Response model for a report template"""
    id: str
    name: str
    description: Optional[str] = None
    report_type: ReportType
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    default_format: ReportFormat
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    created_by: str
    
    class Config:
        from_attributes = True


class ScheduledReportRequest(BaseModel):
    """Request model for scheduling a report"""
    name: str
    template_id: str
    schedule_type: str  # "daily", "weekly", "monthly", "custom"
    schedule_config: Dict[str, Any] = Field(default_factory=dict)
    recipients: List[str] = Field(default_factory=list)
    is_active: bool = True
    parameters_override: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Daily Inventory Report",
                "template_id": "tpl-001",
                "schedule_type": "daily",
                "schedule_config": {
                    "time": "08:00",
                    "timezone": "UTC"
                },
                "recipients": ["manager@example.com", "warehouse@example.com"],
                "is_active": True,
                "parameters_override": {
                    "include_projections": True
                }
            }
        }


class ScheduledReportResponse(BaseModel):
    """Response model for a scheduled report"""
    id: str
    name: str
    template_id: str
    schedule_type: str
    schedule_config: Dict[str, Any]
    recipients: List[str]
    is_active: bool
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime
    created_by: str
    parameters_override: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class ReportListResponse(BaseModel):
    """Response model for listing reports"""
    reports: List[ReportResponse]
    total: int
    page: int
    page_size: int
    
    class Config:
        from_attributes = True


class ReportTemplateListResponse(BaseModel):
    """Response model for listing report templates"""
    templates: List[ReportTemplateResponse]
    total: int
    page: int
    page_size: int
    
    class Config:
        from_attributes = True


class ScheduledReportListResponse(BaseModel):
    """Response model for listing scheduled reports"""
    scheduled_reports: List[ScheduledReportResponse]
    total: int
    page: int
    page_size: int
    
    class Config:
        from_attributes = True


class ReportGenerationStatus(BaseModel):
    """Status of report generation"""
    report_id: str
    status: ReportStatus
    progress: Optional[int] = Field(None, ge=0, le=100)
    message: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    
    class Config:
        from_attributes = True