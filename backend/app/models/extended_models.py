# app/models/extended_models.py

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey, Text, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from uuid import uuid4

from app.models.base import Base

class ForecastModel(Base):
    __tablename__ = "forecast_models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), unique=True, nullable=False)
    display_name = Column(String(200))
    description = Column(Text)
    model_type = Column(String(50))  # arima, lstm, prophet, etc.
    average_accuracy = Column(Float, default=0.0)
    best_use_cases = Column(Text)  # comma-separated list
    default_parameters = Column(JSONB)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

# Renamed from AnalyticsMetric to ExtendedAnalyticsMetric to avoid conflict
class ExtendedAnalyticsMetric(Base):
    __tablename__ = "extended_analytics_metrics"  # Changed table name
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    metric_type = Column(String(50), nullable=False)  # forecast, accuracy, kpi, etc.
    entity_type = Column(String(50))  # product, warehouse, supplier
    entity_id = Column(UUID(as_uuid=True))
    warehouse_id = Column(UUID(as_uuid=True), ForeignKey("warehouses.id"), nullable=True)
    metric_date = Column(DateTime(timezone=True), nullable=False)
    value = Column(Float, nullable=False)
    meta_data = Column(JSONB)  # Changed from metadata to meta_data
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Relationships - commented out until models exist
    # warehouse = relationship("Warehouse", back_populates="analytics_metrics")
    # creator = relationship("User", back_populates="analytics_metrics")

class ExportJob(Base):
    __tablename__ = "export_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    export_type = Column(String(50), nullable=False)
    format = Column(String(20), nullable=False)  # csv, xlsx, pdf, json
    parameters = Column(JSONB)
    status = Column(Enum('pending', 'processing', 'completed', 'failed', name='export_status'), default='pending')
    file_path = Column(String(500))
    file_size = Column(Integer)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    # user = relationship("User", back_populates="export_jobs")

class ReportTemplate(Base):
    __tablename__ = "report_templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    category = Column(String(50))
    template_type = Column(String(50))
    parameters = Column(JSONB)  # Required parameters for the template
    layout_config = Column(JSONB)  # Layout configuration
    preview_image = Column(String(500))
    estimated_generation_time = Column(Integer)  # in seconds
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    # reports = relationship("ExtendedReport", back_populates="template")

# Renamed from Report to ExtendedReport to avoid conflict with analytics.py
class ExtendedReport(Base):
    __tablename__ = "extended_reports"  # Changed table name
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(200), nullable=False)
    template_id = Column(UUID(as_uuid=True), ForeignKey("report_templates.id"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    parameters = Column(JSONB)
    status = Column(Enum('generating', 'completed', 'failed', name='extended_report_status'), default='generating')
    file_path = Column(String(500))
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    # template = relationship("ReportTemplate", back_populates="reports")
    # user = relationship("User", back_populates="reports")

class ScheduledReport(Base):
    __tablename__ = "scheduled_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(200), nullable=False)
    template_id = Column(UUID(as_uuid=True), ForeignKey("report_templates.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    schedule_type = Column(String(50))  # daily, weekly, monthly
    schedule_config = Column(JSONB)  # Cron expression or specific config
    parameters = Column(JSONB)
    recipients = Column(JSONB)  # List of email addresses
    is_active = Column(Boolean, default=True)
    last_run = Column(DateTime(timezone=True))
    next_run = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    # template = relationship("ReportTemplate")
    # user = relationship("User", back_populates="scheduled_reports")

class SystemSetting(Base):
    __tablename__ = "system_settings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text)
    value_type = Column(String(20), default='string')
    description = Column(Text)
    category = Column(String(50))
    is_public = Column(Boolean, default=False)  # Whether non-admins can read
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))

class NotificationSetting(Base):
    __tablename__ = "notification_settings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False)
    email_enabled = Column(Boolean, default=True)
    push_enabled = Column(Boolean, default=False)
    sms_enabled = Column(Boolean, default=False)
    notification_types = Column(JSONB)  # Dict of notification type -> enabled
    quiet_hours_start = Column(String(5))  # HH:MM format
    quiet_hours_end = Column(String(5))  # HH:MM format
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    # user = relationship("User", back_populates="notification_settings")

class WidgetType(Base):
    __tablename__ = "widget_types"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), unique=True, nullable=False)
    display_name = Column(String(200))
    category = Column(String(50))
    description = Column(Text)
    default_config = Column(JSONB)
    preview_image = Column(String(500))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class SyncHistory(Base):
    """Track data synchronization history"""
    __tablename__ = "sync_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sync_type = Column(String(50), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String(20))  # pending, running, completed, failed
    records_processed = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    error_message = Column(Text)
    sync_metadata = Column(JSON)