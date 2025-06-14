# app/models/extended_models.py

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey, Text, Enum, Numeric
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

# ExtendedAnalyticsMetric - No inheritance, standalone model
class ExtendedAnalyticsMetric(Base):
    __tablename__ = "extended_analytics_metrics"
    __table_args__ = {"extend_existing": True}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # forecast, accuracy, kpi, etc.
    entity_type = Column(String(50))  # product, warehouse, supplier
    entity_id = Column(UUID(as_uuid=True))
    warehouse_id = Column(UUID(as_uuid=True), ForeignKey("warehouses.id"), nullable=True)
    metric_date = Column(DateTime(timezone=True), nullable=False)
    value = Column(Float, nullable=False)
    meta_data = Column(JSONB)
    # Extended fields
    forecast_accuracy = Column(Numeric(5, 2))  # Percentage
    confidence_level = Column(Numeric(5, 2))  # Percentage
    anomaly_detected = Column(Boolean, default=False)
    anomaly_score = Column(Numeric(5, 2))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    def __repr__(self):
        return f"<ExtendedAnalyticsMetric(name={self.name}, type={self.metric_type})>"

# Alias for compatibility
AnalyticsMetric = ExtendedAnalyticsMetric

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

# Report model - matches what reports.py expects
class Report(Base):
    __tablename__ = "reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    template_id = Column(UUID(as_uuid=True), ForeignKey("report_templates.id"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    parameters = Column(JSONB)
    status = Column(String(50), nullable=False, default='generating')
    file_path = Column(String(500))
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    template = relationship("ReportTemplate", foreign_keys=[template_id])
    user = relationship("User", foreign_keys=[user_id])
    
    def __repr__(self):
        return f"<Report(name={self.name}, status={self.status})>"

# ExtendedReport as alias for backwards compatibility
ExtendedReport = Report

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
    template = relationship("ReportTemplate")
    user = relationship("User", foreign_keys=[user_id])


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
    user = relationship("User", back_populates="notification_settings")

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

class DataSource(Base):
    """Data source configuration model"""
    __tablename__ = 'data_sources'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    source_type = Column(String(50))  # 'database', 'api', 'file', 'stream'
    connection_string = Column(Text)  # Encrypted
    config = Column(JSONB, default={})  # Additional configuration
    is_active = Column(Boolean, default=True)
    test_query = Column(Text)  # Query to test connection
    last_tested = Column(DateTime(timezone=True))
    last_sync = Column(DateTime(timezone=True))
    sync_frequency = Column(Integer)  # Minutes
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

class APIKey(Base):
    """API key model for external access"""
    __tablename__ = 'api_keys'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    key_hash = Column(String(255), unique=True, nullable=False)
    name = Column(String(100))
    description = Column(Text)
    permissions = Column(JSONB, default=[])  # List of allowed endpoints/operations
    rate_limit = Column(Integer, default=1000)  # Requests per hour
    expires_at = Column(DateTime(timezone=True))
    last_used = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user = relationship('User')

class AuditLog(Base):
    """Audit log for tracking system changes"""
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    action = Column(String(50), nullable=False)  # 'create', 'update', 'delete', 'export'
    resource_type = Column(String(50), nullable=False)  # Model name
    resource_id = Column(UUID(as_uuid=True))
    changes = Column(JSONB)  # Before/after values
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user = relationship('User')