"""
Extended models for additional functionality
"""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Numeric, Boolean, DateTime, 
    ForeignKey, Text, JSON, Float, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.models.base import Base
from app.models.analytics import AnalyticsMetric


class ExtendedAnalyticsMetric(AnalyticsMetric):
    """Extended analytics metric model that inherits from AnalyticsMetric"""
    __tablename__ = 'extended_analytics_metrics'
    
    # Additional fields beyond base AnalyticsMetric
    forecast_accuracy = Column(Numeric(5, 2))  # Percentage
    confidence_level = Column(Numeric(5, 2))  # Percentage
    anomaly_detected = Column(Boolean, default=False)
    anomaly_score = Column(Numeric(5, 2))
    
    def __repr__(self):
        return f"<ExtendedAnalyticsMetric(name={self.name}, type={self.metric_type})>"


class Report(Base):
    """Report model for generating and storing reports"""
    __tablename__ = 'reports'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    report_type = Column(String(50), nullable=False)  # 'inventory', 'sales', 'performance', etc.
    format = Column(String(20), default='pdf')  # 'pdf', 'excel', 'csv'
    config = Column(JSONB, default={})  # Report configuration and parameters
    template_id = Column(UUID(as_uuid=True))  # Optional template reference
    data = Column(JSONB)  # Cached report data
    file_path = Column(Text)  # Path to generated file
    status = Column(String(20), default='pending')  # 'pending', 'generating', 'completed', 'failed'
    error_message = Column(Text)
    generated_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship('User', foreign_keys=[user_id])
    
    def __repr__(self):
        return f"<Report(name={self.name}, type={self.report_type})>"


class ScheduledReport(Base):
    """Scheduled report model for recurring reports"""
    __tablename__ = 'scheduled_reports'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    report_id = Column(UUID(as_uuid=True), ForeignKey('reports.id'))
    schedule = Column(String(50))  # Cron expression
    frequency = Column(String(20))  # 'daily', 'weekly', 'monthly'
    next_run = Column(DateTime(timezone=True))
    last_run = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    recipients = Column(JSONB, default=[])  # Email addresses
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship('User', foreign_keys=[user_id])
    report = relationship('Report')
    
    def __repr__(self):
        return f"<ScheduledReport(report_id={self.report_id}, frequency={self.frequency})>"


class ExportJob(Base):
    """Export job model for tracking data exports"""
    __tablename__ = 'export_jobs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    status = Column(String(20), default='pending')  # 'pending', 'processing', 'completed', 'failed'
    export_type = Column(String(50))  # 'full_backup', 'filtered_data', 'report'
    format = Column(String(20))  # 'csv', 'excel', 'json'
    filters = Column(JSONB, default={})
    progress = Column(Integer, default=0)  # Percentage
    file_path = Column(Text)
    file_size = Column(Integer)  # Bytes
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user = relationship('User', foreign_keys=[user_id])
    
    def __repr__(self):
        return f"<ExportJob(type={self.export_type}, status={self.status})>"


class NotificationSetting(Base):
    """Notification settings model for user preferences"""
    __tablename__ = 'notification_settings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), unique=True)
    email_enabled = Column(Boolean, default=True)
    email_frequency = Column(String(20), default='immediate')  # 'immediate', 'daily', 'weekly'
    push_enabled = Column(Boolean, default=False)
    sms_enabled = Column(Boolean, default=False)
    notification_types = Column(JSONB, default={})  # {alert_type: enabled}
    quiet_hours_start = Column(String(5))  # HH:MM format
    quiet_hours_end = Column(String(5))  # HH:MM format
    timezone = Column(String(50), default='UTC')
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='notification_settings')
    
    def __repr__(self):
        return f"<NotificationSetting(user_id={self.user_id})>"


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
    
    def __repr__(self):
        return f"<DataSource(name={self.name}, type={self.source_type})>"


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
    
    def __repr__(self):
        return f"<APIKey(name={self.name}, user_id={self.user_id})>"


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
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_log_user', 'user_id', 'created_at'),
        Index('idx_audit_log_resource', 'resource_type', 'resource_id'),
    )
    
    def __repr__(self):
        return f"<AuditLog(action={self.action}, resource_type={self.resource_type})>"


# For backward compatibility - alias ExtendedAnalyticsMetric as AnalyticsMetric
# AnalyticsMetric = ExtendedAnalyticsMetric