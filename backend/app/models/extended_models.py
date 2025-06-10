# backend/app/models/extended_models.py
# Add these models to your existing models or create new model files

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Float, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from app.models.base import Base

# User Preferences and Settings
class UserPreference(Base):
    __tablename__ = 'user_preferences'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    key = Column(String(100), nullable=False)
    value = Column(Text)
    value_type = Column(String(20), default='string')  # string, json, number, boolean
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="preferences")

class NotificationSetting(Base):
    __tablename__ = 'notification_settings'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), unique=True, nullable=False)
    email_enabled = Column(Boolean, default=True)
    push_enabled = Column(Boolean, default=False)
    sms_enabled = Column(Boolean, default=False)
    notification_types = Column(JSON)  # JSON object with notification type preferences
    quiet_hours_start = Column(String(5))  # HH:MM format
    quiet_hours_end = Column(String(5))    # HH:MM format
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="notification_settings")

class SystemSetting(Base):
    __tablename__ = 'system_settings'
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text)
    value_type = Column(String(20), default='string')
    description = Column(Text)
    category = Column(String(50))
    is_public = Column(Boolean, default=False)  # Whether non-admins can see this setting
    updated_by = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Reports and Templates
class ReportTemplate(Base):
    __tablename__ = 'report_templates'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    category = Column(String(50))
    template_type = Column(String(50))  # inventory, orders, financial, etc.
    parameters = Column(JSON)  # JSON schema for parameters
    template_content = Column(Text)  # Template definition
    preview_image = Column(String(500))
    estimated_generation_time = Column(Integer)  # in seconds
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Report(Base):
    __tablename__ = 'reports'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    template_id = Column(Integer, ForeignKey('report_templates.id'))
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    parameters = Column(JSON)
    status = Column(String(20), default='pending')  # pending, generating, completed, failed
    file_path = Column(String(500))
    file_size = Column(Integer)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    template = relationship("ReportTemplate")
    user = relationship("User", back_populates="reports")

class ScheduledReport(Base):
    __tablename__ = 'scheduled_reports'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    template_id = Column(Integer, ForeignKey('report_templates.id'))
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    schedule_type = Column(String(20))  # daily, weekly, monthly
    schedule_config = Column(JSON)  # Cron-like configuration
    parameters = Column(JSON)
    recipients = Column(JSON)  # List of email addresses
    is_active = Column(Boolean, default=True)
    last_run = Column(DateTime)
    next_run = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    template = relationship("ReportTemplate")
    user = relationship("User")

# Dashboard and Widgets
class Dashboard(Base):
    __tablename__ = 'dashboards'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    is_public = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)
    layout_config = Column(JSON)  # Grid layout configuration
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="dashboards")
    widgets = relationship("DashboardWidget", back_populates="dashboard", cascade="all, delete-orphan")

class DashboardWidget(Base):
    __tablename__ = 'dashboard_widgets'
    
    id = Column(Integer, primary_key=True, index=True)
    dashboard_id = Column(Integer, ForeignKey('dashboards.id'), nullable=False)
    widget_type = Column(String(50), nullable=False)
    title = Column(String(200))
    config = Column(JSON)  # Widget-specific configuration
    position = Column(JSON)  # Position and size in grid
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    dashboard = relationship("Dashboard", back_populates="widgets")

class WidgetType(Base):
    __tablename__ = 'widget_types'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    display_name = Column(String(100))
    category = Column(String(50))
    description = Column(Text)
    default_config = Column(JSON)
    preview_image = Column(String(500))
    is_active = Column(Boolean, default=True)

# Export Jobs
class ExportJob(Base):
    __tablename__ = 'export_jobs'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    export_type = Column(String(50), nullable=False)
    format = Column(String(20), nullable=False)
    parameters = Column(JSON)
    status = Column(String(20), default='pending')
    file_path = Column(String(500))
    file_size = Column(Integer)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    user = relationship("User")

# Analytics Cache for Performance
class AnalyticsCache(Base):
    __tablename__ = 'analytics_cache'
    
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(200), unique=True, nullable=False)
    cache_type = Column(String(50))
    data = Column(JSON)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
# Multi-Tier Network Cache
class NetworkCache(Base):
    __tablename__ = 'network_cache'
    
    id = Column(Integer, primary_key=True, index=True)
    network_type = Column(String(50))
    parameters = Column(JSON)
    graph_data = Column(JSON)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)