"""
Visualization and charting models
"""

from datetime import datetime
from uuid import uuid4
from typing import Optional, List

from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, ForeignKey,
    Text, JSON, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from app.models.base import Base


class Chart(Base):
    """Chart configuration and data model"""
    __tablename__ = 'charts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    chart_type = Column(String(50), nullable=False)  # 'bar', 'line', 'pie', 'scatter', 'heatmap', etc.
    config = Column(JSONB, nullable=False)  # Chart.js or D3.js configuration
    data_source = Column(String(50))  # 'query', 'api', 'static'
    query_id = Column(UUID(as_uuid=True), ForeignKey('natural_language_queries.id'))
    refresh_interval = Column(Integer)  # in seconds, null for no auto-refresh
    is_public = Column(Boolean, default=False)
    tags = Column(ARRAY(String))
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    created_by_user = relationship('User', foreign_keys=[created_by])
    query = relationship('NaturalLanguageQuery', backref='charts')
    chart_data = relationship('ChartData', back_populates='chart', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Chart(name={self.name}, type={self.chart_type})>"


class ChartData(Base):
    """Stored chart data for caching and historical tracking"""
    __tablename__ = 'chart_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    chart_id = Column(UUID(as_uuid=True), ForeignKey('charts.id', ondelete='CASCADE'), nullable=False)
    data = Column(JSONB, nullable=False)  # The actual chart data
    data_timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    expires_at = Column(DateTime(timezone=True))
    checksum = Column(String(64))  # For detecting data changes
    row_count = Column(Integer)
    meta_data = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    chart = relationship('Chart', back_populates='chart_data')
    
    # Indexes
    __table_args__ = (
        Index('idx_chart_data_timestamp', 'chart_id', 'data_timestamp'),
    )
    
    def __repr__(self):
        return f"<ChartData(chart_id={self.chart_id}, timestamp={self.data_timestamp})>"


class Dashboard(Base):
    """Dashboard containing multiple charts"""
    __tablename__ = 'dashboards'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    layout = Column(JSONB, nullable=False)  # Grid layout configuration
    is_default = Column(Boolean, default=False)
    is_public = Column(Boolean, default=False)
    refresh_interval = Column(Integer)  # in seconds
    theme = Column(String(50), default='light')
    tags = Column(ARRAY(String))
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    created_by_user = relationship('User', foreign_keys=[created_by])
    widgets = relationship('DashboardWidget', back_populates='dashboard', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Dashboard(name={self.name})>"


class DashboardWidget(Base):
    """Widgets within a dashboard"""
    __tablename__ = 'dashboard_widgets'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    dashboard_id = Column(UUID(as_uuid=True), ForeignKey('dashboards.id', ondelete='CASCADE'), nullable=False)
    widget_type = Column(String(50), nullable=False)  # 'chart', 'metric', 'table', 'text'
    chart_id = Column(UUID(as_uuid=True), ForeignKey('charts.id'))
    position = Column(JSONB, nullable=False)  # {x, y, w, h} for grid layout
    config = Column(JSONB, default={})  # Widget-specific configuration
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    dashboard = relationship('Dashboard', back_populates='widgets')
    chart = relationship('Chart')
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('dashboard_id', 'position', name='uq_dashboard_widget_position'),
    )
    
    def __repr__(self):
        return f"<DashboardWidget(dashboard_id={self.dashboard_id}, type={self.widget_type})>"


class VisualizationTemplate(Base):
    """Reusable visualization templates"""
    __tablename__ = 'visualization_templates'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    category = Column(String(50))  # 'inventory', 'sales', 'logistics', etc.
    chart_type = Column(String(50), nullable=False)
    default_config = Column(JSONB, nullable=False)
    example_data = Column(JSONB)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<VisualizationTemplate(name={self.name}, type={self.chart_type})>"