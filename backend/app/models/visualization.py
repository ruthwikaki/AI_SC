"""
Visualization-related database models
from app.models.base import Base
"""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4

from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, ForeignKey,
    Text, ARRAY, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.models.base import Base

class ChartType(Base):
    """Available chart types in the system"""
    __tablename__ = 'chart_types'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(50), unique=True, nullable=False)
    display_name = Column(String(100))
    description = Column(Text)
    component_name = Column(String(100))
    default_config = Column(JSONB, default={})
    supported_data_types = Column(ARRAY(Text), default=[])
    min_data_points = Column(Integer)
    max_data_points = Column(Integer)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    charts = relationship('Chart', back_populates='chart_type')
    
    def __repr__(self):
        return f"<ChartType(name={self.name}, display_name={self.display_name})>"
    
    def supports_data_type(self, data_type: str) -> bool:
        """Check if chart type supports a given data type"""
        return data_type in self.supported_data_types if self.supported_data_types else True
    
    def validate_data_points(self, count: int) -> bool:
        """Validate if data point count is within limits"""
        if self.min_data_points and count < self.min_data_points:
            return False
        if self.max_data_points and count > self.max_data_points:
            return False
        return True


class Chart(Base):
    """Chart configurations and metadata"""
    __tablename__ = 'charts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    chart_type_id = Column(UUID(as_uuid=True), ForeignKey('chart_types.id'), nullable=False)
    query_id = Column(UUID(as_uuid=True), ForeignKey('natural_language_queries.id'))
    data_source = Column(JSONB, nullable=False)
    config = Column(JSONB, default={})
    filters = Column(JSONB, default={})
    is_public = Column(Boolean, default=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chart_type = relationship('ChartType', back_populates='charts')
    query = relationship('NaturalLanguageQuery', back_populates='charts')
    created_by_user = relationship('User', back_populates='created_charts')
    saved_by_users = relationship('SavedChart', back_populates='chart')
    dashboards = relationship('DashboardChart', back_populates='chart')
    
    def __repr__(self):
        return f"<Chart(id={self.id}, title={self.title})>"
    
    @property
    def chart_type_name(self) -> Optional[str]:
        """Get chart type name"""
        return self.chart_type.name if self.chart_type else None
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return self.config.get(key, default) if self.config else default
    
    def update_config(self, updates: Dict[str, Any]):
        """Update chart configuration"""
        if not self.config:
            self.config = {}
        self.config.update(updates)
        self.updated_at = datetime.utcnow()


class SavedChart(Base):
    """User-saved charts"""
    __tablename__ = 'saved_charts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    chart_id = Column(UUID(as_uuid=True), ForeignKey('charts.id'), nullable=False)
    name = Column(String(255))
    is_favorite = Column(Boolean, default=False)
    tags = Column(ARRAY(Text), default=[])
    saved_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user = relationship('User')
    chart = relationship('Chart', back_populates='saved_by_users')
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'chart_id', name='uq_user_chart'),
    )
    
    def __repr__(self):
        return f"<SavedChart(user_id={self.user_id}, chart_id={self.chart_id})>"


class Dashboard(Base):
    """Dashboard configurations"""
    __tablename__ = 'dashboards'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)  # Added from new version
    layout_config = Column(JSONB, nullable=False, default={})
    theme = Column(String(50), default='default')
    refresh_interval = Column(Integer)  # seconds
    is_public = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)
    tags = Column(ARRAY(Text), default=[])
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship('User', foreign_keys=[user_id])  # Added from new version
    created_by_user = relationship('User', back_populates='created_dashboards', foreign_keys=[created_by])
    charts = relationship('DashboardChart', back_populates='dashboard', cascade='all, delete-orphan')
    widgets = relationship('DashboardWidget', back_populates='dashboard', cascade='all, delete-orphan')  # Added from new version
    
    def __repr__(self):
        return f"<Dashboard(id={self.id}, name={self.name})>"
    
    @property
    def chart_count(self) -> int:
        """Get number of charts in dashboard"""
        return len(self.charts) if self.charts else 0
    
    def get_layout_value(self, key: str, default: Any = None) -> Any:
        """Get layout configuration value"""
        return self.layout_config.get(key, default) if self.layout_config else default
    
    def update_layout(self, updates: Dict[str, Any]):
        """Update layout configuration"""
        if not self.layout_config:
            self.layout_config = {}
        self.layout_config.update(updates)
        self.updated_at = datetime.utcnow()


class DashboardChart(Base):
    """Charts within dashboards"""
    __tablename__ = 'dashboard_charts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    dashboard_id = Column(UUID(as_uuid=True), ForeignKey('dashboards.id', ondelete='CASCADE'), nullable=False)
    chart_id = Column(UUID(as_uuid=True), ForeignKey('charts.id'), nullable=False)
    position = Column(JSONB, nullable=False)  # {x, y, w, h}
    config_overrides = Column(JSONB, default={})
    display_order = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    dashboard = relationship('Dashboard', back_populates='charts')
    chart = relationship('Chart', back_populates='dashboards')
    
    def __repr__(self):
        return f"<DashboardChart(dashboard_id={self.dashboard_id}, chart_id={self.chart_id})>"
    
    @property
    def x(self) -> Optional[int]:
        """Get x position"""
        return self.position.get('x') if self.position else None
    
    @property
    def y(self) -> Optional[int]:
        """Get y position"""
        return self.position.get('y') if self.position else None
    
    @property
    def width(self) -> Optional[int]:
        """Get width"""
        return self.position.get('w') if self.position else None
    
    @property
    def height(self) -> Optional[int]:
        """Get height"""
        return self.position.get('h') if self.position else None
    
    def update_position(self, x: int, y: int, w: int, h: int):
        """Update chart position in dashboard"""
        self.position = {'x': x, 'y': y, 'w': w, 'h': h}
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Get chart config with dashboard overrides applied"""
        if not self.chart:
            return {}
        
        config = self.chart.config.copy() if self.chart.config else {}
        if self.config_overrides:
            config.update(self.config_overrides)
        
        return config


class DashboardWidget(Base):
    """Generic widgets within dashboards (Added from new version)"""
    __tablename__ = 'dashboard_widgets'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    dashboard_id = Column(UUID(as_uuid=True), ForeignKey('dashboards.id'), nullable=False)
    widget_type = Column(String(50), nullable=False)
    title = Column(String(200))
    config = Column(JSONB)
    position = Column(JSONB)  # {x, y, w, h}
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    dashboard = relationship('Dashboard', back_populates='widgets')
    
    def __repr__(self):
        return f"<DashboardWidget(dashboard_id={self.dashboard_id}, widget_type={self.widget_type})>"