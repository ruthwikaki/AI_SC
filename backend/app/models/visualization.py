"""

Visualization and charting models

"""



from datetime import datetime

from uuid import uuid4

from typing import Optional, List



from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    JSONB,
    String,
    Text,
    UniqueConstraint
)

from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

from sqlalchemy.orm import relationship



from app.models.base import Base









class ChartType(Base):

    """Chart type definitions"""

    __tablename__ = 'chart_types'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(50), unique=True, nullable=False)

    display_name = Column(String(100))

    description = Column(Text)

    config_schema = Column(JSONB)

    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    

    def __repr__(self):

        return f"<ChartType(name={self.name})>"





class SavedChart(Base):

    """User saved charts"""

    __tablename__ = 'saved_charts'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)

    chart_id = Column(UUID(as_uuid=True), ForeignKey('charts.id'), nullable=False)

    name = Column(String(255))

    is_favorite = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    

    # Relationships

    user = relationship('User', back_populates='saved_charts')

    chart = relationship('Chart')

    

    # Unique constraint

    __table_args__ = (

        UniqueConstraint('user_id', 'chart_id', name='uq_user_saved_chart'),

    )

    

    def __repr__(self):

        return f"<SavedChart(user_id={self.user_id}, chart_id={self.chart_id})>"





class DashboardChart(Base):

    """Charts in dashboards"""

    __tablename__ = 'dashboard_charts'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    dashboard_id = Column(UUID(as_uuid=True), ForeignKey('dashboards.id'), nullable=False)

    chart_id = Column(UUID(as_uuid=True), ForeignKey('charts.id'), nullable=False)

    position_x = Column(Integer, default=0)

    position_y = Column(Integer, default=0)

    width = Column(Integer, default=6)

    height = Column(Integer, default=4)

    config_overrides = Column(JSONB)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    

    # Relationships

    dashboard = relationship('Dashboard', back_populates='charts')

    chart = relationship('Chart')

    

    def __repr__(self):

        return f"<DashboardChart(dashboard_id={self.dashboard_id}, chart_id={self.chart_id})>"





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