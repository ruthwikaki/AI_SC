"""
Analytics and metrics models
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


class AnalyticsMetric(Base):
    """Analytics metrics tracking model"""
    __tablename__ = 'analytics_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # 'inventory', 'sales', 'supplier', 'logistics'
    value = Column(Numeric(20, 4), nullable=False)
    unit = Column(String(20))  # 'percentage', 'currency', 'count', 'days'
    period_start = Column(DateTime(timezone=True))
    period_end = Column(DateTime(timezone=True))
    calculated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    dimension1_name = Column(String(50))  # e.g., 'product', 'supplier', 'warehouse'
    dimension1_value = Column(String(100))
    dimension2_name = Column(String(50))
    dimension2_value = Column(String(100))
    meta_data = Column(JSONB, default={})
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    creator = relationship('User', foreign_keys=[created_by], backref='analytics_metrics')
    
    # Indexes for better query performance
    __table_args__ = (
        Index('idx_analytics_metric_type_period', 'metric_type', 'period_start', 'period_end'),
        Index('idx_analytics_calculated_at', 'calculated_at'),
        Index('idx_analytics_dimensions', 'dimension1_name', 'dimension1_value'),
    )
    
    def __repr__(self):
        return f"<AnalyticsMetric(name={self.name}, type={self.metric_type}, value={self.value})>"


class AnalyticsSummary(Base):
    """Pre-aggregated analytics summaries"""
    __tablename__ = 'analytics_summaries'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    summary_type = Column(String(50), nullable=False)  # 'daily', 'weekly', 'monthly'
    summary_date = Column(DateTime(timezone=True), nullable=False)
    metrics = Column(JSONB, nullable=False)  # Stores various KPIs
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_summary_type_date', 'summary_type', 'summary_date'),
    )
    
    def __repr__(self):
        return f"<AnalyticsSummary(type={self.summary_type}, date={self.summary_date})>"


class KPIDefinition(Base):
    """KPI definitions and calculation rules"""
    __tablename__ = 'kpi_definitions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    code = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    category = Column(String(50))  # 'inventory', 'sales', 'supplier', 'logistics'
    calculation_method = Column(Text)  # SQL or formula
    unit = Column(String(20))
    target_value = Column(Numeric(20, 4))
    target_direction = Column(String(20))  # 'higher_better', 'lower_better', 'target'
    is_active = Column(Boolean, default=True)
    refresh_frequency = Column(String(20))  # 'realtime', 'hourly', 'daily'
    meta_data = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<KPIDefinition(code={self.code}, name={self.name})>"