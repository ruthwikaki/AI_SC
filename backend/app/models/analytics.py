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
    __table_args__ = {"extend_existing": True}
    
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

# Added missing models that other parts of the code expect

class AnalyticsResult(Base):
    """Stored analytics computation results"""
    __tablename__ = 'analytics_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    analytics_type = Column(String(100), nullable=False)
    parameters = Column(JSONB, nullable=False, default={})
    result_data = Column(JSONB, nullable=False)
    summary = Column(JSONB, default={})
    recommendations = Column(JSONB, default=list)
    execution_time_ms = Column(Integer)
    status = Column(String(50), nullable=False, default='completed')
    error_message = Column(Text)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AnalyticsResult(id={self.id}, type={self.analytics_type}, status={self.status})>"

class AnalyticsMetric(Base):
    """Analytics metrics model"""
    __tablename__ = 'analytics_metrics'
    __table_args__ = {"extend_existing": True}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)
    value = Column(Numeric(20, 4), nullable=False)
    unit = Column(String(20))
    period_start = Column(DateTime(timezone=True))
    period_end = Column(DateTime(timezone=True))
    calculated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    meta_data = Column(JSONB, default={})
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AnalyticsMetric(name={self.name}, type={self.metric_type})>"
class ScheduledAnalytic(Base):
    """ScheduledAnalytic model"""
    __tablename__ = 'scheduled_analytics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ScheduledAnalytic(id={self.id})>"

class AnalyticsTemplate(Base):
    """AnalyticsTemplate model"""
    __tablename__ = 'analytics_templates'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<AnalyticsTemplate(id={self.id})>"

class ReportSchedule(Base):
    """ReportSchedule model"""
    __tablename__ = 'report_schedules'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ReportSchedule(id={self.id})>"

class SupplierPerformanceMetric(Base):
    """SupplierPerformanceMetric model"""
    __tablename__ = 'supplier_performance_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SupplierPerformanceMetric(id={self.id})>"

class InventoryMetric(Base):
    """InventoryMetric model"""
    __tablename__ = 'inventory_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<InventoryMetric(id={self.id})>"

class DeliveryPerformance(Base):
    """DeliveryPerformance model"""
    __tablename__ = 'delivery_performance'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<DeliveryPerformance(id={self.id})>"

class RiskAssessment(Base):
    """RiskAssessment model"""
    __tablename__ = 'risk_assessments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<RiskAssessment(id={self.id})>"

class ComplianceCheck(Base):
    """ComplianceCheck model"""
    __tablename__ = 'compliance_checks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ComplianceCheck(id={self.id})>"

class ABCAnalysisResult(Base):
    """ABCAnalysisResult model"""
    __tablename__ = 'abc_analysis_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ABCAnalysisResult(id={self.id})>"

class ForecastResult(Base):
    """ForecastResult model"""
    __tablename__ = 'forecast_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ForecastResult(id={self.id})>"

class SafetyStockCalculation(Base):
    """SafetyStockCalculation model"""
    __tablename__ = 'safety_stock_calculations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SafetyStockCalculation(id={self.id})>"

class SupplyChainNetwork(Base):
    """SupplyChainNetwork model"""
    __tablename__ = 'supply_chain_networks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SupplyChainNetwork(id={self.id})>"

class NetworkNode(Base):
    """NetworkNode model"""
    __tablename__ = 'network_nodes'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<NetworkNode(id={self.id})>"

class NetworkEdge(Base):
    """NetworkEdge model"""
    __tablename__ = 'network_edges'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<NetworkEdge(id={self.id})>"

class BottleneckAnalysis(Base):
    """BottleneckAnalysis model"""
    __tablename__ = 'bottleneck_analysis'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<BottleneckAnalysis(id={self.id})>"

class RiskPropagationScenario(Base):
    """RiskPropagationScenario model"""
    __tablename__ = 'risk_propagation_scenarios'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<RiskPropagationScenario(id={self.id})>"

class DisruptionImpact(Base):
    """DisruptionImpact model"""
    __tablename__ = 'disruption_impacts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<DisruptionImpact(id={self.id})>"

class Report(Base):
    """Report model"""
    __tablename__ = 'reports'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    report_type = Column(String(50))
    parameters = Column(JSONB, default={})
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Report(id={self.id}, name={self.name})>"
