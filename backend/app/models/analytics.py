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





class ABCAnalysisResult(Base):

    """ABCAnalysisResult model"""

    __tablename__ = 'abc_analysis_results'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<ABCAnalysisResult(id={self.id})>"





class AnalyticsMetric(Base):



    """TODO: Implement AnalyticsMetric"""
    __tablename__ = "analyticsmetrics"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    pass

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







class AnalyticsTemplate(Base):

    """AnalyticsTemplate model"""

    __tablename__ = 'analytics_templates'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<AnalyticsTemplate(id={self.id})>"





class BottleneckAnalysis(Base):

    """BottleneckAnalysis model"""

    __tablename__ = 'bottleneck_analysis'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<BottleneckAnalysis(id={self.id})>"





class ComplianceCheck(Base):

    """ComplianceCheck model"""

    __tablename__ = 'compliance_checks'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<ComplianceCheck(id={self.id})>"





class DeliveryPerformance(Base):

    """DeliveryPerformance model"""

    __tablename__ = 'delivery_performance'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<DeliveryPerformance(id={self.id})>"





class DisruptionImpact(Base):

    """DisruptionImpact model"""

    __tablename__ = 'disruption_impacts'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<DisruptionImpact(id={self.id})>"





class ForecastResult(Base):

    """ForecastResult model"""

    __tablename__ = 'forecast_results'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<ForecastResult(id={self.id})>"





class InventoryMetric(Base):

    """InventoryMetric model"""

    __tablename__ = 'inventory_metrics'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<InventoryMetric(id={self.id})>"





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





class NetworkEdge(Base):

    """NetworkEdge model"""

    __tablename__ = 'network_edges'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<NetworkEdge(id={self.id})>"





class NetworkNode(Base):

    """NetworkNode model"""

    __tablename__ = 'network_nodes'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<NetworkNode(id={self.id})>"





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



class ReportSchedule(Base):

    """ReportSchedule model"""

    __tablename__ = 'report_schedules'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<ReportSchedule(id={self.id})>"





class RiskAssessment(Base):

    """RiskAssessment model"""

    __tablename__ = 'risk_assessments'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<RiskAssessment(id={self.id})>"





class RiskPropagationScenario(Base):

    """RiskPropagationScenario model"""

    __tablename__ = 'risk_propagation_scenarios'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<RiskPropagationScenario(id={self.id})>"





class SafetyStockCalculation(Base):

    """SafetyStockCalculation model"""

    __tablename__ = 'safety_stock_calculations'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<SafetyStockCalculation(id={self.id})>"





class ScheduledAnalytic(Base):

    """ScheduledAnalytic model"""

    __tablename__ = 'scheduled_analytics'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<ScheduledAnalytic(id={self.id})>"





class SupplierPerformanceMetric(Base):

    """SupplierPerformanceMetric model"""

    __tablename__ = 'supplier_performance_metrics'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<SupplierPerformanceMetric(id={self.id})>"





class SupplyChainNetwork(Base):

    """SupplyChainNetwork model"""

    __tablename__ = 'supply_chain_networks'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<SupplyChainNetwork(id={self.id})>"





