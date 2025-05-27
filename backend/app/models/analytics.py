"""
Analytics and reporting database models
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from uuid import uuid4
from decimal import Decimal

from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, ForeignKey,
    Text, Date, DECIMAL, BigInteger, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


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
    
    # Relationships
    created_by_user = relationship('User')
    
    def __repr__(self):
        return f"<AnalyticsResult(id={self.id}, type={self.analytics_type}, status={self.status})>"
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get analytics parameter value"""
        return self.parameters.get(key, default) if self.parameters else default
    
    def get_recommendation_count(self) -> int:
        """Get number of recommendations"""
        return len(self.recommendations) if self.recommendations else 0


class ScheduledAnalytic(Base):
    """Scheduled analytics jobs"""
    __tablename__ = 'scheduled_analytics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    analytics_type = Column(String(100), nullable=False)
    parameters = Column(JSONB, nullable=False, default={})
    schedule_cron = Column(String(100), nullable=False)
    timezone = Column(String(50), default='UTC')
    is_active = Column(Boolean, default=True)
    last_run_at = Column(DateTime(timezone=True))
    next_run_at = Column(DateTime(timezone=True))
    notification_config = Column(JSONB, default={})
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    created_by_user = relationship('User')
    
    def __repr__(self):
        return f"<ScheduledAnalytic(name={self.name}, type={self.analytics_type})>"
    
    @property
    def is_overdue(self) -> bool:
        """Check if scheduled run is overdue"""
        if self.next_run_at and self.is_active:
            return datetime.utcnow() > self.next_run_at
        return False


class AnalyticsTemplate(Base):
    """Pre-configured analytics templates"""
    __tablename__ = 'analytics_templates'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)
    analytics_type = Column(String(100), nullable=False)
    description = Column(Text)
    default_parameters = Column(JSONB, default={})
    parameter_schema = Column(JSONB, default={})
    tags = Column(ARRAY(Text), default=[])
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AnalyticsTemplate(name={self.name}, category={self.category})>"


class Report(Base):
    """Generated reports"""
    __tablename__ = 'reports'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    report_type = Column(String(100), nullable=False)
    format = Column(String(50), nullable=False, default='pdf')
    parameters = Column(JSONB, nullable=False, default={})
    content = Column(JSONB)
    file_url = Column(Text)
    file_size_bytes = Column(BigInteger)
    status = Column(String(50), nullable=False, default='pending')
    error_message = Column(Text)
    generated_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    generated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    generated_by_user = relationship('User')
    
    def __repr__(self):
        return f"<Report(name={self.name}, type={self.report_type}, status={self.status})>"
    
    @property
    def is_ready(self) -> bool:
        """Check if report is ready for download"""
        return self.status == 'completed' and self.file_url is not None


class ReportSchedule(Base):
    """Scheduled report generation"""
    __tablename__ = 'report_schedules'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    report_type = Column(String(100), nullable=False)
    parameters = Column(JSONB, nullable=False, default={})
    schedule_cron = Column(String(100), nullable=False)
    timezone = Column(String(50), default='UTC')
    format = Column(String(50), default='pdf')
    recipients = Column(JSONB, nullable=False, default=list)
    is_active = Column(Boolean, default=True)
    last_generated_at = Column(DateTime(timezone=True))
    next_run_at = Column(DateTime(timezone=True))
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    created_by_user = relationship('User')
    
    def __repr__(self):
        return f"<ReportSchedule(name={self.name}, type={self.report_type})>"


class SupplierPerformanceMetric(Base):
    """Supplier performance metrics over time"""
    __tablename__ = 'supplier_performance_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    supplier_id = Column(UUID(as_uuid=True), ForeignKey('suppliers.id'), nullable=False)
    metric_date = Column(Date, nullable=False)
    on_time_delivery_rate = Column(DECIMAL(5, 2))
    quality_score = Column(DECIMAL(5, 2))
    price_competitiveness = Column(DECIMAL(5, 2))
    response_time_hours = Column(DECIMAL(10, 2))
    defect_rate = Column(DECIMAL(5, 2))
    order_accuracy_rate = Column(DECIMAL(5, 2))
    overall_score = Column(DECIMAL(5, 2))
    metrics_detail = Column(JSONB, default={})
    calculated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    supplier = relationship('Supplier', back_populates='performance_metrics')
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('supplier_id', 'metric_date', name='uq_supplier_metric_date'),
    )
    
    def __repr__(self):
        return f"<SupplierPerformanceMetric(supplier_id={self.supplier_id}, date={self.metric_date})>"
    
    @property
    def is_high_performer(self) -> bool:
        """Check if supplier is a high performer"""
        return self.overall_score and self.overall_score >= 85


class InventoryMetric(Base):
    """Inventory performance metrics"""
    __tablename__ = 'inventory_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    metric_date = Column(Date, nullable=False)
    location_code = Column(String(100))
    total_inventory_value = Column(DECIMAL(15, 2))
    inventory_turnover_ratio = Column(DECIMAL(10, 2))
    days_of_inventory = Column(DECIMAL(10, 2))
    stockout_occurrences = Column(Integer)
    overstock_items = Column(Integer)
    dead_stock_value = Column(DECIMAL(15, 2))
    carrying_cost = Column(DECIMAL(15, 2))
    metrics_detail = Column(JSONB, default={})
    calculated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def __repr__(self):
        return f"<InventoryMetric(date={self.metric_date}, location={self.location_code})>"
    
    @property
    def health_status(self) -> str:
        """Determine inventory health status"""
        if self.stockout_occurrences and self.stockout_occurrences > 5:
            return 'critical'
        elif self.inventory_turnover_ratio and self.inventory_turnover_ratio < 4:
            return 'warning'
        return 'healthy'


class DeliveryPerformance(Base):
    """Delivery and logistics performance tracking"""
    __tablename__ = 'delivery_performance'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    shipment_id = Column(UUID(as_uuid=True), ForeignKey('shipments.id'))
    carrier_name = Column(String(255))
    route_code = Column(String(100))
    scheduled_delivery = Column(DateTime(timezone=True))
    actual_delivery = Column(DateTime(timezone=True))
    delivery_variance_hours = Column(DECIMAL(10, 2))
    on_time = Column(Boolean)
    delay_reason = Column(String(255))
    customer_satisfaction_score = Column(DECIMAL(3, 2))
    cost_variance = Column(DECIMAL(15, 2))
    metadata = Column(JSONB, default={})
    recorded_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    shipment = relationship('Shipment')
    
    def __repr__(self):
        return f"<DeliveryPerformance(id={self.id}, carrier={self.carrier_name}, on_time={self.on_time})>"


class RiskAssessment(Base):
    """Risk assessments for suppliers and supply chain"""
    __tablename__ = 'risk_assessments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    assessment_date = Column(Date, nullable=False)
    supplier_id = Column(UUID(as_uuid=True), ForeignKey('suppliers.id'))
    risk_category = Column(String(100), nullable=False)
    risk_level = Column(String(50), nullable=False)
    risk_score = Column(DECIMAL(5, 2), nullable=False)
    impact_score = Column(DECIMAL(5, 2))
    probability_score = Column(DECIMAL(5, 2))
    risk_factors = Column(JSONB, default=list)
    mitigation_actions = Column(JSONB, default=list)
    assessment_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    supplier = relationship('Supplier')
    assessor = relationship('User')
    
    def __repr__(self):
        return f"<RiskAssessment(id={self.id}, category={self.risk_category}, level={self.risk_level})>"
    
    @property
    def is_high_risk(self) -> bool:
        """Check if this is a high risk assessment"""
        return self.risk_level in ['high', 'critical'] or (self.risk_score and self.risk_score >= 70)


class ComplianceCheck(Base):
    """Compliance tracking for suppliers"""
    __tablename__ = 'compliance_checks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    check_date = Column(Date, nullable=False)
    supplier_id = Column(UUID(as_uuid=True), ForeignKey('suppliers.id'))
    compliance_type = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)
    score = Column(DECIMAL(5, 2))
    findings = Column(JSONB, default=list)
    required_actions = Column(JSONB, default=list)
    due_date = Column(Date)
    completed_date = Column(Date)
    checked_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    supplier = relationship('Supplier')
    checker = relationship('User')
    
    def __repr__(self):
        return f"<ComplianceCheck(id={self.id}, type={self.compliance_type}, status={self.status})>"
    
    @property
    def is_compliant(self) -> bool:
        """Check if supplier is compliant"""
        return self.status in ['compliant', 'passed']
    
    @property
    def is_overdue(self) -> bool:
        """Check if compliance actions are overdue"""
        if self.due_date and not self.completed_date:
            return date.today() > self.due_date
        return False


class ABCAnalysisResult(Base):
    """ABC inventory analysis results"""
    __tablename__ = 'abc_analysis_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_date = Column(Date, nullable=False)
    analysis_type = Column(String(50), nullable=False)
    item_type = Column(String(50), nullable=False)
    item_id = Column(UUID(as_uuid=True), nullable=False)
    category = Column(String(1), nullable=False)
    annual_value = Column(DECIMAL(15, 2))
    annual_quantity = Column(DECIMAL(15, 3))
    percentage_of_total = Column(DECIMAL(5, 2))
    cumulative_percentage = Column(DECIMAL(5, 2))
    recommendations = Column(JSONB, default=list)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("category IN ('A', 'B', 'C')", name='ck_abc_category'),
    )
    
    def __repr__(self):
        return f"<ABCAnalysisResult(item_id={self.item_id}, category={self.category})>"


class ForecastResult(Base):
    """Demand forecasting results"""
    __tablename__ = 'forecast_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    forecast_date = Column(Date, nullable=False)
    item_type = Column(String(50), nullable=False)
    item_id = Column(UUID(as_uuid=True), nullable=False)
    forecast_method = Column(String(100), nullable=False)
    forecast_horizon_days = Column(Integer, nullable=False)
    forecasted_values = Column(JSONB, nullable=False)
    confidence_intervals = Column(JSONB, default={})
    accuracy_metrics = Column(JSONB, default={})
    model_parameters = Column(JSONB, default={})
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    created_by_user = relationship('User')
    
    def __repr__(self):
        return f"<ForecastResult(item_id={self.item_id}, method={self.forecast_method})>"
    
    def get_forecast_for_date(self, target_date: date) -> Optional[Dict[str, Any]]:
        """Get forecast value for a specific date"""
        if not self.forecasted_values:
            return None
        
        date_str = target_date.isoformat()
        return self.forecasted_values.get(date_str)


class SafetyStockCalculation(Base):
    """Safety stock calculation results"""
    __tablename__ = 'safety_stock_calculations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    calculation_date = Column(Date, nullable=False)
    product_id = Column(UUID(as_uuid=True), ForeignKey('products.id'))
    material_id = Column(UUID(as_uuid=True), ForeignKey('materials.id'))
    location_code = Column(String(100))
    service_level = Column(DECIMAL(5, 2), nullable=False)
    lead_time_days = Column(Integer, nullable=False)
    demand_mean = Column(DECIMAL(15, 3))
    demand_std_dev = Column(DECIMAL(15, 3))
    lead_time_std_dev = Column(DECIMAL(10, 3))
    calculated_safety_stock = Column(DECIMAL(15, 3), nullable=False)
    current_stock = Column(DECIMAL(15, 3))
    recommendation = Column(String(255))
    calculation_method = Column(String(100))
    parameters = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    product = relationship('Product')
    material = relationship('Material')
    
    # Constraints
    __table_args__ = (
        CheckConstraint('(product_id IS NOT NULL AND material_id IS NULL) OR '
                       '(product_id IS NULL AND material_id IS NOT NULL)',
                       name='ck_safety_stock_product_or_material'),
    )
    
    def __repr__(self):
        item_type = 'product' if self.product_id else 'material'
        item_id = self.product_id or self.material_id
        return f"<SafetyStockCalculation({item_type}={item_id}, safety_stock={self.calculated_safety_stock})>"
    
    @property
    def needs_adjustment(self) -> bool:
        """Check if safety stock needs adjustment"""
        if self.current_stock is None:
            return True
        return self.current_stock < self.calculated_safety_stock


class SupplyChainNetwork(Base):
    """Supply chain network configurations"""
    __tablename__ = 'supply_chain_networks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    network_type = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    configuration = Column(JSONB, default={})
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    created_by_user = relationship('User')
    nodes = relationship('NetworkNode', back_populates='network', cascade='all, delete-orphan')
    edges = relationship('NetworkEdge', back_populates='network', cascade='all, delete-orphan')
    bottleneck_analyses = relationship('BottleneckAnalysis', back_populates='network')
    risk_scenarios = relationship('RiskPropagationScenario', back_populates='network')
    
    def __repr__(self):
        return f"<SupplyChainNetwork(name={self.name}, type={self.network_type})>"
    
    @property
    def node_count(self) -> int:
        """Get number of nodes in network"""
        return len(self.nodes) if self.nodes else 0
    
    @property
    def edge_count(self) -> int:
        """Get number of edges in network"""
        return len(self.edges) if self.edges else 0


class NetworkNode(Base):
    """Nodes in supply chain network"""
    __tablename__ = 'network_nodes'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    network_id = Column(UUID(as_uuid=True), ForeignKey('supply_chain_networks.id', ondelete='CASCADE'), nullable=False)
    node_type = Column(String(100), nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    entity_type = Column(String(100), nullable=False)
    tier_level = Column(Integer)
    position_x = Column(DECIMAL(10, 2))
    position_y = Column(DECIMAL(10, 2))
    attributes = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    network = relationship('SupplyChainNetwork', back_populates='nodes')
    outgoing_edges = relationship('NetworkEdge', foreign_keys='NetworkEdge.source_node_id', back_populates='source_node')
    incoming_edges = relationship('NetworkEdge', foreign_keys='NetworkEdge.target_node_id', back_populates='target_node')
    
    def __repr__(self):
        return f"<NetworkNode(id={self.id}, type={self.node_type}, tier={self.tier_level})>"


class NetworkEdge(Base):
    """Edges/connections in supply chain network"""
    __tablename__ = 'network_edges'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    network_id = Column(UUID(as_uuid=True), ForeignKey('supply_chain_networks.id', ondelete='CASCADE'), nullable=False)
    source_node_id = Column(UUID(as_uuid=True), ForeignKey('network_nodes.id'), nullable=False)
    target_node_id = Column(UUID(as_uuid=True), ForeignKey('network_nodes.id'), nullable=False)
    edge_type = Column(String(100), nullable=False)
    weight = Column(DECIMAL(10, 3), default=1.0)
    capacity = Column(DECIMAL(15, 3))
    flow_rate = Column(DECIMAL(15, 3))
    attributes = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    network = relationship('SupplyChainNetwork', back_populates='edges')
    source_node = relationship('NetworkNode', foreign_keys=[source_node_id], back_populates='outgoing_edges')
    target_node = relationship('NetworkNode', foreign_keys=[target_node_id], back_populates='incoming_edges')
    
    def __repr__(self):
        return f"<NetworkEdge(source={self.source_node_id}, target={self.target_node_id}, type={self.edge_type})>"
    
    @property
    def utilization_rate(self) -> Optional[float]:
        """Calculate edge utilization rate"""
        if self.capacity and self.flow_rate and self.capacity > 0:
            return (self.flow_rate / self.capacity) * 100
        return None


class BottleneckAnalysis(Base):
    """Bottleneck identification in supply chain networks"""
    __tablename__ = 'bottleneck_analysis'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    network_id = Column(UUID(as_uuid=True), ForeignKey('supply_chain_networks.id'), nullable=False)
    analysis_date = Column(DateTime(timezone=True), nullable=False)
    bottleneck_type = Column(String(100), nullable=False)
    node_id = Column(UUID(as_uuid=True), ForeignKey('network_nodes.id'))
    edge_id = Column(UUID(as_uuid=True), ForeignKey('network_edges.id'))
    severity_score = Column(DECIMAL(5, 2), nullable=False)
    impact_assessment = Column(JSONB, nullable=False, default={})
    affected_paths = Column(JSONB, default=list)
    mitigation_options = Column(JSONB, default=list)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    network = relationship('SupplyChainNetwork', back_populates='bottleneck_analyses')
    node = relationship('NetworkNode')
    edge = relationship('NetworkEdge')
    created_by_user = relationship('User')
    
    def __repr__(self):
        return f"<BottleneckAnalysis(id={self.id}, type={self.bottleneck_type}, severity={self.severity_score})>"
    
    @property
    def is_critical(self) -> bool:
        """Check if bottleneck is critical"""
        return self.severity_score and self.severity_score >= 70


class RiskPropagationScenario(Base):
    """Risk propagation scenarios for network analysis"""
    __tablename__ = 'risk_propagation_scenarios'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    network_id = Column(UUID(as_uuid=True), ForeignKey('supply_chain_networks.id'), nullable=False)
    scenario_type = Column(String(100), nullable=False)
    disruption_source_nodes = Column(ARRAY(UUID(as_uuid=True)))
    disruption_parameters = Column(JSONB, nullable=False, default={})
    propagation_model = Column(String(100), nullable=False)
    simulation_results = Column(JSONB, default={})
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    network = relationship('SupplyChainNetwork', back_populates='risk_scenarios')
    created_by_user = relationship('User')
    impacts = relationship('DisruptionImpact', back_populates='scenario', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<RiskPropagationScenario(name={self.name}, type={self.scenario_type})>"


class DisruptionImpact(Base):
    """Impact analysis for disruption scenarios"""
    __tablename__ = 'disruption_impacts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    scenario_id = Column(UUID(as_uuid=True), ForeignKey('risk_propagation_scenarios.id'), nullable=False)
    affected_node_id = Column(UUID(as_uuid=True), ForeignKey('network_nodes.id'))
    impact_type = Column(String(100), nullable=False)
    impact_level = Column(String(50), nullable=False)
    time_to_impact_hours = Column(DECIMAL(10, 2))
    recovery_time_hours = Column(DECIMAL(10, 2))
    financial_impact = Column(DECIMAL(15, 2))
    operational_impact = Column(JSONB, default={})
    calculated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    scenario = relationship('RiskPropagationScenario', back_populates='impacts')
    affected_node = relationship('NetworkNode')
    
    def __repr__(self):
        return f"<DisruptionImpact(scenario_id={self.scenario_id}, type={self.impact_type}, level={self.impact_level})>"
    
    @property
    def total_disruption_hours(self) -> Optional[float]:
        """Calculate total disruption time"""
        if self.time_to_impact_hours and self.recovery_time_hours:
            return self.time_to_impact_hours + self.recovery_time_hours
        return None