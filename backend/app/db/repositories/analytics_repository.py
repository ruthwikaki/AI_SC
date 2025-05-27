"""
Analytics repository for analytics operations and results management
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from uuid import UUID
import logging
import json

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, or_, and_, desc, asc
from sqlalchemy.exc import IntegrityError

from app.models import (
    AnalyticsResult, ScheduledAnalytic, AnalyticsTemplate,
    Report, ReportSchedule, SupplierPerformanceMetric,
    InventoryMetric, DeliveryPerformance, RiskAssessment,
    ComplianceCheck, ABCAnalysisResult, ForecastResult,
    SafetyStockCalculation, SupplyChainNetwork, NetworkNode,
    NetworkEdge, BottleneckAnalysis, RiskPropagationScenario,
    DisruptionImpact
)

logger = logging.getLogger(__name__)


class AnalyticsRepository:
    """Repository for analytics-related database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # =====================================================
    # Analytics Result Operations
    # =====================================================
    
    def save_analytics_result(
        self,
        analytics_type: str,
        parameters: Dict[str, Any],
        result_data: Dict[str, Any],
        created_by: UUID,
        summary: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None,
        execution_time_ms: Optional[int] = None,
        status: str = 'completed',
        error_message: Optional[str] = None
    ) -> AnalyticsResult:
        """Save analytics computation result"""
        result = AnalyticsResult(
            analytics_type=analytics_type,
            parameters=parameters,
            result_data=result_data,
            summary=summary or {},
            recommendations=recommendations or [],
            execution_time_ms=execution_time_ms,
            status=status,
            error_message=error_message,
            created_by=created_by
        )
        
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        return result
    
    def get_analytics_result(self, result_id: UUID) -> Optional[AnalyticsResult]:
        """Get analytics result by ID"""
        return self.db.query(AnalyticsResult).filter(
            AnalyticsResult.id == result_id
        ).first()
    
    def get_analytics_results(
        self,
        analytics_type: Optional[str] = None,
        created_by: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 50
    ) -> Tuple[List[AnalyticsResult], int]:
        """Get analytics results with filters"""
        query = self.db.query(AnalyticsResult)
        
        if analytics_type:
            query = query.filter(AnalyticsResult.analytics_type == analytics_type)
        
        if created_by:
            query = query.filter(AnalyticsResult.created_by == created_by)
        
        if start_date:
            query = query.filter(AnalyticsResult.created_at >= start_date)
        
        if end_date:
            query = query.filter(AnalyticsResult.created_at <= end_date)
        
        if status:
            query = query.filter(AnalyticsResult.status == status)
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        results = query.order_by(
            desc(AnalyticsResult.created_at)
        ).offset(skip).limit(limit).all()
        
        return results, total
    
    # =====================================================
    # Scheduled Analytics Operations
    # =====================================================
    
    def create_scheduled_analytic(
        self,
        name: str,
        analytics_type: str,
        parameters: Dict[str, Any],
        schedule_cron: str,
        created_by: UUID,
        timezone: str = 'UTC',
        notification_config: Optional[Dict[str, Any]] = None
    ) -> ScheduledAnalytic:
        """Create scheduled analytics job"""
        scheduled = ScheduledAnalytic(
            name=name,
            analytics_type=analytics_type,
            parameters=parameters,
            schedule_cron=schedule_cron,
            timezone=timezone,
            notification_config=notification_config or {},
            created_by=created_by,
            is_active=True
        )
        
        # Calculate next run time (simplified - in production use croniter)
        scheduled.next_run_at = datetime.utcnow() + timedelta(hours=1)
        
        self.db.add(scheduled)
        self.db.commit()
        self.db.refresh(scheduled)
        return scheduled
    
    def get_scheduled_analytics(
        self,
        is_active: Optional[bool] = None,
        created_by: Optional[UUID] = None
    ) -> List[ScheduledAnalytic]:
        """Get scheduled analytics"""
        query = self.db.query(ScheduledAnalytic)
        
        if is_active is not None:
            query = query.filter(ScheduledAnalytic.is_active == is_active)
        
        if created_by:
            query = query.filter(ScheduledAnalytic.created_by == created_by)
        
        return query.order_by(ScheduledAnalytic.next_run_at).all()
    
    def update_scheduled_analytic_status(
        self,
        scheduled_id: UUID,
        last_run_at: datetime,
        next_run_at: datetime
    ) -> Optional[ScheduledAnalytic]:
        """Update scheduled analytic after execution"""
        scheduled = self.db.query(ScheduledAnalytic).filter(
            ScheduledAnalytic.id == scheduled_id
        ).first()
        
        if not scheduled:
            return None
        
        scheduled.last_run_at = last_run_at
        scheduled.next_run_at = next_run_at
        self.db.commit()
        self.db.refresh(scheduled)
        return scheduled
    
    def toggle_scheduled_analytic(
        self,
        scheduled_id: UUID,
        is_active: bool
    ) -> bool:
        """Enable/disable scheduled analytic"""
        scheduled = self.db.query(ScheduledAnalytic).filter(
            ScheduledAnalytic.id == scheduled_id
        ).first()
        
        if not scheduled:
            return False
        
        scheduled.is_active = is_active
        scheduled.updated_at = datetime.utcnow()
        self.db.commit()
        return True
    
    # =====================================================
    # Analytics Template Operations
    # =====================================================
    
    def get_analytics_templates(
        self,
        category: Optional[str] = None,
        analytics_type: Optional[str] = None,
        is_active: bool = True
    ) -> List[AnalyticsTemplate]:
        """Get analytics templates"""
        query = self.db.query(AnalyticsTemplate)
        
        if is_active is not None:
            query = query.filter(AnalyticsTemplate.is_active == is_active)
        
        if category:
            query = query.filter(AnalyticsTemplate.category == category)
        
        if analytics_type:
            query = query.filter(AnalyticsTemplate.analytics_type == analytics_type)
        
        return query.order_by(
            AnalyticsTemplate.category,
            AnalyticsTemplate.name
        ).all()
    
    # =====================================================
    # Report Operations
    # =====================================================
    
    def create_report(
        self,
        name: str,
        report_type: str,
        parameters: Dict[str, Any],
        generated_by: UUID,
        format: str = 'pdf',
        content: Optional[Dict[str, Any]] = None
    ) -> Report:
        """Create report record"""
        report = Report(
            name=name,
            report_type=report_type,
            format=format,
            parameters=parameters,
            content=content,
            status='generating',
            generated_by=generated_by
        )
        
        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)
        return report
    
    def update_report_status(
        self,
        report_id: UUID,
        status: str,
        file_url: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> Optional[Report]:
        """Update report generation status"""
        report = self.db.query(Report).filter(
            Report.id == report_id
        ).first()
        
        if not report:
            return None
        
        report.status = status
        if file_url:
            report.file_url = file_url
        if file_size_bytes:
            report.file_size_bytes = file_size_bytes
        if error_message:
            report.error_message = error_message
        
        self.db.commit()
        self.db.refresh(report)
        return report
    
    def get_user_reports(
        self,
        user_id: UUID,
        report_type: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 50
    ) -> Tuple[List[Report], int]:
        """Get user's generated reports"""
        query = self.db.query(Report).filter(
            Report.generated_by == user_id
        )
        
        if report_type:
            query = query.filter(Report.report_type == report_type)
        
        if status:
            query = query.filter(Report.status == status)
        
        total = query.count()
        
        reports = query.order_by(
            desc(Report.generated_at)
        ).offset(skip).limit(limit).all()
        
        return reports, total
    
    # =====================================================
    # Performance Metrics Operations
    # =====================================================
    
    def save_inventory_metrics(
        self,
        metric_date: date,
        metrics_data: Dict[str, Any],
        location_code: Optional[str] = None
    ) -> InventoryMetric:
        """Save inventory performance metrics"""
        metric = InventoryMetric(
            metric_date=metric_date,
            location_code=location_code,
            **metrics_data
        )
        
        self.db.add(metric)
        self.db.commit()
        self.db.refresh(metric)
        return metric
    
    def get_inventory_metrics_trend(
        self,
        start_date: date,
        end_date: date,
        location_code: Optional[str] = None,
        metric_name: str = 'inventory_turnover_ratio'
    ) -> List[Dict[str, Any]]:
        """Get inventory metrics trend"""
        query = self.db.query(
            InventoryMetric.metric_date,
            getattr(InventoryMetric, metric_name)
        ).filter(
            InventoryMetric.metric_date.between(start_date, end_date)
        )
        
        if location_code:
            query = query.filter(InventoryMetric.location_code == location_code)
        
        results = query.order_by(InventoryMetric.metric_date).all()
        
        return [
            {
                "date": result[0].isoformat(),
                "value": float(result[1]) if result[1] else 0
            }
            for result in results
        ]
    
    def save_delivery_performance(
        self,
        performance_data: Dict[str, Any]
    ) -> DeliveryPerformance:
        """Save delivery performance record"""
        performance = DeliveryPerformance(**performance_data)
        
        self.db.add(performance)
        self.db.commit()
        self.db.refresh(performance)
        return performance
    
    def get_delivery_performance_stats(
        self,
        start_date: datetime,
        end_date: datetime,
        carrier_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get delivery performance statistics"""
        query = self.db.query(DeliveryPerformance).filter(
            DeliveryPerformance.scheduled_delivery.between(start_date, end_date)
        )
        
        if carrier_name:
            query = query.filter(DeliveryPerformance.carrier_name == carrier_name)
        
        total_deliveries = query.count()
        on_time_deliveries = query.filter(DeliveryPerformance.on_time == True).count()
        
        avg_variance = self.db.query(
            func.avg(DeliveryPerformance.delivery_variance_hours)
        ).filter(
            DeliveryPerformance.scheduled_delivery.between(start_date, end_date),
            DeliveryPerformance.delivery_variance_hours.isnot(None)
        ).scalar() or 0
        
        return {
            "total_deliveries": total_deliveries,
            "on_time_deliveries": on_time_deliveries,
            "on_time_rate": (on_time_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0,
            "average_variance_hours": float(avg_variance)
        }
    
    # =====================================================
    # Network Analysis Operations
    # =====================================================
    
    def create_supply_chain_network(
        self,
        name: str,
        network_type: str,
        created_by: UUID,
        description: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None
    ) -> SupplyChainNetwork:
        """Create supply chain network"""
        network = SupplyChainNetwork(
            name=name,
            description=description,
            network_type=network_type,
            configuration=configuration or {},
            created_by=created_by,
            is_active=True
        )
        
        self.db.add(network)
        self.db.commit()
        self.db.refresh(network)
        return network
    
    def add_network_node(
        self,
        network_id: UUID,
        node_type: str,
        entity_id: UUID,
        entity_type: str,
        tier_level: Optional[int] = None,
        position: Optional[Dict[str, float]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> NetworkNode:
        """Add node to network"""
        node = NetworkNode(
            network_id=network_id,
            node_type=node_type,
            entity_id=entity_id,
            entity_type=entity_type,
            tier_level=tier_level,
            position_x=position.get('x') if position else None,
            position_y=position.get('y') if position else None,
            attributes=attributes or {}
        )
        
        self.db.add(node)
        self.db.commit()
        self.db.refresh(node)
        return node
    
    def add_network_edge(
        self,
        network_id: UUID,
        source_node_id: UUID,
        target_node_id: UUID,
        edge_type: str,
        weight: float = 1.0,
        capacity: Optional[float] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> NetworkEdge:
        """Add edge to network"""
        edge = NetworkEdge(
            network_id=network_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type=edge_type,
            weight=weight,
            capacity=capacity,
            attributes=attributes or {}
        )
        
        self.db.add(edge)
        self.db.commit()
        self.db.refresh(edge)
        return edge
    
    def get_network_by_id(
        self,
        network_id: UUID,
        include_nodes: bool = True,
        include_edges: bool = True
    ) -> Optional[SupplyChainNetwork]:
        """Get network by ID"""
        query = self.db.query(SupplyChainNetwork)
        
        if include_nodes:
            query = query.options(joinedload(SupplyChainNetwork.nodes))
        
        if include_edges:
            query = query.options(joinedload(SupplyChainNetwork.edges))
        
        return query.filter(SupplyChainNetwork.id == network_id).first()
    
    def save_bottleneck_analysis(
        self,
        network_id: UUID,
        analysis_date: datetime,
        bottleneck_type: str,
        severity_score: float,
        impact_assessment: Dict[str, Any],
        created_by: UUID,
        node_id: Optional[UUID] = None,
        edge_id: Optional[UUID] = None,
        affected_paths: Optional[List[Any]] = None,
        mitigation_options: Optional[List[str]] = None
    ) -> BottleneckAnalysis:
        """Save bottleneck analysis result"""
        analysis = BottleneckAnalysis(
            network_id=network_id,
            analysis_date=analysis_date,
            bottleneck_type=bottleneck_type,
            node_id=node_id,
            edge_id=edge_id,
            severity_score=severity_score,
            impact_assessment=impact_assessment,
            affected_paths=affected_paths or [],
            mitigation_options=mitigation_options or [],
            created_by=created_by
        )
        
        self.db.add(analysis)
        self.db.commit()
        self.db.refresh(analysis)
        return analysis
    
    def create_risk_scenario(
        self,
        name: str,
        network_id: UUID,
        scenario_type: str,
        disruption_source_nodes: List[UUID],
        disruption_parameters: Dict[str, Any],
        propagation_model: str,
        created_by: UUID
    ) -> RiskPropagationScenario:
        """Create risk propagation scenario"""
        scenario = RiskPropagationScenario(
            name=name,
            network_id=network_id,
            scenario_type=scenario_type,
            disruption_source_nodes=disruption_source_nodes,
            disruption_parameters=disruption_parameters,
            propagation_model=propagation_model,
            created_by=created_by
        )
        
        self.db.add(scenario)
        self.db.commit()
        self.db.refresh(scenario)
        return scenario
    
    def save_disruption_impacts(
        self,
        scenario_id: UUID,
        impacts: List[Dict[str, Any]]
    ) -> List[DisruptionImpact]:
        """Save disruption impact analysis results"""
        impact_records = []
        
        for impact_data in impacts:
            impact = DisruptionImpact(
                scenario_id=scenario_id,
                **impact_data
            )
            self.db.add(impact)
            impact_records.append(impact)
        
        self.db.commit()
        return impact_records
    
    # =====================================================
    # Analytics Summary Operations
    # =====================================================
    
    def get_analytics_summary(
        self,
        user_id: Optional[UUID] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get analytics usage summary"""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        query = self.db.query(AnalyticsResult)
        if user_id:
            query = query.filter(AnalyticsResult.created_by == user_id)
        
        query = query.filter(AnalyticsResult.created_at >= since_date)
        
        # Analytics by type
        analytics_by_type = dict(
            query.with_entities(
                AnalyticsResult.analytics_type,
                func.count(AnalyticsResult.id)
            ).group_by(AnalyticsResult.analytics_type).all()
        )
        
        # Success rate
        total_analytics = query.count()
        successful_analytics = query.filter(
            AnalyticsResult.status == 'completed'
        ).count()
        
        # Average execution time
        avg_execution_time = query.with_entities(
            func.avg(AnalyticsResult.execution_time_ms)
        ).filter(
            AnalyticsResult.execution_time_ms.isnot(None)
        ).scalar() or 0
        
        # Recent failures
        recent_failures = query.filter(
            AnalyticsResult.status == 'failed'
        ).order_by(desc(AnalyticsResult.created_at)).limit(5).all()
        
        return {
            "total_analytics_run": total_analytics,
            "successful_analytics": successful_analytics,
            "success_rate": (successful_analytics / total_analytics * 100) if total_analytics > 0 else 0,
            "analytics_by_type": analytics_by_type,
            "average_execution_time_ms": float(avg_execution_time),
            "recent_failures": [
                {
                    "id": str(f.id),
                    "type": f.analytics_type,
                    "error": f.error_message,
                    "created_at": f.created_at.isoformat()
                }
                for f in recent_failures
            ]
        }