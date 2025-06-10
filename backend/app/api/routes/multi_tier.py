# backend/app/api/routes/multi_tier.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.db.database import get_db
from app.api.middleware.auth import get_current_user
from app.models.user import User
from app.models.supply_chain import Supplier, Order, Inventory
from app.multiTier.network_visualization.graph_builder import GraphBuilder
from app.multiTier.network_visualization.bottleneck_identifier import BottleneckIdentifier
from app.multiTier.risk_propagation.cascade_analyzer import CascadeAnalyzer
from app.multiTier.risk_propagation.impact_calculator import ImpactCalculator
from app.multiTier.supplier_mapping.network_builder import NetworkBuilder
from app.multiTier.supplier_mapping.tier_classifier import TierClassifier
from app.schemas.supply_chain import (
    NetworkGraphResponse,
    RiskAnalysisResponse,
    ScenarioSimulationRequest,
    ScenarioSimulationResponse,
    SupplierTierResponse
)

router = APIRouter(prefix="/api/multi-tier", tags=["multi-tier"])

@router.get("/network/visualization", response_model=NetworkGraphResponse)
async def get_network_visualization(
    supplier_id: Optional[int] = Query(None, description="Root supplier ID"),
    depth: int = Query(3, ge=1, le=5, description="Network depth to visualize"),
    include_inactive: bool = Query(False, description="Include inactive suppliers"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get supply chain network visualization data"""
    try:
        # Get suppliers based on filters
        query = db.query(Supplier)
        if not include_inactive:
            query = query.filter(Supplier.status == 'active')
        
        suppliers = query.all()
        
        # Build network graph
        graph_builder = GraphBuilder(db)
        network_data = graph_builder.build_network(
            suppliers=suppliers,
            root_supplier_id=supplier_id,
            depth=depth
        )
        
        # Identify bottlenecks
        bottleneck_identifier = BottleneckIdentifier(db)
        bottlenecks = bottleneck_identifier.identify_bottlenecks(network_data)
        
        return NetworkGraphResponse(
            nodes=network_data['nodes'],
            edges=network_data['edges'],
            bottlenecks=bottlenecks,
            metrics={
                'total_suppliers': len(suppliers),
                'active_connections': len(network_data['edges']),
                'identified_bottlenecks': len(bottlenecks)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk/analysis", response_model=RiskAnalysisResponse)
async def analyze_supply_chain_risk(
    risk_type: Optional[str] = Query(None, description="Type of risk to analyze"),
    time_horizon: int = Query(30, description="Time horizon in days"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Analyze supply chain risks and propagation patterns"""
    try:
        # Initialize analyzers
        cascade_analyzer = CascadeAnalyzer(db)
        impact_calculator = ImpactCalculator(db)
        
        # Get current supply chain state
        suppliers = db.query(Supplier).filter(Supplier.status == 'active').all()
        
        # Analyze cascading risks
        cascade_risks = cascade_analyzer.analyze_cascade_risks(
            suppliers=suppliers,
            risk_type=risk_type,
            time_horizon=time_horizon
        )
        
        # Calculate impact metrics
        impact_metrics = impact_calculator.calculate_impacts(
            cascade_risks=cascade_risks,
            time_horizon=time_horizon
        )
        
        return RiskAnalysisResponse(
            risk_scores=cascade_risks['risk_scores'],
            propagation_paths=cascade_risks['propagation_paths'],
            impact_metrics=impact_metrics,
            recommendations=cascade_risks['recommendations'],
            analysis_timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scenario/simulate", response_model=ScenarioSimulationResponse)
async def simulate_scenario(
    scenario: ScenarioSimulationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Simulate what-if scenarios for supply chain disruptions"""
    try:
        # Initialize simulation components
        cascade_analyzer = CascadeAnalyzer(db)
        impact_calculator = ImpactCalculator(db)
        
        # Run simulation
        simulation_results = cascade_analyzer.simulate_disruption(
            disrupted_suppliers=scenario.disrupted_suppliers,
            disruption_severity=scenario.severity,
            duration_days=scenario.duration_days,
            disruption_type=scenario.disruption_type
        )
        
        # Calculate recovery strategies
        recovery_strategies = impact_calculator.calculate_recovery_strategies(
            simulation_results=simulation_results,
            target_recovery_time=scenario.target_recovery_time
        )
        
        return ScenarioSimulationResponse(
            scenario_id=simulation_results['scenario_id'],
            affected_suppliers=simulation_results['affected_suppliers'],
            supply_impact=simulation_results['supply_impact'],
            financial_impact=simulation_results['financial_impact'],
            recovery_strategies=recovery_strategies,
            timeline=simulation_results['timeline']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/suppliers/tiers", response_model=List[SupplierTierResponse])
async def get_supplier_tiers(
    recalculate: bool = Query(False, description="Force recalculation of tiers"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get supplier tier classification"""
    try:
        tier_classifier = TierClassifier(db)
        
        # Get or calculate supplier tiers
        if recalculate:
            tier_data = tier_classifier.classify_all_suppliers()
        else:
            tier_data = tier_classifier.get_cached_tiers()
        
        return [
            SupplierTierResponse(
                supplier_id=item['supplier_id'],
                supplier_name=item['supplier_name'],
                tier_level=item['tier_level'],
                tier_score=item['tier_score'],
                connections_count=item['connections_count'],
                risk_level=item['risk_level']
            )
            for item in tier_data
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/network/metrics")
async def get_network_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive network metrics"""
    try:
        network_builder = NetworkBuilder(db)
        metrics = network_builder.calculate_network_metrics()
        
        return {
            "network_density": metrics['density'],
            "average_path_length": metrics['avg_path_length'],
            "clustering_coefficient": metrics['clustering_coefficient'],
            "centrality_measures": metrics['centrality_measures'],
            "resilience_score": metrics['resilience_score'],
            "last_updated": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))