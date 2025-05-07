"""
Risk cascade analyzer module for multi-tier supply chain analysis.

This module provides functionality to analyze how risks propagate through 
the supply chain network across multiple tiers of suppliers.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
import networkx as nx
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import random

from app.db.connectors.postgres import PostgresConnector
from app.db.schema.schema_discovery import discover_client_schema
from app.db.schema.schema_mapper import get_domain_mappings
from app.utils.logger import get_logger
from app.multiTier.supplier_mapping.network_builder import SupplierNetworkBuilder

# Initialize logger
logger = get_logger(__name__)

class RiskCascadeAnalyzer:
    """
    Analyzes how risks propagate through the supply chain network.
    
    This class analyzes risk scenarios to determine how supply chain
    disruptions can cascade through the network and impact operations.
    """
    
    def __init__(
        self,
        client_id: str,
        connection_id: Optional[str] = None,
        network_builder: Optional[SupplierNetworkBuilder] = None
    ):
        """
        Initialize the risk cascade analyzer.
        
        Args:
            client_id: Client identifier
            connection_id: Optional database connection identifier
            network_builder: Optional existing network builder instance
        """
        self.client_id = client_id
        self.connection_id = connection_id
        self.network_builder = network_builder
        self.db_connector = None
        self.schema = None
        self.domain_mappings = None
        self.graph = None
        self.risk_scenarios = {}
        self.risk_factors = {}
        
    async def initialize(self) -> None:
        """
        Initialize the cascade analyzer with client schema and connections.
        """
        try:
            # Get database connector
            self.db_connector = PostgresConnector(
                client_id=self.client_id,
                connection_id=self.connection_id
            )
            
            # Discover schema
            self.schema = await discover_client_schema(
                client_id=self.client_id,
                connection_id=self.connection_id
            )
            
            # Get domain mappings
            self.domain_mappings = await get_domain_mappings(self.client_id)
            
            # Initialize network builder if not provided
            if not self.network_builder:
                self.network_builder = SupplierNetworkBuilder(
                    client_id=self.client_id,
                    connection_id=self.connection_id
                )
                await self.network_builder.initialize()
            
            logger.info(f"Initialized risk cascade analyzer for client: {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error initializing risk cascade analyzer: {str(e)}")
            raise
    
    async def build_risk_model(
        self,
        graph: Optional[nx.DiGraph] = None,
        include_tier3_plus: bool = True,
        recalculate_network: bool = False,
        load_risk_factors: bool = True
    ) -> nx.DiGraph:
        """
        Build the risk model for cascade analysis.
        
        Args:
            graph: Optional existing supplier network graph
            include_tier3_plus: Whether to include tier 3+ suppliers
            recalculate_network: Whether to rebuild the network even if graph is provided
            load_risk_factors: Whether to load risk factors from database
            
        Returns:
            NetworkX directed graph with risk factors
        """
        try:
            if not self.db_connector:
                await self.initialize()
            
            # Build or use network
            if not graph or recalculate_network:
                self.graph = await self.network_builder.build_network(include_tier3_plus=include_tier3_plus)
            else:
                self.graph = graph
            
            # Load risk factors if requested
            if load_risk_factors:
                await self._load_risk_factors()
                
            # Apply risk factors to network
            await self._apply_risk_factors_to_network()
            
            logger.info(f"Built risk model with {len(self.graph.nodes)} nodes and {len(self.risk_factors)} risk factors")
            
            return self.graph
            
        except Exception as e:
            logger.error(f"Error building risk model: {str(e)}")
            raise
    
    async def _load_risk_factors(self) -> None:
        """
        Load risk factors for suppliers from the database.
        """
        try:
            # Reset risk factors
            self.risk_factors = {}
            
            # Find relevant tables for risk factors
            supplier_table = self._find_table_by_domain("supplier")
            risk_assessment_table = self._find_table_by_domain("risk_assessment")
            
            if not supplier_table:
                logger.warning("Could not find supplier table for risk factor loading")
                return
            
            # If we have a dedicated risk assessment table, use it
            if risk_assessment_table:
                query = f"""
                SELECT 
                    ra.supplier_id,
                    ra.risk_type,
                    ra.risk_score,
                    ra.last_assessment_date,
                    ra.assessment_notes
                FROM 
                    {risk_assessment_table} ra
                WHERE 
                    ra.is_active = TRUE
                """
                
                # Execute query
                result = await self.db_connector.execute_query(query)
                risk_assessments = result.get("data", [])
                
                # Process risk assessments
                for assessment in risk_assessments:
                    supplier_id = str(assessment.get("supplier_id"))
                    risk_type = assessment.get("risk_type", "unknown")
                    risk_score = assessment.get("risk_score", 0.0)
                    
                    if supplier_id not in self.risk_factors:
                        self.risk_factors[supplier_id] = {}
                    
                    self.risk_factors[supplier_id][risk_type] = risk_score
            
            # Load operational metrics as risk factors
            query = f"""
            SELECT 
                s.id AS supplier_id,
                s.country,
                s.financial_stability_score,
                s.compliance_score,
                s.capacity_utilization,
                s.single_source,
                COALESCE(po_metrics.on_time_delivery_rate, 0) AS on_time_delivery_rate,
                COALESCE(po_metrics.quality_rejection_rate, 0) AS quality_rejection_rate,
                COALESCE(po_metrics.avg_lead_time, 0) AS avg_lead_time,
                COALESCE(po_metrics.order_count, 0) AS order_count
            FROM 
                {supplier_table} s
            LEFT JOIN (
                SELECT 
                    po.supplier_id,
                    COUNT(po.id) AS order_count,
                    AVG(CASE WHEN po.delivery_date <= po.requested_date THEN 1 ELSE 0 END) AS on_time_delivery_rate,
                    AVG(po.rejection_rate) AS quality_rejection_rate,
                    AVG(po.lead_time) AS avg_lead_time
                FROM 
                    purchase_orders po
                WHERE 
                    po.order_date >= CURRENT_DATE - INTERVAL '1 year'
                GROUP BY 
                    po.supplier_id
            ) po_metrics ON s.id = po_metrics.supplier_id
            """
            
            # Execute query
            result = await self.db_connector.execute_query(query)
            supplier_metrics = result.get("data", [])
            
            # Process supplier metrics
            for supplier in supplier_metrics:
                supplier_id = str(supplier.get("supplier_id"))
                
                if supplier_id not in self.risk_factors:
                    self.risk_factors[supplier_id] = {}
                
                # Add operational metrics as risk factors
                if supplier.get("financial_stability_score") is not None:
                    self.risk_factors[supplier_id]["financial"] = 1.0 - (supplier.get("financial_stability_score") / 100.0)
                
                if supplier.get("compliance_score") is not None:
                    self.risk_factors[supplier_id]["compliance"] = 1.0 - (supplier.get("compliance_score") / 100.0)
                
                if supplier.get("capacity_utilization") is not None:
                    capacity = supplier.get("capacity_utilization") / 100.0
                    # High capacity utilization is a risk (less buffer)
                    if capacity > 0.9:
                        self.risk_factors[supplier_id]["capacity"] = (capacity - 0.9) * 10.0  # Scale from 0-1
                    else:
                        self.risk_factors[supplier_id]["capacity"] = 0.0
                
                if supplier.get("single_source") is not None and supplier.get("single_source"):
                    self.risk_factors[supplier_id]["single_source"] = 1.0
                else:
                    self.risk_factors[supplier_id]["single_source"] = 0.0
                
                # Add performance metrics as risk factors
                if supplier.get("on_time_delivery_rate") is not None:
                    otd_rate = supplier.get("on_time_delivery_rate")
                    self.risk_factors[supplier_id]["delivery"] = 1.0 - otd_rate
                
                if supplier.get("quality_rejection_rate") is not None:
                    self.risk_factors[supplier_id]["quality"] = supplier.get("quality_rejection_rate")
                
                # Add country-based risk (simplified)
                country = supplier.get("country")
                if country:
                    # In a real implementation, look up country risk from a geopolitical risk database
                    # For now, just assign random risk level to certain countries
                    high_risk_countries = ["Fictitious Country 1", "Fictitious Country 2"]
                    medium_risk_countries = ["Fictitious Country 3", "Fictitious Country 4"]
                    
                    if country in high_risk_countries:
                        self.risk_factors[supplier_id]["geopolitical"] = 0.8
                    elif country in medium_risk_countries:
                        self.risk_factors[supplier_id]["geopolitical"] = 0.5
                    else:
                        self.risk_factors[supplier_id]["geopolitical"] = 0.2
            
            # Calculate composite risk score for each supplier
            for supplier_id, risks in self.risk_factors.items():
                if risks:
                    # Weighted average of risk factors
                    weights = {
                        "financial": 0.2,
                        "compliance": 0.1,
                        "capacity": 0.15,
                        "single_source": 0.2,
                        "delivery": 0.15,
                        "quality": 0.1,
                        "geopolitical": 0.1
                    }
                    
                    total_weight = 0.0
                    weighted_score = 0.0
                    
                    for risk_type, score in risks.items():
                        weight = weights.get(risk_type, 0.1)
                        weighted_score += score * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        self.risk_factors[supplier_id]["composite"] = weighted_score / total_weight
                    else:
                        self.risk_factors[supplier_id]["composite"] = 0.0
            
            logger.info(f"Loaded risk factors for {len(self.risk_factors)} suppliers")
            
        except Exception as e:
            logger.error(f"Error loading risk factors: {str(e)}")
            # Create synthetic risk factors as fallback
            await self._create_synthetic_risk_factors()
    
    async def _create_synthetic_risk_factors(self) -> None:
        """
        Create synthetic risk factors when real data is not available.
        This is used for testing and when the client database doesn't have risk data.
        """
        try:
            # Reset risk factors
            self.risk_factors = {}
            
            # Generate synthetic risk factors for all suppliers in the network
            for node in self.graph.nodes():
                if node == "company":
                    continue
                    
                self.risk_factors[node] = {
                    "financial": random.uniform(0.0, 0.3),
                    "compliance": random.uniform(0.0, 0.4),
                    "capacity": random.uniform(0.0, 0.5),
                    "single_source": random.uniform(0.0, 1.0) > 0.7,
                    "delivery": random.uniform(0.0, 0.2),
                    "quality": random.uniform(0.0, 0.1),
                    "geopolitical": random.uniform(0.1, 0.6),
                }
                
                # Calculate composite risk score
                risk_values = [v for k, v in self.risk_factors[node].items() if isinstance(v, (int, float))]
                if risk_values:
                    self.risk_factors[node]["composite"] = sum(risk_values) / len(risk_values)
                else:
                    self.risk_factors[node]["composite"] = 0.0
            
            logger.info(f"Created synthetic risk factors for {len(self.risk_factors)} suppliers")
            
        except Exception as e:
            logger.error(f"Error creating synthetic risk factors: {str(e)}")
            raise
    
    async def _apply_risk_factors_to_network(self) -> None:
        """
        Apply risk factors to the network graph.
        """
        try:
            # Ensure we have a graph
            if not self.graph:
                logger.error("No network graph available")
                return
            
            # Apply risk factors to nodes
            for node in self.graph.nodes():
                if node == "company":
                    continue
                    
                risk_data = self.risk_factors.get(node, {})
                
                # Add risk data to node attributes
                for risk_type, value in risk_data.items():
                    self.graph.nodes[node][f"risk_{risk_type}"] = value
                
                # Set default composite risk if not available
                if "risk_composite" not in self.graph.nodes[node]:
                    self.graph.nodes[node]["risk_composite"] = 0.2  # Default low risk
            
            # Calculate edge risk factors based on node risk and relationship
            for source, target in self.graph.edges():
                # Get risk scores from nodes
                source_risk = self.graph.nodes[source].get("risk_composite", 0.0)
                target_risk = self.graph.nodes[target].get("risk_composite", 0.0)
                
                # Calculate edge risk factor (propagation risk)
                # Higher source risk and higher target risk = higher propagation risk
                edge_risk = (source_risk + target_risk) / 2.0
                
                # Adjust based on relationship type
                edge_type = self.graph.edges[source, target].get("type", "")
                
                if edge_type == "single_source":
                    # Single-source relationships have higher propagation risk
                    edge_risk *= 1.5
                
                # Ensure risk is in 0-1 range
                edge_risk = min(max(edge_risk, 0.0), 1.0)
                
                # Set edge risk attribute
                self.graph.edges[source, target]["risk_propagation"] = edge_risk
                
            logger.info("Applied risk factors to network")
            
        except Exception as e:
            logger.error(f"Error applying risk factors to network: {str(e)}")
            raise
    
    async def analyze_disruption_scenario(
        self,
        disrupted_suppliers: List[str],
        disruption_duration: int = 30,  # days
        disruption_severity: float = 1.0,
        recovery_rate: float = 0.1,  # 10% recovery per day
        simulation_days: int = 90
    ) -> Dict[str, Any]:
        """
        Analyze a disruption scenario starting from specified suppliers.
        
        Args:
            disrupted_suppliers: List of supplier IDs experiencing initial disruption
            disruption_duration: Duration of the disruption in days
            disruption_severity: Severity of the disruption (0-1)
            recovery_rate: Daily recovery rate after disruption period
            simulation_days: Number of days to simulate
            
        Returns:
            Dictionary with cascade analysis results
        """
        try:
            if not self.graph:
                await self.build_risk_model()
            
            # Create a copy of the graph for simulation
            sim_graph = self.graph.copy()
            
            # Initialize disruption state for all nodes
            for node in sim_graph.nodes():
                sim_graph.nodes[node]["disruption"] = 0.0
                sim_graph.nodes[node]["recovery_day"] = 0
            
            # Set initial disruption for specified suppliers
            for supplier_id in disrupted_suppliers:
                if sim_graph.has_node(supplier_id):
                    sim_graph.nodes[supplier_id]["disruption"] = disruption_severity
                    sim_graph.nodes[supplier_id]["recovery_day"] = disruption_duration
            
            # Track disruption over time
            time_series = []
            impacted_suppliers = {0: set(disrupted_suppliers)}
            
            # Run simulation for specified number of days
            for day in range(1, simulation_days + 1):
                # Track disrupted suppliers on this day
                impacted_suppliers[day] = set()
                
                # Process node state for this day
                for node in sim_graph.nodes():
                    # Skip the company node
                    if node == "company":
                        continue
                        
                    current_disruption = sim_graph.nodes[node]["disruption"]
                    recovery_day = sim_graph.nodes[node]["recovery_day"]
                    
                    # Process recovery
                    if current_disruption > 0:
                        if day >= recovery_day:
                            # Apply recovery rate
                            new_disruption = max(0, current_disruption - recovery_rate)
                            sim_graph.nodes[node]["disruption"] = new_disruption
                        
                        # Track as impacted if still disrupted
                        if sim_graph.nodes[node]["disruption"] > 0:
                            impacted_suppliers[day].add(node)
                    
                    # If this supplier is disrupted, propagate to downstream suppliers
                    if current_disruption > 0:
                        self._propagate_disruption(sim_graph, node, day, disruption_duration)
                
                # Record state for this day
                time_series.append(self._capture_daily_state(sim_graph, day))
            
            # Analyze the results
            impact_analysis = self._analyze_simulation_results(
                sim_graph, 
                time_series, 
                impacted_suppliers,
                disrupted_suppliers
            )
            
            # Store scenario for later reference
            scenario_id = f"scenario_{len(self.risk_scenarios) + 1}"
            self.risk_scenarios[scenario_id] = {
                "id": scenario_id,
                "disrupted_suppliers": disrupted_suppliers,
                "disruption_duration": disruption_duration,
                "disruption_severity": disruption_severity,
                "recovery_rate": recovery_rate,
                "simulation_days": simulation_days,
                "results": impact_analysis
            }
            
            logger.info(f"Analyzed disruption scenario with {len(disrupted_suppliers)} initial suppliers")
            
            return self.risk_scenarios[scenario_id]
            
        except Exception as e:
            logger.error(f"Error analyzing disruption scenario: {str(e)}")
            raise
    
    def _propagate_disruption(
        self,
        graph: nx.DiGraph,
        node: str,
        day: int,
        disruption_duration: int
    ) -> None:
        """
        Propagate disruption from a node to its downstream suppliers.
        
        Args:
            graph: Simulation graph
            node: Node experiencing disruption
            day: Current simulation day
            disruption_duration: Base duration of disruption
        """
        # Get current disruption level
        disruption_level = graph.nodes[node]["disruption"]
        
        # If no significant disruption, no propagation
        if disruption_level < 0.1:
            return
        
        # Get downstream nodes (customers of this supplier)
        for successor in graph.predecessors(node):
            # Skip if trying to propagate to company node
            if successor == "company":
                continue
                
            # Calculate propagated disruption
            # Factors:
            # 1. Source disruption level
            # 2. Edge propagation risk
            # 3. Target node resilience (1 - risk_composite)
            edge_risk = graph.edges[successor, node].get("risk_propagation", 0.5)
            target_resilience = 1.0 - graph.nodes[successor].get("risk_composite", 0.0)
            
            # Propagated disruption is reduced by target resilience
            propagated_disruption = disruption_level * edge_risk * (1.0 - target_resilience)
            
            # Apply minimum threshold for disruption propagation
            if propagated_disruption < 0.05:
                continue
                
            # Get current target disruption
            current_target_disruption = graph.nodes[successor]["disruption"]
            
            # Only update if propagated disruption is higher
            if propagated_disruption > current_target_disruption:
                graph.nodes[successor]["disruption"] = propagated_disruption
                
                # Set recovery day based on propagation (add delay for propagation)
                recovery_delay = int(3 * edge_risk)  # 0-3 days delay based on edge risk
                graph.nodes[successor]["recovery_day"] = day + disruption_duration + recovery_delay
    
    def _capture_daily_state(self, graph: nx.DiGraph, day: int) -> Dict[str, Any]:
        """
        Capture the state of the network for a specific day.
        
        Args:
            graph: Simulation graph
            day: Current simulation day
            
        Returns:
            Dictionary with network state for the day
        """
        # Count suppliers by disruption level
        disruption_levels = {
            "none": 0,        # 0
            "minor": 0,       # 0.01-0.3
            "moderate": 0,    # 0.3-0.6
            "severe": 0       # 0.6-1.0
        }
        
        # Track disrupted suppliers
        disrupted_suppliers = []
        
        # Calculate overall network disruption
        total_suppliers = 0
        total_disruption = 0.0
        
        for node, data in graph.nodes(data=True):
            if node == "company":
                continue
                
            total_suppliers += 1
            disruption = data.get("disruption", 0.0)
            total_disruption += disruption
            
            # Categorize disruption level
            if disruption <= 0.0:
                disruption_levels["none"] += 1
            elif disruption <= 0.3:
                disruption_levels["minor"] += 1
            elif disruption <= 0.6:
                disruption_levels["moderate"] += 1
            else:
                disruption_levels["severe"] += 1
            
            # Track significantly disrupted suppliers
            if disruption >= 0.3:
                disrupted_suppliers.append({
                    "id": node,
                    "name": graph.nodes[node].get("name", "Unknown"),
                    "tier": graph.nodes[node].get("tier", 0),
                    "disruption": disruption,
                    "recovery_day": data.get("recovery_day", 0)
                })
        
        # Calculate average disruption
        avg_disruption = total_disruption / total_suppliers if total_suppliers > 0 else 0.0
        
        # Capture network state
        return {
            "day": day,
            "disruption_levels": disruption_levels,
            "disrupted_suppliers_count": len(disrupted_suppliers),
            "avg_disruption": avg_disruption,
            "disrupted_suppliers": disrupted_suppliers
        }
    
    def _analyze_simulation_results(
        self,
        graph: nx.DiGraph,
        time_series: List[Dict[str, Any]],
        impacted_suppliers: Dict[int, Set[str]],
        initially_disrupted: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze the results of a disruption simulation.
        
        Args:
            graph: Simulation graph
            time_series: Daily state snapshots
            impacted_suppliers: Mapping of day to set of impacted suppliers
            initially_disrupted: List of initially disrupted suppliers
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Track disruption spread metrics
            max_disruption_day = 0
            max_disruption_count = 0
            max_disruption_avg = 0.0
            
            disruption_duration = 0
            recovery_time = len(time_series)
            
            # Find day with maximum disruption
            for day_state in time_series:
                day = day_state["day"]
                disrupted_count = day_state["disrupted_suppliers_count"]
                avg_disruption = day_state["avg_disruption"]
                
                if disrupted_count > max_disruption_count:
                    max_disruption_day = day
                    max_disruption_count = disrupted_count
                    max_disruption_avg = avg_disruption
                
                # Count days with significant disruption
                if avg_disruption > 0.1:  # Arbitrary threshold
                    disruption_duration += 1
                    
                # Find recovery time (all suppliers recovered)
                if disrupted_count == 0 and day > max_disruption_day:
                    recovery_time = day
                    break
            
            # Calculate cascade metrics
            initial_count = len(initially_disrupted)
            cascade_ratio = max_disruption_count / initial_count if initial_count > 0 else 0
            
            # Analyze impact by tier
            tier_impact = {}
            for node, data in graph.nodes(data=True):
                if node == "company":
                    continue
                    
                tier = data.get("tier", 0)
                if tier not in tier_impact:
                    tier_impact[tier] = {
                        "supplier_count": 0,
                        "disrupted_count": 0,
                        "avg_disruption": 0.0
                    }
                
                tier_impact[tier]["supplier_count"] += 1
                
                # Check if significantly disrupted during simulation
                max_disruption = 0.0
                for day_state in time_series:
                    for supplier in day_state["disrupted_suppliers"]:
                        if supplier["id"] == node and supplier["disruption"] > max_disruption:
                            max_disruption = supplier["disruption"]
                
                if max_disruption > 0.3:  # Arbitrary threshold
                    tier_impact[tier]["disrupted_count"] += 1
                
                tier_impact[tier]["avg_disruption"] += max_disruption
            
            # Calculate averages
            for tier, data in tier_impact.items():
                if data["supplier_count"] > 0:
                    data["avg_disruption"] = data["avg_disruption"] / data["supplier_count"]
                    data["disruption_percentage"] = (data["disrupted_count"] / data["supplier_count"]) * 100
            
            # Find critical path of disruption
            critical_path = self._find_critical_disruption_path(graph, initially_disrupted, time_series)
            
            # Compile results
            results = {
                "summary": {
                    "initial_disruption_count": initial_count,
                    "max_disruption_count": max_disruption_count,
                    "max_disruption_day": max_disruption_day,
                    "cascade_ratio": cascade_ratio,
                    "disruption_duration": disruption_duration,
                    "recovery_time": recovery_time,
                    "max_disruption_percentage": (max_disruption_count / (len(graph.nodes) - 1)) * 100
                },
                "tier_impact": tier_impact,
                "critical_path": critical_path,
                "time_series": time_series
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing simulation results: {str(e)}")
            # Return partial results
            return {
                "summary": {
                    "error": str(e)
                },
                "time_series": time_series
            }
    
    def _find_critical_disruption_path(
        self,
        graph: nx.DiGraph,
        initially_disrupted: List[str],
        time_series: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find the critical path of disruption propagation.
        
        Args:
            graph: Simulation graph
            initially_disrupted: List of initially disrupted suppliers
            time_series: Daily state snapshots
            
        Returns:
            List of nodes representing the critical disruption path
        """
        # Start with initially disrupted suppliers
        critical_path = []
        for supplier_id in initially_disrupted:
            if graph.has_node(supplier_id):
                critical_path.append({
                    "id": supplier_id,
                    "name": graph.nodes[supplier_id].get("name", "Unknown"),
                    "tier": graph.nodes[supplier_id].get("tier", 0),
                    "step": 0,
                    "disruption": 1.0
                })
        
        # Track propagation through simulation
        propagation_steps = {}
        max_step = 0
        
        # Analyze daily disruptions to find propagation order
        for day_idx, day_state in enumerate(time_series):
            day = day_state["day"]
            
            # Skip day 0 (initial state)
            if day == 0:
                continue
            
            # Get newly disrupted suppliers for this day
            newly_disrupted = set()
            for supplier in day_state["disrupted_suppliers"]:
                supplier_id = supplier["id"]
                
                # Skip initially disrupted suppliers
                if supplier_id in initially_disrupted:
                    continue
                
                # Skip if already counted in propagation steps
                if supplier_id in propagation_steps:
                    continue
                
                # Check if this supplier was not disrupted on previous day
                was_disrupted_before = False
                if day_idx > 0:
                    for prev_supplier in time_series[day_idx-1]["disrupted_suppliers"]:
                        if prev_supplier["id"] == supplier_id:
                            was_disrupted_before = True
                            break
                
                if not was_disrupted_before:
                    newly_disrupted.add(supplier_id)
                    
                    # Determine step in propagation
                    # This is an approximation - in reality would need causal tracing
                    step = day  # Use day as a proxy for step
                    propagation_steps[supplier_id] = step
                    max_step = max(max_step, step)
            
            # For each newly disrupted supplier, find the most likely source
            for supplier_id in newly_disrupted:
                # Look at upstream suppliers that were already disrupted
                potential_sources = []
                for src, tgt in graph.edges():
                    if tgt == supplier_id:
                        # Check if source was disrupted before this supplier
                        src_step = propagation_steps.get(src, -1)
                        if src_step >= 0 and src_step < day:
                            edge_risk = graph.edges[src, tgt].get("risk_propagation", 0.5)
                            potential_sources.append((src, edge_risk))
                
                # Sort by edge risk (highest first)
                potential_sources.sort(key=lambda x: x[1], reverse=True)
                
                # Add to critical path if we found a source
                if potential_sources:
                    source_id, edge_risk = potential_sources[0]
                    disruption = graph.nodes[supplier_id].get("disruption", 0.0)
                    
                    critical_path.append({
                        "id": supplier_id,
                        "name": graph.nodes[supplier_id].get("name", "Unknown"),
                        "tier": graph.nodes[supplier_id].get("tier", 0),
                        "step": day,
                        "disruption": disruption,
                        "source_id": source_id,
                        "source_name": graph.nodes[source_id].get("name", "Unknown"),
                        "propagation_risk": edge_risk
                    })
        
        # Sort by step
        critical_path.sort(key=lambda x: x.get("step", 0))
        
        return critical_path
    
    def _find_table_by_domain(self, domain_concept: str) -> Optional[str]:
        """
        Find the table name corresponding to a domain concept.
        
        Args:
            domain_concept: Domain concept to look for
            
        Returns:
            Table name or None if not found
        """
        if not self.domain_mappings:
            return None
        
        # Look for exact match in domain mappings
        for mapping in self.domain_mappings:
            if mapping.get("domain_concept") == domain_concept and mapping.get("custom_column") is None:
                return mapping.get("custom_table")
        
        # Fall back to table names in schema that might match
        if self.schema and hasattr(self.schema, "tables"):
            for table in self.schema.tables:
                table_name = table.get("name", "").lower()
                if domain_concept in table_name:
                    return table.get("name")
        
        return None
    
    async def compare_scenarios(
        self,
        scenario_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple scenarios to identify key differences.
        
        Args:
            scenario_ids: List of scenario IDs to compare
            
        Returns:
            Comparative analysis of scenarios
        """
        try:
            # Validate scenarios
            scenarios = []
            for scenario_id in scenario_ids:
                if scenario_id in self.risk_scenarios:
                    scenarios.append(self.risk_scenarios[scenario_id])
                else:
                    logger.warning(f"Scenario {scenario_id} not found")
            
            if len(scenarios) < 2:
                raise ValueError("At least two valid scenarios required for comparison")
            
            # Prepare comparison data
            comparison = {
                "scenarios": [],
                "impact_comparison": {},
                "recovery_comparison": {},
                "tier_impact_comparison": {}
            }
            
            # Extract scenario data
            for scenario in scenarios:
                scenario_summary = {
                    "id": scenario["id"],
                    "disrupted_suppliers": len(scenario["disrupted_suppliers"]),
                    "disruption_duration": scenario["disruption_duration"],
                    "disruption_severity": scenario["disruption_severity"],
                    "recovery_rate": scenario["recovery_rate"]
                }
                
                # Add summary metrics
                summary = scenario.get("results", {}).get("summary", {})
                for key, value in summary.items():
                    scenario_summary[key] = value
                
                comparison["scenarios"].append(scenario_summary)
            
            # Compare impact metrics
            impact_metrics = ["max_disruption_count", "cascade_ratio", "max_disruption_percentage"]
            for metric in impact_metrics:
                comparison["impact_comparison"][metric] = []
                for scenario in comparison["scenarios"]:
                    comparison["impact_comparison"][metric].append({
                        "scenario_id": scenario["id"],
                        "value": scenario.get(metric, 0)
                    })
                
                # Sort by metric value (descending)
                comparison["impact_comparison"][metric].sort(key=lambda x: x["value"], reverse=True)
            
            # Compare recovery metrics
            recovery_metrics = ["disruption_duration", "recovery_time"]
            for metric in recovery_metrics:
                comparison["recovery_comparison"][metric] = []
                for scenario in comparison["scenarios"]:
                    comparison["recovery_comparison"][metric].append({
                        "scenario_id": scenario["id"],
                        "value": scenario.get(metric, 0)
                    })
                
                # Sort by metric value (ascending is better for recovery)
                comparison["recovery_comparison"][metric].sort(key=lambda x: x["value"])
            
            # Compare tier impact
            tiers = set()
            for scenario in scenarios:
                tier_impact = scenario.get("results", {}).get("tier_impact", {})
                for tier in tier_impact:
                    tiers.add(tier)
            
            for tier in sorted(tiers):
                comparison["tier_impact_comparison"][f"tier_{tier}"] = []
                for scenario in scenarios:
                    tier_impact = scenario.get("results", {}).get("tier_impact", {}).get(tier, {})
                    comparison["tier_impact_comparison"][f"tier_{tier}"].append({
                        "scenario_id": scenario["id"],
                        "disruption_percentage": tier_impact.get("disruption_percentage", 0),
                        "avg_disruption": tier_impact.get("avg_disruption", 0)
                    })
                
                # Sort by disruption percentage (ascending)
                comparison["tier_impact_comparison"][f"tier_{tier}"].sort(
                    key=lambda x: x["disruption_percentage"]
                )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing scenarios: {str(e)}")
            raise
    
    async def get_high_risk_suppliers(
        self,
        risk_threshold: float = 0.7,
        centrality_weight: float = 0.3,
        tier1_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get high-risk suppliers based on risk factors and network position.
        
        Args:
            risk_threshold: Threshold for high risk categorization
            centrality_weight: Weight given to network centrality in risk calculation
            tier1_only: Whether to include only tier 1 suppliers
            
        Returns:
            List of high-risk suppliers with risk data
        """
        try:
            if not self.graph:
                await self.build_risk_model()
            
            high_risk_suppliers = []
            
            for node, data in self.graph.nodes(data=True):
                if node == "company":
                    continue
                
                # Skip non-tier 1 suppliers if tier1_only is True
                if tier1_only and data.get("tier", 0) != 1:
                    continue
                
                # Get base risk score
                risk_composite = data.get("risk_composite", 0.0)
                
                # Adjust risk score with network centrality
                centrality = data.get("betweenness_centrality", 0.0)
                downstream_impact = data.get("downstream_impact", 0)
                
                # Normalize downstream impact
                max_downstream = 10  # Arbitrary max value for normalization
                normalized_downstream = min(downstream_impact / max_downstream, 1.0)
                
                # Calculate network position risk
                network_risk = (centrality + normalized_downstream) / 2.0
                
                # Calculate weighted total risk
                total_risk = (risk_composite * (1 - centrality_weight)) + (network_risk * centrality_weight)
                
                # Check if above threshold
                if total_risk >= risk_threshold:
                    high_risk_suppliers.append({
                        "id": node,
                        "name": data.get("name", "Unknown"),
                        "tier": data.get("tier", 0),
                        "risk_composite": risk_composite,
                        "network_risk": network_risk,
                        "total_risk": total_risk,
                        "risk_factors": {
                            k.replace("risk_", ""): v for k, v in data.items() 
                            if k.startswith("risk_") and k != "risk_composite"
                        },
                        "centrality": centrality,
                        "downstream_impact": downstream_impact
                    })
            
            # Sort by total risk (descending)
            high_risk_suppliers.sort(key=lambda x: x["total_risk"], reverse=True)
            
            return high_risk_suppliers
            
        except Exception as e:
            logger.error(f"Error getting high-risk suppliers: {str(e)}")
            raise
    
    def export_risk_model(self, output_format: str = "json") -> Any:
        """
        Export the risk model data.
        
        Args:
            output_format: Format to export (json, csv, etc.)
            
        Returns:
            Exported risk model in the specified format
        """
        if not self.graph:
            raise ValueError("No risk model available. Run build_risk_model() first.")
        
        # Export graph data
        nodes_data = []
        for node, data in self.graph.nodes(data=True):
            node_data = {
                "id": node,
                "name": data.get("name", "Unknown"),
                "tier": data.get("tier", 0)
            }
            
            # Add risk factors
            for key, value in data.items():
                if key.startswith("risk_"):
                    node_data[key] = value
            
            # Add network metrics
            for metric in ["betweenness_centrality", "downstream_impact"]:
                if metric in data:
                    node_data[metric] = data[metric]
            
            nodes_data.append(node_data)
        
        edges_data = []
        for source, target, data in self.graph.edges(data=True):
            edge_data = {
                "source": source,
                "source_name": self.graph.nodes[source].get("name", "Unknown"),
                "target": target,
                "target_name": self.graph.nodes[target].get("name", "Unknown"),
                "type": data.get("type", "unknown")
            }
            
            # Add risk factors
            for key, value in data.items():
                if key.startswith("risk_"):
                    edge_data[key] = value
            
            edges_data.append(edge_data)
        
        # Create export data
        export_data = {
            "nodes": nodes_data,
            "edges": edges_data,
            "risk_factors": self.risk_factors
        }
        
        # Export in requested format
        if output_format == "json":
            return json.dumps(export_data, indent=2)
        elif output_format == "csv":
            # Export nodes and edges as separate DataFrames
            nodes_df = pd.DataFrame(nodes_data)
            edges_df = pd.DataFrame(edges_data)
            
            # Return a dict with both DataFrames
            return {
                "nodes": nodes_df.to_csv(index=False),
                "edges": edges_df.to_csv(index=False)
            }
        else:
            raise ValueError(f"Unsupported output format: {output_format}")