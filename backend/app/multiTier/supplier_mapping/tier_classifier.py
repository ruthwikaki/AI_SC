"""
Supplier tier classifier module for multi-tier supply chain analysis.

This module provides functionality to classify suppliers into tiers
based on their relationships and roles within the supply chain.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx
import pandas as pd
from datetime import datetime
import logging
import json

from app.db.connectors.postgres import PostgresConnector
from app.db.schema.schema_discovery import discover_client_schema
from app.db.schema.schema_mapper import get_domain_mappings
from app.utils.logger import get_logger
from app.multiTier.supplier_mapping.network_builder import SupplierNetworkBuilder

# Initialize logger
logger = get_logger(__name__)

class SupplierTierClassifier:
    """
    Classifies suppliers into tiers based on network analysis and business rules.
    
    This class analyzes supplier relationships to determine their position
    in the supply chain hierarchy and their strategic importance.
    """
    
    def __init__(
        self,
        client_id: str,
        connection_id: Optional[str] = None,
        network_builder: Optional[SupplierNetworkBuilder] = None
    ):
        """
        Initialize the supplier tier classifier.
        
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
        self.supplier_tiers = {}
        self.strategic_suppliers = set()
        
    async def initialize(self) -> None:
        """
        Initialize the tier classifier with client schema and connections.
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
            
            logger.info(f"Initialized supplier tier classifier for client: {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error initializing supplier tier classifier: {str(e)}")
            raise
    
    async def classify_tiers(
        self,
        graph: Optional[nx.DiGraph] = None,
        include_strategic_analysis: bool = True,
        recalculate_network: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Classify suppliers into tiers based on network position and business data.
        
        Args:
            graph: Optional existing supplier network graph
            include_strategic_analysis: Whether to identify strategic suppliers
            recalculate_network: Whether to rebuild the network even if graph is provided
            
        Returns:
            Dictionary mapping supplier IDs to their tier information
        """
        try:
            if not self.db_connector:
                await self.initialize()
            
            # Build or use network
            if not graph or recalculate_network:
                graph = await self.network_builder.build_network(include_tier3_plus=True)
            
            # Reset tier classifications
            self.supplier_tiers = {}
            self.strategic_suppliers = set()
            
            # Classify based on network structure
            await self._classify_by_network_position(graph)
            
            # Identify strategic suppliers if requested
            if include_strategic_analysis:
                await self._identify_strategic_suppliers(graph)
            
            # Format and return tier classifications
            result = {}
            for supplier_id, tier_info in self.supplier_tiers.items():
                # Skip the company node
                if supplier_id == "company":
                    continue
                    
                result[supplier_id] = {
                    "supplier_id": supplier_id,
                    "name": graph.nodes[supplier_id].get("name", "Unknown"),
                    "network_tier": tier_info.get("network_tier"),
                    "adjusted_tier": tier_info.get("adjusted_tier"),
                    "is_strategic": supplier_id in self.strategic_suppliers,
                    "bottleneck_risk": tier_info.get("bottleneck_risk", 0),
                    "tier_confidence": tier_info.get("tier_confidence", 1.0),
                    "metrics": {
                        "centrality": graph.nodes[supplier_id].get("betweenness_centrality", 0),
                        "downstream_impact": graph.nodes[supplier_id].get("downstream_impact", 0),
                        "risk_propagation": graph.nodes[supplier_id].get("risk_propagation", 0)
                    }
                }
            
            logger.info(f"Classified {len(result)} suppliers into tiers")
            return result
            
        except Exception as e:
            logger.error(f"Error classifying supplier tiers: {str(e)}")
            raise
    
    async def _classify_by_network_position(self, graph: nx.DiGraph) -> None:
        """
        Classify suppliers based on their position in the network.
        
        Args:
            graph: Supplier network graph
        """
        try:
            # Start with network-based tier classification from node attributes
            for node, data in graph.nodes(data=True):
                network_tier = data.get("tier")
                if network_tier is not None:
                    self.supplier_tiers[node] = {
                        "network_tier": network_tier,
                        "adjusted_tier": network_tier,
                        "tier_confidence": 1.0
                    }
            
            # Adjust classifications based on relationship patterns
            # Look for inconsistencies in the network structure
            for node in graph.nodes():
                if node == "company":
                    continue
                    
                tier_info = self.supplier_tiers.get(node, {})
                network_tier = tier_info.get("network_tier")
                
                if network_tier is None:
                    continue
                
                # Check for suppliers that should be in a different tier
                # Example: Tier 2 supplier that also supplies directly to the company
                if network_tier > 1 and "company" in graph.predecessors(node):
                    # This supplier is both tier 1 and a deeper tier
                    tier_info["adjusted_tier"] = 1
                    tier_info["tier_confidence"] = 0.9
                    tier_info["notes"] = "Reclassified as Tier 1 due to direct relationship with company"
                
                # Look for bottleneck suppliers (those that many tier 1 suppliers depend on)
                if network_tier > 1:
                    tier1_predecessors = [pred for pred in graph.predecessors(node) 
                                         if self.supplier_tiers.get(pred, {}).get("network_tier") == 1]
                    
                    if len(tier1_predecessors) >= 3:  # Arbitrary threshold - depends on industry
                        bottleneck_risk = min(1.0, len(tier1_predecessors) / 10.0)  # Scale from 0 to 1
                        tier_info["bottleneck_risk"] = bottleneck_risk
                        tier_info["notes"] = f"Potential bottleneck supplier used by {len(tier1_predecessors)} tier 1 suppliers"
                
                # Update the tier info
                self.supplier_tiers[node] = tier_info
                
        except Exception as e:
            logger.error(f"Error classifying by network position: {str(e)}")
            raise
    
    async def _identify_strategic_suppliers(self, graph: nx.DiGraph) -> None:
        """
        Identify strategic suppliers based on various criteria.
        
        Args:
            graph: Supplier network graph
        """
        try:
            # Get spending data for suppliers
            supplier_table = self._find_table_by_domain("supplier")
            purchase_order_table = self._find_table_by_domain("purchase_order")
            
            if not supplier_table or not purchase_order_table:
                logger.warning("Could not find required tables for strategic supplier analysis")
                return
            
            # Build query to get supplier spend and risk metrics
            query = f"""
            SELECT 
                s.id,
                s.name,
                SUM(po.total_amount) AS annual_spend,
                COUNT(DISTINCT po.id) AS order_count,
                AVG(po.lead_time) AS avg_lead_time,
                COUNT(DISTINCT p.id) AS unique_products,
                (SELECT COUNT(*) FROM shipments sh 
                 WHERE sh.supplier_id = s.id AND sh.status = 'delayed') AS delayed_shipments
            FROM 
                {supplier_table} s
            LEFT JOIN 
                {purchase_order_table} po ON s.id = po.supplier_id
            LEFT JOIN 
                order_items oi ON po.id = oi.order_id
            LEFT JOIN 
                products p ON oi.product_id = p.id
            WHERE 
                po.order_date >= DATE_TRUNC('year', CURRENT_DATE)
            GROUP BY 
                s.id, s.name
            """
            
            # Execute query
            result = await self.db_connector.execute_query(query)
            supplier_metrics = result.get("data", [])
            
            # Process supplier metrics
            supplier_data = {}
            for supplier in supplier_metrics:
                supplier_id = str(supplier.get("id"))
                if supplier_id not in self.supplier_tiers:
                    continue
                    
                annual_spend = supplier.get("annual_spend", 0)
                unique_products = supplier.get("unique_products", 0)
                avg_lead_time = supplier.get("avg_lead_time", 0)
                delayed_shipments = supplier.get("delayed_shipments", 0)
                order_count = supplier.get("order_count", 0)
                
                # Calculate metrics
                if order_count > 0:
                    delay_rate = delayed_shipments / order_count
                else:
                    delay_rate = 0
                
                supplier_data[supplier_id] = {
                    "annual_spend": annual_spend,
                    "unique_products": unique_products,
                    "avg_lead_time": avg_lead_time,
                    "delay_rate": delay_rate
                }
            
            # Calculate spend thresholds for Pareto analysis (80/20 rule)
            if supplier_data:
                total_spend = sum(s["annual_spend"] for s in supplier_data.values())
                if total_spend > 0:
                    # Sort suppliers by spend in descending order
                    sorted_suppliers = sorted(
                        supplier_data.items(),
                        key=lambda x: x[1]["annual_spend"],
                        reverse=True
                    )
                    
                    # Calculate cumulative spend
                    cumulative_spend = 0
                    for supplier_id, data in sorted_suppliers:
                        cumulative_spend += data["annual_spend"]
                        data["spend_percentile"] = cumulative_spend / total_spend
                        supplier_data[supplier_id] = data
                        
                        # Mark top spend suppliers as strategic (80% of spend)
                        if cumulative_spend / total_spend <= 0.8:
                            self.strategic_suppliers.add(supplier_id)
            
            # Identify suppliers with unique/critical products or materials
            # This would typically come from a separate analysis or manual categorization
            # For demonstration, we'll use the unique products count as a proxy
            for supplier_id, data in supplier_data.items():
                if data.get("unique_products", 0) > 10:  # Arbitrary threshold
                    self.strategic_suppliers.add(supplier_id)
            
            # Identify strategic suppliers based on network centrality
            for node, data in graph.nodes(data=True):
                if node == "company":
                    continue
                    
                # Consider highly central suppliers as strategic (potential bottlenecks)
                centrality = data.get("betweenness_centrality", 0)
                if centrality > 0.1:  # Arbitrary threshold
                    self.strategic_suppliers.add(node)
                
                # Consider suppliers with high downstream impact as strategic
                downstream_impact = data.get("downstream_impact", 0)
                if downstream_impact > 5:  # Arbitrary threshold
                    self.strategic_suppliers.add(node)
            
            logger.info(f"Identified {len(self.strategic_suppliers)} strategic suppliers")
            
        except Exception as e:
            logger.error(f"Error identifying strategic suppliers: {str(e)}")
            # Continue without strategic classification
    
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
    
    async def assign_tier_labels(
        self,
        suppliers: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Assign descriptive tier labels to suppliers.
        
        Args:
            suppliers: Dictionary of supplier tier information
            
        Returns:
            Updated supplier information with tier labels
        """
        # Define tier label mappings
        tier_labels = {
            0: "Company",
            1: "Direct Supplier",
            2: "Sub-Supplier",
            3: "Tier 3 Supplier",
            4: "Extended Network Supplier",
            5: "Deep Tier Supplier"
        }
        
        # Assign strategic tier labels
        strategic_labels = {
            1: "Strategic Direct Supplier",
            2: "Strategic Sub-Supplier",
            3: "Strategic Tier 3 Supplier",
            4: "Strategic Extended Network Supplier",
            5: "Strategic Deep Tier Supplier"
        }
        
        # Assign bottleneck labels
        bottleneck_threshold = 0.5  # Arbitrary threshold
        bottleneck_labels = {
            2: "Bottleneck Sub-Supplier",
            3: "Bottleneck Tier 3 Supplier",
            4: "Bottleneck Extended Network Supplier",
            5: "Bottleneck Deep Tier Supplier"
        }
        
        # Apply labels
        for supplier_id, info in suppliers.items():
            tier = info.get("adjusted_tier")
            is_strategic = info.get("is_strategic", False)
            bottleneck_risk = info.get("bottleneck_risk", 0)
            
            # Default tier label
            if tier in tier_labels:
                info["tier_label"] = tier_labels[tier]
            else:
                info["tier_label"] = f"Tier {tier} Supplier"
            
            # Apply strategic label if applicable
            if is_strategic and tier in strategic_labels:
                info["tier_label"] = strategic_labels[tier]
            
            # Apply bottleneck label if applicable
            if bottleneck_risk >= bottleneck_threshold and tier in bottleneck_labels:
                info["tier_label"] = bottleneck_labels[tier]
        
        return suppliers
    
    async def export_tier_classifications(
        self,
        output_format: str = "json"
    ) -> Any:
        """
        Export tier classifications in the specified format.
        
        Args:
            output_format: Format to export (json, csv, etc.)
            
        Returns:
            Exported data in the specified format
        """
        if not self.supplier_tiers:
            raise ValueError("No tier classifications available. Run classify_tiers() first.")
        
        # Build classification data
        classifications = []
        for supplier_id, tier_info in self.supplier_tiers.items():
            if supplier_id == "company":
                continue
                
            classifications.append({
                "supplier_id": supplier_id,
                "network_tier": tier_info.get("network_tier"),
                "adjusted_tier": tier_info.get("adjusted_tier"),
                "is_strategic": supplier_id in self.strategic_suppliers,
                "bottleneck_risk": tier_info.get("bottleneck_risk", 0),
                "tier_confidence": tier_info.get("tier_confidence", 1.0),
                "notes": tier_info.get("notes", "")
            })
        
        # Export in requested format
        if output_format == "json":
            return json.dumps(classifications, indent=2)
        elif output_format == "csv":
            df = pd.DataFrame(classifications)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")