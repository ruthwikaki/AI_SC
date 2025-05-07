"""
Bottleneck identifier module for multi-tier supply chain network analysis.

This module provides functionality to identify bottlenecks and
critical nodes in the supply chain network.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx
import pandas as pd
from datetime import datetime
import logging
import json
import math

from app.db.connectors.postgres import PostgresConnector
from app.db.schema.schema_discovery import discover_client_schema
from app.db.schema.schema_mapper import get_domain_mappings
from app.utils.logger import get_logger
from app.multiTier.supplier_mapping.network_builder import SupplierNetworkBuilder

# Initialize logger
logger = get_logger(__name__)

class SupplyChainBottleneckIdentifier:
    """
    Identifies bottlenecks and critical nodes in the supply chain network.
    
    This class analyzes the network structure and supplier data to identify
    bottlenecks, single points of failure, and other critical vulnerabilities.
    """
    
    def __init__(
        self,
        client_id: str,
        connection_id: Optional[str] = None,
        network_builder: Optional[SupplierNetworkBuilder] = None
    ):
        """
        Initialize the bottleneck identifier.
        
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
        self.bottlenecks = []
        self.critical_paths = []
        
    async def initialize(self) -> None:
        """
        Initialize the bottleneck identifier with client schema and connections.
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
            
            logger.info(f"Initialized bottleneck identifier for client: {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error initializing bottleneck identifier: {str(e)}")
            raise
    
    async def identify_bottlenecks(
        self,
        graph: Optional[nx.DiGraph] = None,
        recalculate_network: bool = False,
        include_tier3_plus: bool = True,
        min_centrality: float = 0.1,
        min_downstream_impact: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Identify bottlenecks in the supplier network.
        
        Args:
            graph: Optional existing supplier network graph
            recalculate_network: Whether to rebuild the network even if graph is provided
            include_tier3_plus: Whether to include tier 3+ suppliers
            min_centrality: Minimum centrality score for bottleneck identification
            min_downstream_impact: Minimum downstream impact for bottleneck identification
            
        Returns:
            List of identified bottlenecks with metadata
        """
        try:
            if not self.db_connector:
                await self.initialize()
            
            # Build or use network
            if not graph or recalculate_network:
                self.graph = await self.network_builder.build_network(include_tier3_plus=include_tier3_plus)
            else:
                self.graph = graph
            
            # Reset bottlenecks
            self.bottlenecks = []
            
            # Apply standard network analysis
            await self._apply_network_analysis()
            
            # Identify bottlenecks based on network metrics
            for node, data in self.graph.nodes(data=True):
                # Skip the company node
                if node == "company":
                    continue
                
                # Get key metrics
                centrality = data.get("betweenness_centrality", 0)
                downstream_impact = data.get("downstream_impact", 0)
                tier = data.get("tier", 0)
                
                # Check if this node is a potential bottleneck based on network position
                is_bottleneck = (centrality >= min_centrality) or (downstream_impact >= min_downstream_impact)
                
                if is_bottleneck:
                    bottleneck = {
                        "id": node,
                        "name": data.get("name", "Unknown"),
                        "tier": tier,
                        "centrality": centrality,
                        "downstream_impact": downstream_impact,
                        "risk_score": data.get("risk_composite", 0) if "risk_composite" in data else None,
                        "bottleneck_type": []
                    }
                    
                    # Determine bottleneck type
                    if centrality >= min_centrality:
                        bottleneck["bottleneck_type"].append("network_centrality")
                    
                    if downstream_impact >= min_downstream_impact:
                        bottleneck["bottleneck_type"].append("downstream_dependence")
                    
                    # Check for single source situations
                    if self._is_single_source(node):
                        bottleneck["bottleneck_type"].append("single_source")
                    
                    # Add to identified bottlenecks
                    self.bottlenecks.append(bottleneck)
            
            # Add capacity bottlenecks from supplier data
            await self._identify_capacity_bottlenecks()
            
            # Add material-based bottlenecks from product/material relationships
            await self._identify_material_bottlenecks()
            
            # Sort bottlenecks by impact score
            for bottleneck in self.bottlenecks:
                # Calculate impact score (combination of centrality and downstream impact)
                centrality = bottleneck.get("centrality", 0)
                downstream_impact = bottleneck.get("downstream_impact", 0)
                risk_score = bottleneck.get("risk_score", 0) or 0
                
                impact_score = (centrality * 0.3) + (downstream_impact / 10 * 0.5) + (risk_score * 0.2)
                bottleneck["impact_score"] = impact_score
                
                # Determine severity
                if impact_score >= 0.7:
                    bottleneck["severity"] = "high"
                elif impact_score >= 0.4:
                    bottleneck["severity"] = "medium"
                else:
                    bottleneck["severity"] = "low"
            
            # Sort by impact score (descending)
            self.bottlenecks.sort(key=lambda x: x.get("impact_score", 0), reverse=True)
            
            logger.info(f"Identified {len(self.bottlenecks)} bottlenecks in the supplier network")
            
            return self.bottlenecks
            
        except Exception as e:
            logger.error(f"Error identifying bottlenecks: {str(e)}")
            raise
    
    async def _apply_network_analysis(self) -> None:
        """
        Apply network analysis metrics to the graph.
        """
        try:
            # Calculate node-level metrics
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            # Apply metrics to nodes
            for node in self.graph.nodes():
                self.graph.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
                self.graph.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
                
                # Calculate additional metrics
                if node != "company":
                    # Dependency on this supplier (downstream impact)
                    try:
                        downstream_nodes = nx.descendants(self.graph, node)
                        downstream_count = len(downstream_nodes)
                        self.graph.nodes[node]['downstream_impact'] = downstream_count
                    except nx.NetworkXError:
                        # Handle non-reachable nodes
                        self.graph.nodes[node]['downstream_impact'] = 0
            
            # Calculate edge betweenness centrality
            edge_betweenness = nx.edge_betweenness_centrality(self.graph)
            
            # Apply edge metrics
            for edge, centrality in edge_betweenness.items():
                source, target = edge
                self.graph.edges[source, target]['edge_betweenness'] = centrality
            
            logger.info("Applied network analysis metrics")
            
        except Exception as e:
            logger.error(f"Error applying network analysis: {str(e)}")
            # Continue without metrics
    
    def _is_single_source(self, node_id: str) -> bool:
        """
        Check if the node is a single source supplier for any material.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if single source, False otherwise
        """
        try:
            # Get predecessors (customers of this supplier)
            predecessors = list(self.graph.predecessors(node_id))
            
            # Check if any predecessor exclusively sources from this supplier
            for predecessor in predecessors:
                # Skip company node
                if predecessor == "company":
                    continue
                
                # Get all suppliers of this predecessor
                suppliers = list(self.graph.successors(predecessor))
                
                # If this supplier is the only one, it's a single source
                if len(suppliers) == 1 and suppliers[0] == node_id:
                    return True
            
            # Check if this supplier provides unique materials
            if "materials" in self.graph.nodes[node_id]:
                materials = self.graph.nodes[node_id]["materials"]
                
                # Check if any material is uniquely provided by this supplier
                for material in materials:
                    material_id = material.get("id")
                    
                    # Find other suppliers of this material
                    other_suppliers = []
                    for other_node in self.graph.nodes():
                        if other_node != node_id and "materials" in self.graph.nodes[other_node]:
                            other_materials = self.graph.nodes[other_node]["materials"]
                            if any(m.get("id") == material_id for m in other_materials):
                                other_suppliers.append(other_node)
                    
                    # If no other suppliers, this is a single source
                    if not other_suppliers:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking single source for node {node_id}: {str(e)}")
            return False
    
    async def _identify_capacity_bottlenecks(self) -> None:
        """
        Identify bottlenecks based on supplier capacity constraints.
        """
        try:
            # Find supplier table
            supplier_table = self._find_table_by_domain("supplier")
            
            if not supplier_table:
                logger.warning("Could not find supplier table for capacity analysis")
                return
            
            # Query capacity data
            query = f"""
            SELECT 
                id,
                name,
                capacity_utilization,
                max_capacity,
                current_load
            FROM 
                {supplier_table}
            WHERE 
                capacity_utilization > 80
            """
            
            # Execute query
            result = await self.db_connector.execute_query(query)
            suppliers = result.get("data", [])
            
            # Process capacity-constrained suppliers
            for supplier in suppliers:
                supplier_id = str(supplier.get("id"))
                
                # Skip if not in network
                if not self.graph.has_node(supplier_id):
                    continue
                
                # Skip if already identified as a bottleneck
                if any(b["id"] == supplier_id for b in self.bottlenecks):
                    # Update existing bottleneck
                    for bottleneck in self.bottlenecks:
                        if bottleneck["id"] == supplier_id:
                            bottleneck["capacity_utilization"] = supplier.get("capacity_utilization")
                            bottleneck["max_capacity"] = supplier.get("max_capacity")
                            bottleneck["current_load"] = supplier.get("current_load")
                            
                            if "capacity_constraint" not in bottleneck["bottleneck_type"]:
                                bottleneck["bottleneck_type"].append("capacity_constraint")
                            
                            # Update impact score
                            capacity_utilization = supplier.get("capacity_utilization", 0) / 100
                            centrality = bottleneck.get("centrality", 0)
                            downstream_impact = bottleneck.get("downstream_impact", 0)
                            risk_score = bottleneck.get("risk_score", 0) or 0
                            
                            impact_score = (
                                (centrality * 0.3) + 
                                (downstream_impact / 10 * 0.4) + 
                                (risk_score * 0.1) + 
                                (capacity_utilization * 0.2)
                            )
                            bottleneck["impact_score"] = impact_score
                else:
                    # Create new bottleneck
                    capacity_utilization = supplier.get("capacity_utilization", 0) / 100
                    
                    # Get node data from graph
                    node_data = self.graph.nodes[supplier_id]
                    centrality = node_data.get("betweenness_centrality", 0)
                    downstream_impact = node_data.get("downstream_impact", 0)
                    risk_score = node_data.get("risk_composite", 0) if "risk_composite" in node_data else 0
                    
                    impact_score = (
                        (centrality * 0.3) + 
                        (downstream_impact / 10 * 0.4) + 
                        (risk_score * 0.1) + 
                        (capacity_utilization * 0.2)
                    )
                    
                    # Create bottleneck entry
                    bottleneck = {
                        "id": supplier_id,
                        "name": supplier.get("name", "Unknown"),
                        "tier": node_data.get("tier", 0),
                        "centrality": centrality,
                        "downstream_impact": downstream_impact,
                        "risk_score": risk_score,
                        "bottleneck_type": ["capacity_constraint"],
                        "capacity_utilization": supplier.get("capacity_utilization"),
                        "max_capacity": supplier.get("max_capacity"),
                        "current_load": supplier.get("current_load"),
                        "impact_score": impact_score
                    }
                    
                    # Determine severity
                    if impact_score >= 0.7:
                        bottleneck["severity"] = "high"
                    elif impact_score >= 0.4:
                        bottleneck["severity"] = "medium"
                    else:
                        bottleneck["severity"] = "low"
                    
                    self.bottlenecks.append(bottleneck)
            
            logger.info(f"Identified {len(suppliers)} capacity bottlenecks")
            
        except Exception as e:
            logger.error(f"Error identifying capacity bottlenecks: {str(e)}")
            # Continue without capacity bottlenecks
    
    async def _identify_material_bottlenecks(self) -> None:
        """
        Identify bottlenecks based on material dependencies.
        """
        try:
            # Find material and supplier-material tables
            material_table = self._find_table_by_domain("material")
            material_supplier_table = self._find_table_by_domain("material_supplier")
            product_material_table = self._find_table_by_domain("bill_of_materials")
            
            if not material_table or not material_supplier_table or not product_material_table:
                logger.warning("Could not find required tables for material bottleneck analysis")
                return
            
            # Query critical materials (used in multiple products with limited suppliers)
            query = f"""
            SELECT 
                m.id AS material_id,
                m.name AS material_name,
                COUNT(DISTINCT bom.product_id) AS product_count,
                COUNT(DISTINCT ms.supplier_id) AS supplier_count,
                (SELECT STRING_AGG(DISTINCT s.id::text, ',') 
                 FROM {material_supplier_table} ms2 
                 JOIN suppliers s ON ms2.supplier_id = s.id 
                 WHERE ms2.material_id = m.id) AS supplier_ids
            FROM 
                {material_table} m
            JOIN 
                {product_material_table} bom ON m.id = bom.material_id
            JOIN 
                {material_supplier_table} ms ON m.id = ms.material_id
            GROUP BY 
                m.id, m.name
            HAVING 
                COUNT(DISTINCT bom.product_id) >= 3
                AND COUNT(DISTINCT ms.supplier_id) <= 2
            """
            
            # Execute query
            result = await self.db_connector.execute_query(query)
            critical_materials = result.get("data", [])
            
            # Process critical materials
            for material in critical_materials:
                material_id = str(material.get("material_id"))
                supplier_ids = material.get("supplier_ids", "").split(",")
                product_count = material.get("product_count", 0)
                
                # Check each supplier of this material
                for supplier_id in supplier_ids:
                    if not supplier_id:
                        continue
                    
                    # Skip if not in network
                    if not self.graph.has_node(supplier_id):
                        continue
                    
                    # Get node data from graph
                    node_data = self.graph.nodes[supplier_id]
                    
                    # Check if already a bottleneck
                    existing_bottleneck = next((b for b in self.bottlenecks if b["id"] == supplier_id), None)
                    
                    if existing_bottleneck:
                        # Update existing bottleneck
                        if "critical_material" not in existing_bottleneck["bottleneck_type"]:
                            existing_bottleneck["bottleneck_type"].append("critical_material")
                        
                        # Add material info if not present
                        if "critical_materials" not in existing_bottleneck:
                            existing_bottleneck["critical_materials"] = []
                        
                        existing_bottleneck["critical_materials"].append({
                            "id": material_id,
                            "name": material.get("material_name"),
                            "product_count": product_count,
                            "supplier_count": material.get("supplier_count")
                        })
                        
                        # Update impact score to account for critical material
                        impact_score = existing_bottleneck.get("impact_score", 0)
                        # Increase impact score based on number of dependent products
                        material_impact = min(1.0, product_count / 10) * 0.2
                        existing_bottleneck["impact_score"] = min(1.0, impact_score + material_impact)
                    else:
                        # Create new bottleneck
                        centrality = node_data.get("betweenness_centrality", 0)
                        downstream_impact = node_data.get("downstream_impact", 0)
                        risk_score = node_data.get("risk_composite", 0) if "risk_composite" in node_data else 0
                        
                        # Calculate impact score with material dependency
                        material_impact = min(1.0, product_count / 10) * 0.2
                        impact_score = (
                            (centrality * 0.3) + 
                            (downstream_impact / 10 * 0.4) + 
                            (risk_score * 0.1) + 
                            material_impact
                        )
                        
                        # Create bottleneck entry
                        bottleneck = {
                            "id": supplier_id,
                            "name": node_data.get("name", "Unknown"),
                            "tier": node_data.get("tier", 0),
                            "centrality": centrality,
                            "downstream_impact": downstream_impact,
                            "risk_score": risk_score,
                            "bottleneck_type": ["critical_material"],
                            "critical_materials": [{
                                "id": material_id,
                                "name": material.get("material_name"),
                                "product_count": product_count,
                                "supplier_count": material.get("supplier_count")
                            }],
                            "impact_score": impact_score
                        }
                        
                        # Determine severity
                        if impact_score >= 0.7:
                            bottleneck["severity"] = "high"
                        elif impact_score >= 0.4:
                            bottleneck["severity"] = "medium"
                        else:
                            bottleneck["severity"] = "low"
                        
                        self.bottlenecks.append(bottleneck)
            
            logger.info(f"Identified {len(critical_materials)} material-based bottlenecks")
            
        except Exception as e:
            logger.error(f"Error identifying material bottlenecks: {str(e)}")
            # Continue without material bottlenecks
    
    async def identify_critical_paths(
        self,
        target_products: Optional[List[str]] = None,
        top_products: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify critical supply paths for important products.
        
        Args:
            target_products: Optional list of specific product IDs to analyze
            top_products: Number of top products to analyze if target_products not provided
            
        Returns:
            List of critical paths with metadata
        """
        try:
            if not self.graph:
                # Ensure bottleneck analysis is done
                await self.identify_bottlenecks()
            
            # Reset critical paths
            self.critical_paths = []
            
            # Get product data
            products = await self._get_product_data(target_products, top_products)
            
            # Analyze each product
            for product in products:
                product_id = str(product.get("id"))
                
                # Get materials for this product
                materials = await self._get_product_materials(product_id)
                
                # Analyze each material's supply path
                for material in materials:
                    material_id = str(material.get("id"))
                    material_name = material.get("name", "Unknown")
                    
                    # Get suppliers for this material
                    suppliers = await self._get_material_suppliers(material_id)
                    
                    # For each supplier, find path to bottlenecks
                    for supplier in suppliers:
                        supplier_id = str(supplier.get("id"))
                        supplier_name = supplier.get("name", "Unknown")
                        
                        # Skip if supplier not in network
                        if not self.graph.has_node(supplier_id):
                            continue
                        
                        # Add company as source of path
                        path = [{
                            "id": "company",
                            "name": "Your Company",
                            "type": "company",
                            "tier": 0
                        }]
                        
                        # Get supplier data from graph
                        if self.graph.has_node(supplier_id):
                            supplier_data = self.graph.nodes[supplier_id]
                            supplier_tier = supplier_data.get("tier", 1)
                            
                            # Add supplier to path
                            path.append({
                                "id": supplier_id,
                                "name": supplier_name,
                                "type": "supplier",
                                "tier": supplier_tier,
                                "material_id": material_id,
                                "material_name": material_name
                            })
                            
                            # Find bottlenecks in this supplier's upstream chain
                            bottleneck_paths = self._find_bottleneck_paths(supplier_id)
                            
                            for bottleneck_path in bottleneck_paths:
                                # Create complete path from company to bottleneck
                                full_path = path.copy()
                                full_path.extend(bottleneck_path)
                                
                                # Get bottleneck node (last in path)
                                bottleneck_node = bottleneck_path[-1]
                                bottleneck_id = bottleneck_node.get("id")
                                
                                # Get bottleneck metadata from our bottlenecks list
                                bottleneck_meta = next(
                                    (b for b in self.bottlenecks if b.get("id") == bottleneck_id), 
                                    {}
                                )
                                
                                # Create critical path entry
                                critical_path = {
                                    "product_id": product_id,
                                    "product_name": product.get("name", "Unknown"),
                                    "material_id": material_id,
                                    "material_name": material_name,
                                    "path": full_path,
                                    "bottleneck_id": bottleneck_id,
                                    "bottleneck_name": bottleneck_node.get("name", "Unknown"),
                                    "bottleneck_tier": bottleneck_node.get("tier", 0),
                                    "bottleneck_types": bottleneck_meta.get("bottleneck_type", []),
                                    "severity": bottleneck_meta.get("severity", "low"),
                                    "impact_score": bottleneck_meta.get("impact_score", 0)
                                }
                                
                                self.critical_paths.append(critical_path)
            
            # Sort critical paths by impact score
            self.critical_paths.sort(key=lambda x: x.get("impact_score", 0), reverse=True)
            
            logger.info(f"Identified {len(self.critical_paths)} critical paths")
            
            return self.critical_paths
            
        except Exception as e:
            logger.error(f"Error identifying critical paths: {str(e)}")
            raise
    
    def _find_bottleneck_paths(
        self,
        start_node: str,
        max_depth: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths from start node to bottlenecks.
        
        Args:
            start_node: Starting node ID
            max_depth: Maximum path depth to search
            
        Returns:
            List of paths to bottlenecks
        """
        # Get bottleneck nodes
        bottleneck_nodes = [b["id"] for b in self.bottlenecks]
        
        # Perform depth-first search to find paths
        paths = []
        visited = set()
        
        def dfs(node: str, current_path: List[Dict[str, Any]], depth: int):
            # Check depth limit
            if depth > max_depth:
                return
            
            # Mark as visited
            visited.add(node)
            
            # Get successors (suppliers of this node)
            for supplier in self.graph.successors(node):
                # Skip if already visited (avoid cycles)
                if supplier in visited:
                    continue
                
                # Get supplier data
                supplier_data = self.graph.nodes[supplier]
                
                # Create node entry
                supplier_node = {
                    "id": supplier,
                    "name": supplier_data.get("name", "Unknown"),
                    "type": "supplier",
                    "tier": supplier_data.get("tier", 0)
                }
                
                # Add edge data
                if self.graph.has_edge(node, supplier):
                    edge_data = self.graph.edges[node, supplier]
                    supplier_node["material_id"] = edge_data.get("material_id")
                    supplier_node["material_name"] = edge_data.get("material_name")
                
                # Check if this is a bottleneck
                if supplier in bottleneck_nodes:
                    # Found a path to a bottleneck
                    paths.append(current_path + [supplier_node])
                
                # Continue search
                dfs(supplier, current_path + [supplier_node], depth + 1)
            
            # Backtrack
            visited.remove(node)
        
        # Start DFS from start node
        dfs(start_node, [], 0)
        
        return paths
    
    async def _get_product_data(
        self,
        target_products: Optional[List[str]] = None,
        top_products: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get product data for critical path analysis.
        
        Args:
            target_products: Optional list of specific product IDs to analyze
            top_products: Number of top products to analyze if target_products not provided
            
        Returns:
            List of product data dictionaries
        """
        try:
            # Find product table
            product_table = self._find_table_by_domain("product")
            
            if not product_table:
                logger.warning("Could not find product table for critical path analysis")
                return []
            
            # Build query based on target products
            if target_products:
                query = f"""
                SELECT 
                    p.*,
                    COALESCE(sales.revenue, 0) AS revenue,
                    COALESCE(sales.quantity, 0) AS sales_quantity
                FROM 
                    {product_table} p
                LEFT JOIN (
                    SELECT 
                        product_id,
                        SUM(quantity * price) AS revenue,
                        SUM(quantity) AS quantity
                    FROM 
                        sales
                    WHERE 
                        sale_date >= CURRENT_DATE - INTERVAL '90 days'
                    GROUP BY 
                        product_id
                ) sales ON p.id = sales.product_id
                WHERE 
                    p.id IN :product_ids
                """
                
                params = {"product_ids": tuple(target_products)}
            else:
                # Get top products by revenue
                query = f"""
                SELECT 
                    p.*,
                    COALESCE(sales.revenue, 0) AS revenue,
                    COALESCE(sales.quantity, 0) AS sales_quantity
                FROM 
                    {product_table} p
                LEFT JOIN (
                    SELECT 
                        product_id,
                        SUM(quantity * price) AS revenue,
                        SUM(quantity) AS quantity
                    FROM 
                        sales
                    WHERE 
                        sale_date >= CURRENT_DATE - INTERVAL '90 days'
                    GROUP BY 
                        product_id
                ) sales ON p.id = sales.product_id
                ORDER BY 
                    sales.revenue DESC NULLS LAST
                LIMIT {top_products}
                """
                
                params = {}
            
            # Execute query
            result = await self.db_connector.execute_query(query, params)
            return result.get("data", [])
            
        except Exception as e:
            logger.error(f"Error getting product data: {str(e)}")
            return []
    
    async def _get_product_materials(self, product_id: str) -> List[Dict[str, Any]]:
        """
        Get materials used in a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            List of material data dictionaries
        """
        try:
            # Find bill of materials table
            bom_table = self._find_table_by_domain("bill_of_materials")
            material_table = self._find_table_by_domain("material")
            
            if not bom_table or not material_table:
                logger.warning("Could not find BOM or material table for critical path analysis")
                return []
            
            # Query materials for this product
            query = f"""
            SELECT 
                m.*,
                bom.quantity_required
            FROM 
                {bom_table} bom
            JOIN 
                {material_table} m ON bom.material_id = m.id
            WHERE 
                bom.product_id = :product_id
            """
            
            params = {"product_id": product_id}
            
            # Execute query
            result = await self.db_connector.execute_query(query, params)
            return result.get("data", [])
            
        except Exception as e:
            logger.error(f"Error getting product materials: {str(e)}")
            return []
    
    async def _get_material_suppliers(self, material_id: str) -> List[Dict[str, Any]]:
        """
        Get suppliers that provide a material.
        
        Args:
            material_id: Material identifier
            
        Returns:
            List of supplier data dictionaries
        """
        try:
            # Find material supplier table
            material_supplier_table = self._find_table_by_domain("material_supplier")
            supplier_table = self._find_table_by_domain("supplier")
            
            if not material_supplier_table or not supplier_table:
                logger.warning("Could not find material_supplier or supplier table for critical path analysis")
                return []
            
            # Query suppliers for this material
            query = f"""
            SELECT 
                s.*,
                ms.is_primary,
                ms.lead_time,
                ms.price_per_unit
            FROM 
                {material_supplier_table} ms
            JOIN 
                {supplier_table} s ON ms.supplier_id = s.id
            WHERE 
                ms.material_id = :material_id
            ORDER BY 
                ms.is_primary DESC
            """
            
            params = {"material_id": material_id}
            
            # Execute query
            result = await self.db_connector.execute_query(query, params)
            return result.get("data", [])
            
        except Exception as e:
            logger.error(f"Error getting material suppliers: {str(e)}")
            return []
    
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
    
    async def analyze_bottleneck_mitigation(
        self,
        bottleneck_id: str
    ) -> Dict[str, Any]:
        """
        Analyze mitigation strategies for a specific bottleneck.
        
        Args:
            bottleneck_id: ID of the bottleneck to analyze
            
        Returns:
            Dictionary with mitigation analysis
        """
        try:
            # Find the bottleneck
            bottleneck = next((b for b in self.bottlenecks if b["id"] == bottleneck_id), None)
            
            if not bottleneck:
                raise ValueError(f"Bottleneck {bottleneck_id} not found")
            
            # Get bottleneck data
            bottleneck_type = bottleneck.get("bottleneck_type", [])
            tier = bottleneck.get("tier", 0)
            
            # Initialize mitigation strategies
            mitigation_strategies = []
            
            # Generate strategies based on bottleneck type
            if "single_source" in bottleneck_type:
                # Strategies for single source bottlenecks
                mitigation_strategies.extend([
                    {
                        "strategy": "alternative_supplier",
                        "description": f"Identify and qualify alternative suppliers for {bottleneck['name']}",
                        "implementation_time": "medium",
                        "cost": "medium",
                        "effectiveness": "high"
                    },
                    {
                        "strategy": "inventory_buffer",
                        "description": "Increase safety stock levels for materials supplied by this supplier",
                        "implementation_time": "short",
                        "cost": "medium",
                        "effectiveness": "medium"
                    }
                ])
            
            if "capacity_constraint" in bottleneck_type:
                # Strategies for capacity constrained bottlenecks
                capacity_util = bottleneck.get("capacity_utilization", 0)
                
                mitigation_strategies.extend([
                    {
                        "strategy": "capacity_planning",
                        "description": f"Work with {bottleneck['name']} on capacity planning to reserve capacity",
                        "implementation_time": "medium",
                        "cost": "low",
                        "effectiveness": "medium"
                    },
                    {
                        "strategy": "order_smoothing",
                        "description": "Implement order smoothing to avoid demand spikes",
                        "implementation_time": "short",
                        "cost": "low",
                        "effectiveness": "medium"
                    }
                ])
                
                # If severely constrained, add more strategies
                if capacity_util > 90:
                    mitigation_strategies.append({
                        "strategy": "supplier_development",
                        "description": f"Partner with {bottleneck['name']} to increase their capacity",
                        "implementation_time": "long",
                        "cost": "high",
                        "effectiveness": "high"
                    })
            
            if "critical_material" in bottleneck_type:
                # Strategies for critical material bottlenecks
                critical_materials = bottleneck.get("critical_materials", [])
                material_names = [m.get("name", "Unknown") for m in critical_materials]
                
                material_description = ", ".join(material_names[:3])
                if len(material_names) > 3:
                    material_description += f", and {len(material_names) - 3} more"
                
                mitigation_strategies.extend([
                    {
                        "strategy": "material_substitution",
                        "description": f"Research alternative materials to replace {material_description}",
                        "implementation_time": "long",
                        "cost": "medium",
                        "effectiveness": "high"
                    },
                    {
                        "strategy": "strategic_inventory",
                        "description": f"Establish strategic inventory for {material_description}",
                        "implementation_time": "short",
                        "cost": "medium",
                        "effectiveness": "medium"
                    }
                ])
            
            if "network_centrality" in bottleneck_type:
                # Strategies for network centrality bottlenecks
                mitigation_strategies.extend([
                    {
                        "strategy": "network_redesign",
                        "description": "Redesign supply network to reduce dependence on this supplier",
                        "implementation_time": "long",
                        "cost": "high",
                        "effectiveness": "high"
                    },
                    {
                        "strategy": "risk_monitoring",
                        "description": f"Implement enhanced risk monitoring for {bottleneck['name']}",
                        "implementation_time": "short",
                        "cost": "low",
                        "effectiveness": "medium"
                    }
                ])
            
            if "downstream_dependence" in bottleneck_type:
                # Strategies for downstream dependence bottlenecks
                mitigation_strategies.extend([
                    {
                        "strategy": "supplier_collaboration",
                        "description": f"Establish strategic collaboration with {bottleneck['name']}",
                        "implementation_time": "medium",
                        "cost": "medium",
                        "effectiveness": "high"
                    },
                    {
                        "strategy": "visibility_improvement",
                        "description": "Improve visibility into sub-tier supply chain",
                        "implementation_time": "medium",
                        "cost": "medium",
                        "effectiveness": "medium"
                    }
                ])
            
            # Add generic strategies if none specific
            if not mitigation_strategies:
                mitigation_strategies = [
                    {
                        "strategy": "risk_assessment",
                        "description": f"Conduct detailed risk assessment for {bottleneck['name']}",
                        "implementation_time": "short",
                        "cost": "low",
                        "effectiveness": "medium"
                    },
                    {
                        "strategy": "contingency_planning",
                        "description": "Develop contingency plans for disruption",
                        "implementation_time": "medium",
                        "cost": "low",
                        "effectiveness": "medium"
                    }
                ]
            
            # Calculate implementation priority
            for strategy in mitigation_strategies:
                # Map text values to numeric
                time_score = {"short": 3, "medium": 2, "long": 1}.get(strategy["implementation_time"], 2)
                cost_score = {"low": 3, "medium": 2, "high": 1}.get(strategy["cost"], 2)
                effect_score = {"high": 3, "medium": 2, "low": 1}.get(strategy["effectiveness"], 2)
                
                # Calculate priority score (higher is better)
                priority_score = (time_score * 0.3) + (cost_score * 0.3) + (effect_score * 0.4)
                
                # Map to priority label
                if priority_score >= 2.5:
                    strategy["priority"] = "high"
                elif priority_score >= 2.0:
                    strategy["priority"] = "medium"
                else:
                    strategy["priority"] = "low"
            
            # Sort strategies by priority (high to low)
            priority_order = {"high": 0, "medium": 1, "low": 2}
            mitigation_strategies.sort(key=lambda s: priority_order.get(s.get("priority"), 3))
            
            # Add impact assessment for each strategy
            for strategy in mitigation_strategies:
                if strategy["strategy"] == "alternative_supplier":
                    strategy["impact_assessment"] = "Reduces single-source risk significantly but requires time for supplier qualification"
                elif strategy["strategy"] == "inventory_buffer":
                    strategy["impact_assessment"] = "Provides immediate protection against disruptions but increases carrying costs"
                elif strategy["strategy"] == "capacity_planning":
                    strategy["impact_assessment"] = "Ensures dedicated capacity but may require premium pricing"
                elif strategy["strategy"] == "order_smoothing":
                    strategy["impact_assessment"] = "Reduces demand variability but requires internal planning coordination"
                elif strategy["strategy"] == "supplier_development":
                    strategy["impact_assessment"] = "Long-term capacity increase but requires significant investment"
                elif strategy["strategy"] == "material_substitution":
                    strategy["impact_assessment"] = "Eliminates material dependency but requires redesign and qualification"
                elif strategy["strategy"] == "strategic_inventory":
                    strategy["impact_assessment"] = "Immediate protection for critical materials but increases inventory costs"
                elif strategy["strategy"] == "network_redesign":
                    strategy["impact_assessment"] = "Fundamentally reduces network vulnerability but requires significant change"
                elif strategy["strategy"] == "risk_monitoring":
                    strategy["impact_assessment"] = "Provides early warning of issues but doesn't prevent disruption"
                elif strategy["strategy"] == "supplier_collaboration":
                    strategy["impact_assessment"] = "Improves transparency and coordination but requires relationship investment"
                elif strategy["strategy"] == "visibility_improvement":
                    strategy["impact_assessment"] = "Better sub-tier insights but may be limited by supplier cooperation"
                else:
                    strategy["impact_assessment"] = "Improves preparedness for potential disruptions"
            
            # Build response
            response = {
                "bottleneck_id": bottleneck_id,
                "bottleneck_name": bottleneck.get("name", "Unknown"),
                "bottleneck_type": bottleneck_type,
                "severity": bottleneck.get("severity", "low"),
                "tier": tier,
                "mitigation_strategies": mitigation_strategies,
                "recommendation": mitigation_strategies[0] if mitigation_strategies else None
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing bottleneck mitigation: {str(e)}")
            raise
    
    async def generate_bottleneck_report(
        self,
        include_mitigation: bool = True,
        include_critical_paths: bool = True,
        format_type: str = "detailed"  # "detailed", "summary", "executive"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report on bottlenecks.
        
        Args:
            include_mitigation: Whether to include mitigation strategies
            include_critical_paths: Whether to include critical paths
            format_type: Type of report format
            
        Returns:
            Dictionary with bottleneck report
        """
        try:
            # Ensure bottlenecks are identified
            if not self.bottlenecks:
                await self.identify_bottlenecks()
            
            # Ensure critical paths are identified if requested
            if include_critical_paths and not self.critical_paths:
                await self.identify_critical_paths()
            
            # Initialize report sections
            summary = {
                "total_bottlenecks": len(self.bottlenecks),
                "high_severity": len([b for b in self.bottlenecks if b.get("severity") == "high"]),
                "medium_severity": len([b for b in self.bottlenecks if b.get("severity") == "medium"]),
                "low_severity": len([b for b in self.bottlenecks if b.get("severity") == "low"]),
                "bottleneck_types": {}
            }
            
            # Count bottleneck types
            for bottleneck in self.bottlenecks:
                for btype in bottleneck.get("bottleneck_type", []):
                    if btype not in summary["bottleneck_types"]:
                        summary["bottleneck_types"][btype] = 0
                    summary["bottleneck_types"][btype] += 1
            
            # Prepare bottlenecks for report
            bottlenecks_data = []
            
            for bottleneck in self.bottlenecks:
                bottleneck_data = {
                    "id": bottleneck.get("id"),
                    "name": bottleneck.get("name"),
                    "tier": bottleneck.get("tier"),
                    "severity": bottleneck.get("severity"),
                    "bottleneck_type": bottleneck.get("bottleneck_type"),
                    "impact_score": bottleneck.get("impact_score"),
                    "centrality": bottleneck.get("centrality"),
                    "downstream_impact": bottleneck.get("downstream_impact")
                }
                
                # Add capacity data if available
                if "capacity_utilization" in bottleneck:
                    bottleneck_data["capacity_utilization"] = bottleneck.get("capacity_utilization")
                
                # Add critical material data if available
                if "critical_materials" in bottleneck:
                    bottleneck_data["critical_materials"] = bottleneck.get("critical_materials")
                
                # Add mitigation strategies if requested
                if include_mitigation:
                    try:
                        mitigation = await self.analyze_bottleneck_mitigation(bottleneck.get("id"))
                        bottleneck_data["mitigation_strategies"] = mitigation.get("mitigation_strategies")
                    except Exception as me:
                        logger.error(f"Error getting mitigation for bottleneck {bottleneck.get('id')}: {str(me)}")
                        bottleneck_data["mitigation_strategies"] = []
                
                bottlenecks_data.append(bottleneck_data)
            
            # Prepare critical paths for report if requested
            critical_paths_data = []
            
            if include_critical_paths:
                for path in self.critical_paths:
                    path_data = {
                        "product_name": path.get("product_name"),
                        "material_name": path.get("material_name"),
                        "bottleneck_name": path.get("bottleneck_name"),
                        "bottleneck_tier": path.get("bottleneck_tier"),
                        "severity": path.get("severity"),
                        "path_length": len(path.get("path", [])),
                        "bottleneck_types": path.get("bottleneck_types")
                    }
                    
                    # Include full path details for detailed format
                    if format_type == "detailed":
                        path_data["path"] = path.get("path")
                    
                    critical_paths_data.append(path_data)
            
            # Build report based on format
            if format_type == "summary":
                # Summary format - keep it concise
                bottlenecks_data = bottlenecks_data[:10]  # Top 10 bottlenecks
                
                for bottleneck in bottlenecks_data:
                    # Simplify mitigation strategies
                    if "mitigation_strategies" in bottleneck:
                        bottleneck["mitigation_strategies"] = [
                            {
                                "strategy": s.get("strategy"),
                                "description": s.get("description"),
                                "priority": s.get("priority")
                            }
                            for s in bottleneck.get("mitigation_strategies", [])[:2]  # Top 2 strategies
                        ]
                
                critical_paths_data = critical_paths_data[:5]  # Top 5 critical paths
            
            elif format_type == "executive":
                # Executive format - very concise with focus on high severity
                bottlenecks_data = [b for b in bottlenecks_data if b.get("severity") == "high"]
                
                for bottleneck in bottlenecks_data:
                    # Only include top strategy
                    if "mitigation_strategies" in bottleneck:
                        bottleneck["mitigation_strategies"] = [
                            {
                                "strategy": s.get("strategy"),
                                "description": s.get("description"),
                                "priority": s.get("priority")
                            }
                            for s in bottleneck.get("mitigation_strategies", [])[:1]  # Top strategy only
                        ]
                
                critical_paths_data = [p for p in critical_paths_data if p.get("severity") == "high"]
                critical_paths_data = critical_paths_data[:3]  # Top 3 high severity paths
            
            # Build report
            report = {
                "summary": summary,
                "bottlenecks": bottlenecks_data,
                "critical_paths": critical_paths_data,
                "report_type": format_type,
                "generated_at": datetime.now().isoformat()
            }
            
            # Add recommendations section
            report["recommendations"] = self._generate_report_recommendations(
                bottlenecks_data,
                critical_paths_data
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating bottleneck report: {str(e)}")
            raise
    
    def _generate_report_recommendations(
        self,
        bottlenecks: List[Dict[str, Any]],
        critical_paths: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate high-level recommendations based on bottleneck analysis.
        
        Args:
            bottlenecks: List of bottleneck data
            critical_paths: List of critical path data
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Analyze bottleneck types
        bottleneck_types = {}
        for bottleneck in bottlenecks:
            for btype in bottleneck.get("bottleneck_type", []):
                if btype not in bottleneck_types:
                    bottleneck_types[btype] = []
                bottleneck_types[btype].append(bottleneck)
        
        # Add recommendations based on bottleneck types
        if "single_source" in bottleneck_types and len(bottleneck_types["single_source"]) >= 3:
            recommendations.append({
                "category": "sourcing_strategy",
                "recommendation": "Implement a formal dual-sourcing strategy for critical components",
                "justification": f"Identified {len(bottleneck_types['single_source'])} single-source bottlenecks in the supply chain",
                "priority": "high"
            })
        
        if "capacity_constraint" in bottleneck_types:
            recommendations.append({
                "category": "capacity_management",
                "recommendation": "Develop a capacity management program with key suppliers",
                "justification": f"Identified {len(bottleneck_types['capacity_constraint'])} suppliers with capacity constraints",
                "priority": "medium"
            })
        
        if "critical_material" in bottleneck_types:
            recommendations.append({
                "category": "material_strategy",
                "recommendation": "Create a critical materials management program",
                "justification": f"Identified {len(bottleneck_types['critical_material'])} suppliers providing critical materials",
                "priority": "high"
            })
        
        # Analyze tier distribution
        tier_distribution = {}
        for bottleneck in bottlenecks:
            tier = bottleneck.get("tier", 0)
            if tier not in tier_distribution:
                tier_distribution[tier] = []
            tier_distribution[tier].append(bottleneck)
        
        # Add recommendations based on tier distribution
        sub_tier_count = sum(len(suppliers) for tier, suppliers in tier_distribution.items() if tier > 1)
        if sub_tier_count > 0:
            recommendations.append({
                "category": "visibility",
                "recommendation": "Implement sub-tier supplier visibility program",
                "justification": f"Identified {sub_tier_count} bottlenecks in tier 2 and beyond",
                "priority": "medium"
            })
        
        # Add general recommendations based on bottleneck count
        high_severity_count = len([b for b in bottlenecks if b.get("severity") == "high"])
        if high_severity_count > 0:
            recommendations.append({
                "category": "risk_management",
                "recommendation": "Establish a formal supply chain risk management program",
                "justification": f"Identified {high_severity_count} high-severity bottlenecks",
                "priority": "high"
            })
        
        # Add technology recommendation if appropriate
        if len(bottlenecks) > 10:
            recommendations.append({
                "category": "technology",
                "recommendation": "Implement supply chain risk monitoring technology",
                "justification": f"Complexity of supply chain with {len(bottlenecks)} identified bottlenecks requires systematic monitoring",
                "priority": "medium"
            })
        
        # Add collaboration recommendation based on critical paths
        if critical_paths:
            product_count = len(set(p.get("product_name") for p in critical_paths))
            recommendations.append({
                "category": "collaboration",
                "recommendation": "Establish cross-functional supply chain risk task force",
                "justification": f"Critical paths affect {product_count} key products across multiple tiers",
                "priority": "high" if product_count > 3 else "medium"
            })
        
        # Sort recommendations by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.get("priority"), 3))
        
        return recommendations