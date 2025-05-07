"""
Supplier network builder module for multi-tier supply chain analysis.

This module provides functionality to construct supplier networks from 
client data, connecting suppliers across multiple tiers.
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

# Initialize logger
logger = get_logger(__name__)

class SupplierNetworkBuilder:
    """
    Builds supplier network graphs from client data.
    
    This class analyzes supplier relationships in client data to construct
    a directed graph representing the multi-tier supply chain network.
    """
    
    def __init__(
        self,
        client_id: str,
        connection_id: Optional[str] = None,
    ):
        """
        Initialize the supplier network builder.
        
        Args:
            client_id: Client identifier
            connection_id: Optional database connection identifier
        """
        self.client_id = client_id
        self.connection_id = connection_id
        self.graph = nx.DiGraph()
        self.supplier_data = {}
        self.tier_suppliers = {}
        self.db_connector = None
        self.schema = None
        self.domain_mappings = None
        
    async def initialize(self) -> None:
        """
        Initialize the network builder with client schema and connections.
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
            
            logger.info(f"Initialized supplier network builder for client: {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error initializing supplier network: {str(e)}")
            raise
    
    async def build_network(
        self,
        include_tier3_plus: bool = False,
        cutoff_date: Optional[datetime] = None,
        supplier_ids: Optional[List[str]] = None,
        product_ids: Optional[List[str]] = None,
        material_ids: Optional[List[str]] = None,
    ) -> nx.DiGraph:
        """
        Build the supplier network from client data.
        
        Args:
            include_tier3_plus: Whether to include tier 3+ suppliers
            cutoff_date: Optional date cutoff for historical data
            supplier_ids: Optional list of specific supplier IDs to include
            product_ids: Optional list of product IDs to filter by
            material_ids: Optional list of material IDs to filter by
            
        Returns:
            NetworkX directed graph representing the supplier network
        """
        if not self.db_connector or not self.schema:
            await self.initialize()
        
        try:
            # Reset graph
            self.graph = nx.DiGraph()
            
            # Load tier 1 suppliers (direct suppliers)
            await self._load_tier1_suppliers(
                cutoff_date=cutoff_date,
                supplier_ids=supplier_ids,
                product_ids=product_ids
            )
            
            # Load tier 2 suppliers (suppliers of direct suppliers)
            await self._load_tier2_suppliers(
                cutoff_date=cutoff_date,
                material_ids=material_ids
            )
            
            # Load tier 3+ suppliers if requested
            if include_tier3_plus:
                await self._load_tier3plus_suppliers(cutoff_date=cutoff_date)
            
            # Calculate network metrics
            self._calculate_network_metrics()
            
            logger.info(f"Built supplier network with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            
            return self.graph
            
        except Exception as e:
            logger.error(f"Error building supplier network: {str(e)}")
            raise
    
    async def _load_tier1_suppliers(
        self,
        cutoff_date: Optional[datetime] = None,
        supplier_ids: Optional[List[str]] = None,
        product_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Load tier 1 (direct) suppliers.
        
        Args:
            cutoff_date: Optional date cutoff for historical data
            supplier_ids: Optional list of specific supplier IDs to include
            product_ids: Optional list of product IDs to filter by
        """
        try:
            # Find supplier table using domain mappings
            supplier_table = self._find_table_by_domain("supplier")
            if not supplier_table:
                logger.warning("Could not find supplier table in schema")
                return
            
            # Find supplier relationship tables
            purchase_order_table = self._find_table_by_domain("purchase_order")
            product_table = self._find_table_by_domain("product")
            
            # Build query
            query = f"""
            SELECT 
                s.*, 
                COUNT(DISTINCT po.id) AS order_count,
                SUM(po.total_amount) AS total_spend
            FROM 
                {supplier_table} s
            LEFT JOIN 
                {purchase_order_table} po ON s.id = po.supplier_id
            """
            
            # Add filters
            where_clauses = []
            params = {}
            
            if supplier_ids:
                where_clauses.append("s.id IN :supplier_ids")
                params["supplier_ids"] = tuple(supplier_ids)
                
            if product_ids and product_table:
                query += f" LEFT JOIN order_items oi ON po.id = oi.order_id "
                where_clauses.append("oi.product_id IN :product_ids")
                params["product_ids"] = tuple(product_ids)
                
            if cutoff_date:
                where_clauses.append("po.order_date >= :cutoff_date")
                params["cutoff_date"] = cutoff_date
                
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
                
            query += """
            GROUP BY s.id
            ORDER BY total_spend DESC
            """
            
            # Execute query
            result = await self.db_connector.execute_query(query, params)
            suppliers = result.get("data", [])
            
            # Process suppliers
            for supplier in suppliers:
                supplier_id = str(supplier.get("id"))
                self.graph.add_node(
                    supplier_id,
                    tier=1,
                    name=supplier.get("name"),
                    type="supplier",
                    data=supplier
                )
                
                # Add edge from company to supplier
                self.graph.add_edge(
                    "company",
                    supplier_id,
                    type="buys_from",
                    spend=supplier.get("total_spend", 0),
                    order_count=supplier.get("order_count", 0)
                )
                
                # Store supplier data
                self.supplier_data[supplier_id] = supplier
                
            # Store tier 1 suppliers
            self.tier_suppliers[1] = {s.get("id") for s in suppliers}
            
            # Add the company as the root node
            self.graph.add_node(
                "company",
                tier=0,
                name="Your Company",
                type="company"
            )
            
            logger.info(f"Loaded {len(suppliers)} tier 1 suppliers")
            
        except Exception as e:
            logger.error(f"Error loading tier 1 suppliers: {str(e)}")
            raise
    
    async def _load_tier2_suppliers(
        self,
        cutoff_date: Optional[datetime] = None,
        material_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Load tier 2 suppliers (suppliers of direct suppliers).
        
        Args:
            cutoff_date: Optional date cutoff for historical data
            material_ids: Optional list of material IDs to filter by
        """
        try:
            # We need supplier-supplier relationships
            # This could be through supplier disclosure data, material sourcing data, etc.
            
            # Find relevant tables using domain mappings
            supplier_table = self._find_table_by_domain("supplier")
            material_supplier_table = self._find_table_by_domain("material_supplier")
            material_table = self._find_table_by_domain("material")
            
            if not supplier_table or not material_supplier_table:
                logger.warning("Could not find required tables for tier 2 suppliers")
                return
            
            # Get tier 1 supplier IDs
            tier1_ids = self.tier_suppliers.get(1, set())
            if not tier1_ids:
                logger.warning("No tier 1 suppliers to find tier 2 suppliers for")
                return
            
            # Build query to find tier 2 suppliers
            query = f"""
            SELECT 
                s.*, 
                ms.material_id,
                ms.tier1_supplier_id,
                m.name AS material_name,
                COUNT(DISTINCT ms.material_id) AS material_count
            FROM 
                {supplier_table} s
            JOIN 
                {material_supplier_table} ms ON s.id = ms.supplier_id
            LEFT JOIN 
                {material_table} m ON ms.material_id = m.id
            WHERE 
                ms.tier1_supplier_id IN :tier1_ids
            """
            
            # Add filters
            params = {"tier1_ids": tuple(tier1_ids)}
            
            if material_ids:
                query += " AND ms.material_id IN :material_ids"
                params["material_ids"] = tuple(material_ids)
                
            if cutoff_date:
                query += " AND ms.last_updated >= :cutoff_date"
                params["cutoff_date"] = cutoff_date
                
            query += """
            GROUP BY s.id, ms.material_id, ms.tier1_supplier_id, m.name
            """
            
            # Execute query
            result = await self.db_connector.execute_query(query, params)
            tier2_suppliers = result.get("data", [])
            
            # Process tier 2 suppliers
            tier2_ids = set()
            
            for supplier in tier2_suppliers:
                supplier_id = str(supplier.get("id"))
                tier1_id = str(supplier.get("tier1_supplier_id"))
                material_id = str(supplier.get("material_id"))
                
                # Add node if not already added
                if not self.graph.has_node(supplier_id):
                    self.graph.add_node(
                        supplier_id,
                        tier=2,
                        name=supplier.get("name"),
                        type="supplier",
                        data=supplier
                    )
                    
                    # Store supplier data
                    self.supplier_data[supplier_id] = supplier
                    tier2_ids.add(supplier_id)
                
                # Add edge from tier 1 supplier to tier 2 supplier
                edge_key = (tier1_id, supplier_id, material_id)
                if not self.graph.has_edge(tier1_id, supplier_id):
                    self.graph.add_edge(
                        tier1_id,
                        supplier_id,
                        type="sources_from",
                        material_id=material_id,
                        material_name=supplier.get("material_name")
                    )
            
            # Store tier 2 suppliers
            self.tier_suppliers[2] = tier2_ids
            
            logger.info(f"Loaded {len(tier2_ids)} tier 2 suppliers")
            
        except Exception as e:
            logger.error(f"Error loading tier 2 suppliers: {str(e)}")
            raise
    
    async def _load_tier3plus_suppliers(
        self,
        cutoff_date: Optional[datetime] = None,
        max_tier: int = 5
    ) -> None:
        """
        Load tier 3+ suppliers recursively.
        
        Args:
            cutoff_date: Optional date cutoff for historical data
            max_tier: Maximum tier depth to analyze
        """
        try:
            # Find relevant tables using domain mappings
            supplier_table = self._find_table_by_domain("supplier")
            material_supplier_table = self._find_table_by_domain("material_supplier")
            material_table = self._find_table_by_domain("material")
            
            if not supplier_table or not material_supplier_table:
                logger.warning("Could not find required tables for tier 3+ suppliers")
                return
            
            current_tier = 2
            while current_tier < max_tier:
                # Get current tier supplier IDs
                current_tier_ids = self.tier_suppliers.get(current_tier, set())
                if not current_tier_ids:
                    logger.info(f"No tier {current_tier} suppliers to find tier {current_tier+1} suppliers for")
                    break
                
                # Build query to find next tier suppliers
                query = f"""
                SELECT 
                    s.*, 
                    ms.material_id,
                    ms.tier1_supplier_id,
                    m.name AS material_name
                FROM 
                    {supplier_table} s
                JOIN 
                    {material_supplier_table} ms ON s.id = ms.supplier_id
                LEFT JOIN 
                    {material_table} m ON ms.material_id = m.id
                WHERE 
                    ms.tier1_supplier_id IN :current_tier_ids
                """
                
                # Add filters
                params = {"current_tier_ids": tuple(current_tier_ids)}
                
                if cutoff_date:
                    query += " AND ms.last_updated >= :cutoff_date"
                    params["cutoff_date"] = cutoff_date
                
                # Execute query
                result = await self.db_connector.execute_query(query, params)
                next_tier_suppliers = result.get("data", [])
                
                # Process next tier suppliers
                next_tier_ids = set()
                
                for supplier in next_tier_suppliers:
                    supplier_id = str(supplier.get("id"))
                    source_id = str(supplier.get("tier1_supplier_id"))
                    material_id = str(supplier.get("material_id"))
                    
                    # Skip if already in a lower tier (avoid cycles)
                    if any(supplier_id in self.tier_suppliers.get(t, set()) for t in range(1, current_tier+1)):
                        continue
                    
                    # Add node if not already added
                    if not self.graph.has_node(supplier_id):
                        self.graph.add_node(
                            supplier_id,
                            tier=current_tier + 1,
                            name=supplier.get("name"),
                            type="supplier",
                            data=supplier
                        )
                        
                        # Store supplier data
                        self.supplier_data[supplier_id] = supplier
                        next_tier_ids.add(supplier_id)
                    
                    # Add edge from source supplier to this supplier
                    if not self.graph.has_edge(source_id, supplier_id):
                        self.graph.add_edge(
                            source_id,
                            supplier_id,
                            type="sources_from",
                            material_id=material_id,
                            material_name=supplier.get("material_name")
                        )
                
                # Store next tier suppliers
                self.tier_suppliers[current_tier + 1] = next_tier_ids
                
                logger.info(f"Loaded {len(next_tier_ids)} tier {current_tier+1} suppliers")
                
                # If no next tier suppliers were found, break
                if not next_tier_ids:
                    break
                
                # Move to next tier
                current_tier += 1
            
        except Exception as e:
            logger.error(f"Error loading tier 3+ suppliers: {str(e)}")
            raise
    
    def _calculate_network_metrics(self) -> None:
        """
        Calculate network metrics for supplier analysis.
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
                    downstream_nodes = nx.descendants(self.graph, node)
                    downstream_count = len(downstream_nodes)
                    self.graph.nodes[node]['downstream_impact'] = downstream_count
                    
                    # Risk propagation potential
                    if downstream_count > 0:
                        risk_propagation = betweenness_centrality.get(node, 0) * downstream_count
                    else:
                        risk_propagation = 0
                    self.graph.nodes[node]['risk_propagation'] = risk_propagation
            
            # Calculate graph-level metrics
            try:
                avg_path_length = nx.average_shortest_path_length(self.graph)
                self.graph.graph['avg_path_length'] = avg_path_length
            except nx.NetworkXError:
                # Graph is not strongly connected
                self.graph.graph['avg_path_length'] = None
            
            self.graph.graph['node_count'] = len(self.graph.nodes)
            self.graph.graph['edge_count'] = len(self.graph.edges)
            self.graph.graph['density'] = nx.density(self.graph)
            
            # Calculate tier statistics
            tier_counts = {}
            for node, data in self.graph.nodes(data=True):
                tier = data.get('tier')
                if tier is not None:
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            self.graph.graph['tier_counts'] = tier_counts
            
        except Exception as e:
            logger.error(f"Error calculating network metrics: {str(e)}")
            # Continue with partial metrics
    
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
    
    def get_network_data(self) -> Dict[str, Any]:
        """
        Get the network data in a JSON-serializable format.
        
        Returns:
            Dictionary with nodes, edges and graph metrics
        """
        # Convert NetworkX graph to JSON-serializable format
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            # Make a copy to avoid modifying the original
            node_data = dict(data)
            
            # Remove non-serializable elements
            if 'data' in node_data:
                # Ensure data is serializable
                serializable_data = {}
                for k, v in node_data['data'].items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        serializable_data[k] = v
                node_data['data'] = serializable_data
            
            nodes.append({
                'id': node_id,
                **node_data
            })
        
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                **data
            })
        
        # Include graph metrics
        metrics = {}
        for key, value in self.graph.graph.items():
            if isinstance(value, (str, int, float, bool, type(None), dict, list)):
                metrics[key] = value
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metrics': metrics
        }
    
    async def get_supplier_details(self, supplier_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific supplier.
        
        Args:
            supplier_id: Supplier ID to get details for
            
        Returns:
            Dictionary with supplier details
        """
        if not self.graph.has_node(supplier_id):
            return {}
        
        # Get base supplier data
        details = dict(self.supplier_data.get(supplier_id, {}))
        
        # Add network metrics
        node_data = self.graph.nodes[supplier_id]
        for key, value in node_data.items():
            if key != 'data':
                details[key] = value
        
        # Get upstream and downstream relationships
        details['upstream_suppliers'] = []
        for source, _, data in self.graph.in_edges(supplier_id, data=True):
            if source != "company":  # Skip the company node
                details['upstream_suppliers'].append({
                    'id': source,
                    'name': self.graph.nodes[source].get('name', ''),
                    **data
                })
        
        details['downstream_suppliers'] = []
        for _, target, data in self.graph.out_edges(supplier_id, data=True):
            details['downstream_suppliers'].append({
                'id': target,
                'name': self.graph.nodes[target].get('name', ''),
                **data
            })
        
        # Add additional metrics based on network position
        details['tier'] = node_data.get('tier')
        details['centrality'] = node_data.get('betweenness_centrality', 0)
        details['risk_propagation_score'] = node_data.get('risk_propagation', 0)
        
        return details