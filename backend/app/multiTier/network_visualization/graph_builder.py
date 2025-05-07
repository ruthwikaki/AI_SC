"""
Network graph builder module for multi-tier supply chain visualization.

This module provides functionality to build visualizable network graphs
from supplier network data for multi-tier supply chain analysis.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx
import json
import math
import logging
from datetime import datetime

from app.multiTier.supplier_mapping.network_builder import SupplierNetworkBuilder
from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class SupplyChainGraphBuilder:
    """
    Builds visualizable supply chain network graphs.
    
    This class transforms supply chain network data into formats suitable
    for different visualization libraries and approaches.
    """
    
    def __init__(
        self,
        network_builder: SupplierNetworkBuilder
    ):
        """
        Initialize the supply chain graph builder.
        
        Args:
            network_builder: Network builder instance with supplier network data
        """
        self.network_builder = network_builder
        self.graph = None
        self.client_id = network_builder.client_id
        
    async def initialize_graph(
        self,
        include_tier3_plus: bool = False,
        recalculate_network: bool = False
    ) -> None:
        """
        Initialize or update the network graph.
        
        Args:
            include_tier3_plus: Whether to include tier 3+ suppliers
            recalculate_network: Whether to rebuild the network even if it exists
        """
        try:
            if not self.graph or recalculate_network:
                self.graph = await self.network_builder.build_network(
                    include_tier3_plus=include_tier3_plus
                )
            
            logger.info(f"Initialized graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            
        except Exception as e:
            logger.error(f"Error initializing graph: {str(e)}")
            raise
    
    def build_d3_force_layout(
        self,
        highlight_nodes: Optional[List[str]] = None,
        highlight_edges: Optional[List[Tuple[str, str]]] = None,
        color_by: str = "tier",  # "tier", "risk", "category", etc.
        size_by: str = "impact",  # "impact", "centrality", "spend", etc.
        include_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Build a graph representation suitable for D3.js force-directed layout.
        
        Args:
            highlight_nodes: Optional list of node IDs to highlight
            highlight_edges: Optional list of edge (source, target) tuples to highlight
            color_by: Node attribute to use for coloring
            size_by: Node attribute to use for sizing
            include_metrics: Whether to include graph metrics
            
        Returns:
            Dictionary with nodes and links for D3 visualization
        """
        try:
            if not self.graph:
                raise ValueError("Graph not initialized. Call initialize_graph() first.")
            
            # Prepare highlighted elements sets for quicker lookups
            highlighted_nodes = set(highlight_nodes or [])
            highlighted_edges = set(tuple(edge) for edge in (highlight_edges or []))
            
            # Prepare nodes
            nodes = []
            for node_id, data in self.graph.nodes(data=True):
                # Basic node data
                node = {
                    "id": node_id,
                    "name": data.get("name", "Unknown"),
                    "tier": data.get("tier", 0),
                    "type": data.get("type", "supplier"),
                    "highlighted": node_id in highlighted_nodes
                }
                
                # Add color attribute based on color_by parameter
                if color_by == "tier":
                    node["color"] = self._get_tier_color(data.get("tier", 0))
                elif color_by == "risk" and "risk_composite" in data:
                    node["color"] = self._get_risk_color(data.get("risk_composite", 0))
                elif color_by == "category" and "category" in data:
                    node["color"] = self._get_category_color(data.get("category", "unknown"))
                else:
                    # Default color by tier
                    node["color"] = self._get_tier_color(data.get("tier", 0))
                
                # Add size attribute based on size_by parameter
                if size_by == "impact" and "downstream_impact" in data:
                    # Map downstream impact to size (with minimum size)
                    impact = data.get("downstream_impact", 0)
                    node["size"] = max(5, 5 + (impact * 2))
                elif size_by == "centrality" and "betweenness_centrality" in data:
                    # Map centrality to size (with minimum size)
                    centrality = data.get("betweenness_centrality", 0)
                    node["size"] = max(5, 5 + (centrality * 100))
                elif size_by == "spend" and "total_spend" in data:
                    # Map spend to size (logarithmic scale with minimum size)
                    spend = data.get("total_spend", 0)
                    if spend > 0:
                        node["size"] = max(5, 5 + math.log10(spend + 1))
                    else:
                        node["size"] = 5
                else:
                    # Default size based on tier (inverse for visibility - tier 1 is bigger)
                    tier = data.get("tier", 0)
                    if tier == 0:  # Company node
                        node["size"] = 15
                    else:
                        node["size"] = max(5, 15 - (tier * 2))
                
                # Add metrics if requested
                if include_metrics:
                    metrics = {}
                    
                    # Include network metrics
                    for metric in ["betweenness_centrality", "downstream_impact"]:
                        if metric in data:
                            metrics[metric] = data[metric]
                    
                    # Include risk metrics
                    for key, value in data.items():
                        if key.startswith("risk_"):
                            metrics[key] = value
                    
                    # Include other useful attributes
                    for key in ["category", "country", "order_count", "total_spend"]:
                        if key in data:
                            metrics[key] = data[key]
                    
                    node["metrics"] = metrics
                
                nodes.append(node)
            
            # Prepare links
            links = []
            for source, target, data in self.graph.edges(data=True):
                # Basic link data
                link = {
                    "source": source,
                    "target": target,
                    "type": data.get("type", "default"),
                    "highlighted": (source, target) in highlighted_edges
                }
                
                # Add weight/strength
                if "spend" in data:
                    # Normalize spend for link strength
                    spend = data.get("spend", 0)
                    link["value"] = max(1, math.log10(spend + 1))
                else:
                    link["value"] = 1
                
                # Add metrics if requested
                if include_metrics:
                    metrics = {}
                    
                    # Include useful edge attributes
                    for key in ["material_id", "material_name", "order_count", "spend"]:
                        if key in data:
                            metrics[key] = data[key]
                    
                    # Include risk metrics
                    for key, value in data.items():
                        if key.startswith("risk_"):
                            metrics[key] = value
                    
                    link["metrics"] = metrics
                
                links.append(link)
            
            # Build result
            result = {
                "nodes": nodes,
                "links": links
            }
            
            # Add graph-level metrics if requested
            if include_metrics:
                metrics = {}
                
                # Include graph metrics if available
                for key, value in self.graph.graph.items():
                    metrics[key] = value
                
                result["metrics"] = metrics
            
            logger.info(f"Built D3 force layout with {len(nodes)} nodes and {len(links)} links")
            
            return result
            
        except Exception as e:
            logger.error(f"Error building D3 force layout: {str(e)}")
            raise
    
    def build_sankey_diagram(
        self,
        group_by: str = "tier",  # "tier", "category", "country", etc.
        flow_by: str = "spend",  # "spend", "count", etc.
        limit_nodes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Build a Sankey diagram representation of the supply chain.
        
        Args:
            group_by: Node attribute to group by
            flow_by: Edge attribute to determine flow value
            limit_nodes: Optional limit on number of nodes to include
            
        Returns:
            Dictionary with nodes and links for Sankey visualization
        """
        try:
            if not self.graph:
                raise ValueError("Graph not initialized. Call initialize_graph() first.")
            
            # Group nodes
            node_groups = {}
            
            if group_by == "tier":
                # Create tier groups
                tiers = set()
                for _, data in self.graph.nodes(data=True):
                    tiers.add(data.get("tier", 0))
                
                for tier in sorted(tiers):
                    tier_name = f"Tier {tier}" if tier > 0 else "Company"
                    node_groups[tier_name] = {
                        "nodes": [],
                        "attribute": tier
                    }
                
                # Assign nodes to groups
                for node_id, data in self.graph.nodes(data=True):
                    tier = data.get("tier", 0)
                    tier_name = f"Tier {tier}" if tier > 0 else "Company"
                    node_groups[tier_name]["nodes"].append(node_id)
            
            elif group_by == "category" and any("category" in data for _, data in self.graph.nodes(data=True)):
                # Create category groups
                categories = set()
                for _, data in self.graph.nodes(data=True):
                    if "category" in data:
                        categories.add(data.get("category"))
                    else:
                        categories.add("Other")
                
                # Ensure "Company" category exists
                categories.add("Company")
                
                for category in sorted(categories):
                    node_groups[category] = {
                        "nodes": [],
                        "attribute": category
                    }
                
                # Assign nodes to groups
                for node_id, data in self.graph.nodes(data=True):
                    if data.get("type") == "company":
                        node_groups["Company"]["nodes"].append(node_id)
                    elif "category" in data:
                        node_groups[data["category"]]["nodes"].append(node_id)
                    else:
                        node_groups["Other"]["nodes"].append(node_id)
            
            elif group_by == "country" and any("country" in data for _, data in self.graph.nodes(data=True)):
                # Create country groups
                countries = set()
                for _, data in self.graph.nodes(data=True):
                    if "country" in data:
                        countries.add(data.get("country"))
                    else:
                        countries.add("Unknown")
                
                # Ensure "Home" country exists
                countries.add("Home")
                
                for country in sorted(countries):
                    node_groups[country] = {
                        "nodes": [],
                        "attribute": country
                    }
                
                # Assign nodes to groups
                for node_id, data in self.graph.nodes(data=True):
                    if data.get("type") == "company":
                        node_groups["Home"]["nodes"].append(node_id)
                    elif "country" in data:
                        node_groups[data["country"]]["nodes"].append(node_id)
                    else:
                        node_groups["Unknown"]["nodes"].append(node_id)
            
            else:
                # Default to tier grouping
                return self.build_sankey_diagram(group_by="tier", flow_by=flow_by, limit_nodes=limit_nodes)
            
            # Filter or merge small groups if limit_nodes is set
            if limit_nodes and len(node_groups) > limit_nodes:
                # Sort groups by size
                sorted_groups = sorted(
                    node_groups.items(),
                    key=lambda x: len(x[1]["nodes"]),
                    reverse=True
                )
                
                # Keep top groups, merge the rest into "Other"
                top_groups = dict(sorted_groups[:limit_nodes-1])
                
                # Create "Other" group
                other_group = {"nodes": [], "attribute": "Other"}
                for name, group in sorted_groups[limit_nodes-1:]:
                    other_group["nodes"].extend(group["nodes"])
                
                # Only add Other group if it has nodes
                if other_group["nodes"]:
                    top_groups["Other"] = other_group
                
                node_groups = top_groups
            
            # Build Sankey nodes
            sankey_nodes = []
            node_id_map = {}  # Maps original node IDs to Sankey node indices
            
            # Add nodes for each group
            for group_name, group_data in node_groups.items():
                sankey_nodes.append({
                    "name": group_name,
                    "group": group_name,
                    "nodeIds": group_data["nodes"],
                    "count": len(group_data["nodes"])
                })
            
            # Build Sankey links
            sankey_links = []
            
            # Calculate flows between groups
            group_flows = {}
            
            for source, target, data in self.graph.edges(data=True):
                # Find group for source and target
                source_group = None
                target_group = None
                
                for group_name, group_data in node_groups.items():
                    if source in group_data["nodes"]:
                        source_group = group_name
                    if target in group_data["nodes"]:
                        target_group = group_name
                
                if source_group and target_group and source_group != target_group:
                    # Create flow key
                    flow_key = (source_group, target_group)
                    
                    # Calculate flow value
                    if flow_by == "spend" and "spend" in data:
                        flow_value = data.get("spend", 0)
                    elif flow_by == "count":
                        flow_value = 1
                    else:
                        flow_value = 1
                    
                    # Add to group flows
                    if flow_key not in group_flows:
                        group_flows[flow_key] = 0
                    
                    group_flows[flow_key] += flow_value
            
            # Create Sankey links from group flows
            for (source_group, target_group), value in group_flows.items():
                # Find indices
                source_idx = next((i for i, node in enumerate(sankey_nodes) if node["name"] == source_group), None)
                target_idx = next((i for i, node in enumerate(sankey_nodes) if node["name"] == target_group), None)
                
                if source_idx is not None and target_idx is not None:
                    sankey_links.append({
                        "source": source_idx,
                        "target": target_idx,
                        "value": value,
                        "sourceGroup": source_group,
                        "targetGroup": target_group
                    })
            
            # Build result
            result = {
                "nodes": sankey_nodes,
                "links": sankey_links
            }
            
            logger.info(f"Built Sankey diagram with {len(sankey_nodes)} nodes and {len(sankey_links)} links")
            
            return result
            
        except Exception as e:
            logger.error(f"Error building Sankey diagram: {str(e)}")
            raise
    
    def build_heatmap(
        self,
        x_axis: str = "tier",  # "tier", "category", "country", etc.
        y_axis: str = "risk",  # "risk", "spend", "count", etc.
        value: str = "count",  # "count", "spend", "risk", etc.
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Build a heatmap representation of the supply chain.
        
        Args:
            x_axis: Attribute for x-axis grouping
            y_axis: Attribute for y-axis grouping
            value: Attribute to determine cell values
            normalize: Whether to normalize values
            
        Returns:
            Dictionary with heatmap data
        """
        try:
            if not self.graph:
                raise ValueError("Graph not initialized. Call initialize_graph() first.")
            
            # Get unique values for x and y axes
            x_values = set()
            y_values = set()
            
            # Define how to extract axis values from nodes
            def get_x_value(node_data):
                if x_axis == "tier":
                    tier = node_data.get("tier", 0)
                    return f"Tier {tier}" if tier > 0 else "Company"
                elif x_axis == "category":
                    return node_data.get("category", "Other")
                elif x_axis == "country":
                    return node_data.get("country", "Unknown")
                else:
                    return "Unknown"
            
            def get_y_value(node_data):
                if y_axis == "risk":
                    risk = node_data.get("risk_composite", 0)
                    if risk < 0.3:
                        return "Low Risk"
                    elif risk < 0.7:
                        return "Medium Risk"
                    else:
                        return "High Risk"
                elif y_axis == "spend":
                    spend = node_data.get("total_spend", 0)
                    if spend < 10000:
                        return "Low Spend"
                    elif spend < 100000:
                        return "Medium Spend"
                    else:
                        return "High Spend"
                else:
                    return "Unknown"
            
            # Collect unique axis values
            for _, data in self.graph.nodes(data=True):
                if data.get("type") != "company":  # Skip company node
                    x_values.add(get_x_value(data))
                    y_values.add(get_y_value(data))
            
            # Sort values for consistent display
            x_values = sorted(x_values)
            
            # Custom sort order for y-values if they are risk or spend levels
            if y_axis == "risk":
                y_values = ["Low Risk", "Medium Risk", "High Risk"]
            elif y_axis == "spend":
                y_values = ["Low Spend", "Medium Spend", "High Spend"]
            else:
                y_values = sorted(y_values)
            
            # Initialize data grid
            grid = {}
            for x in x_values:
                grid[x] = {}
                for y in y_values:
                    grid[x][y] = 0
            
            # Populate grid
            for node_id, data in self.graph.nodes(data=True):
                if data.get("type") == "company":
                    continue  # Skip company node
                
                x = get_x_value(data)
                y = get_y_value(data)
                
                if value == "count":
                    grid[x][y] += 1
                elif value == "spend":
                    grid[x][y] += data.get("total_spend", 0)
                elif value == "risk":
                    grid[x][y] += data.get("risk_composite", 0)
                else:
                    grid[x][y] += 1
            
            # Convert to heatmap format
            heatmap_data = []
            
            for x in x_values:
                for y in y_values:
                    cell_value = grid[x][y]
                    heatmap_data.append({
                        "x": x,
                        "y": y,
                        "value": cell_value
                    })
            
            # Normalize values if requested
            if normalize and heatmap_data:
                max_value = max(item["value"] for item in heatmap_data)
                
                if max_value > 0:
                    for item in heatmap_data:
                        item["normalized_value"] = item["value"] / max_value
            
            # Build result
            result = {
                "data": heatmap_data,
                "x_axis": {
                    "name": x_axis,
                    "values": list(x_values)
                },
                "y_axis": {
                    "name": y_axis,
                    "values": list(y_values)
                },
                "value_type": value
            }
            
            logger.info(f"Built heatmap with {len(heatmap_data)} cells")
            
            return result
            
        except Exception as e:
            logger.error(f"Error building heatmap: {str(e)}")
            raise
    
    def _get_tier_color(self, tier: int) -> str:
        """
        Get color for a supplier tier.
        
        Args:
            tier: Supplier tier level
            
        Returns:
            Hex color code
        """
        colors = {
            0: "#0066CC",  # Company (blue)
            1: "#66CC00",  # Tier 1 (green)
            2: "#FFCC00",  # Tier 2 (yellow)
            3: "#FF9900",  # Tier 3 (orange)
            4: "#FF6600",  # Tier 4 (dark orange)
            5: "#FF3300",  # Tier 5+ (red)
        }
        
        # Use tier 5 color for tiers beyond 5
        return colors.get(min(tier, 5), "#CCCCCC")
    
    def _get_risk_color(self, risk: float) -> str:
        """
        Get color for a risk level.
        
        Args:
            risk: Risk value (0-1)
            
        Returns:
            Hex color code
        """
        if risk < 0.3:
            return "#66CC00"  # Low risk (green)
        elif risk < 0.7:
            return "#FFCC00"  # Medium risk (yellow)
        else:
            return "#FF3300"  # High risk (red)
    
    def _get_category_color(self, category: str) -> str:
        """
        Get color for a supplier category.
        
        Args:
            category: Supplier category
            
        Returns:
            Hex color code
        """
        # Generate a color based on the category string
        # This ensures the same category always gets the same color
        colors = {
            "Raw Materials": "#8A2BE2",    # BlueViolet
            "Components": "#20B2AA",       # LightSeaGreen
            "Packaging": "#FF8C00",        # DarkOrange
            "Services": "#4682B4",         # SteelBlue
            "Logistics": "#DAA520",        # GoldenRod
            "Equipment": "#696969",        # DimGray
            "Company": "#0066CC",          # Blue
            "Other": "#CCCCCC"             # Light Gray
        }
        
        # Return color if defined, otherwise generate based on category string
        if category in colors:
            return colors[category]
        
        # Generate a color based on hash of category string
        hash_code = 0
        for char in category:
            hash_code = (hash_code * 31 + ord(char)) & 0xFFFFFF
        
        # Convert to hex color (ensure it's not too light)
        color = f"#{hash_code:06x}"
        return color
    
    async def export_graph(
        self,
        format_type: str = "d3",  # "d3", "sankey", "heatmap", "networkx"
        **kwargs
    ) -> Dict[str, Any]:
        """
        Export the graph in the specified format.
        
        Args:
            format_type: Type of format to export
            **kwargs: Additional format-specific parameters
            
        Returns:
            Formatted graph data for visualization
        """
        try:
            if not self.graph:
                await self.initialize_graph()
            
            if format_type == "d3":
                return self.build_d3_force_layout(**kwargs)
            elif format_type == "sankey":
                return self.build_sankey_diagram(**kwargs)
            elif format_type == "heatmap":
                return self.build_heatmap(**kwargs)
            elif format_type == "networkx":
                # Export as NetworkX dictionary
                data = nx.node_link_data(self.graph)
                return json.loads(json.dumps(data, default=str))
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
        
        except Exception as e:
            logger.error(f"Error exporting graph: {str(e)}")
            raise