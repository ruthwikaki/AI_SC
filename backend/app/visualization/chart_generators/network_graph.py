"""
Network graph generator module for the supply chain LLM platform.

This module provides functionality to generate network graphs from
supply chain data for various analytics use cases.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import networkx as nx
import warnings

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class NetworkGraphGenerator:
    """
    Generates network graphs for supply chain analytics.
    
    This class provides methods to create different types of network graphs,
    including supplier networks, multi-tier supply chains, and relationship
    networks from supply chain data.
    """
    
    def __init__(self):
        """Initialize the network graph generator."""
        self.default_colors = plt.cm.tab10.colors
        self.default_figsize = (12, 10)
        
    def generate_network_graph(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        source_column: str,
        target_column: str,
        title: str = "Network Graph",
        node_size_column: Optional[str] = None,
        edge_weight_column: Optional[str] = None,
        node_color_column: Optional[str] = None,
        node_label_column: Optional[str] = None,
        layout: str = "spring",
        figsize: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a network graph.
        
        Args:
            data: DataFrame or list of dicts containing the data
            source_column: Column name for source nodes
            target_column: Column name for target nodes
            title: Chart title
            node_size_column: Column name for node sizes
            edge_weight_column: Column name for edge weights
            node_color_column: Column name for node colors
            node_label_column: Column name for node labels
            layout: Layout algorithm ('spring', 'circular', 'random', 'shell', 'kamada_kawai')
            figsize: Figure size as (width, height) tuple
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing chart data and metadata
        """
        try:
            # Convert to DataFrame if list
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            # Create network graph
            G = nx.DiGraph()
            
            # Add edges from data
            for _, row in df.iterrows():
                source = row[source_column]
                target = row[target_column]
                
                # Add edge with optional weight
                if edge_weight_column and edge_weight_column in row:
                    weight = row[edge_weight_column]
                    G.add_edge(source, target, weight=weight)
                else:
                    G.add_edge(source, target)
                    
                # Store node attributes
                for node, column in [(source, "source"), (target, "target")]:
                    # Store original node type
                    if node not in G.nodes or "node_type" not in G.nodes[node]:
                        G.nodes[node]["node_type"] = column
                        
                    # Store node size if specified
                    if node_size_column and node_size_column in row:
                        G.nodes[node]["size"] = row[node_size_column]
                        
                    # Store node color if specified
                    if node_color_column and node_color_column in row:
                        G.nodes[node]["color"] = row[node_color_column]
                        
                    # Store node label if specified
                    if node_label_column and node_label_column in row:
                        G.nodes[node]["label"] = row[node_label_column]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Get node positions based on specified layout
            if layout == "spring":
                pos = nx.spring_layout(G, seed=42)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            elif layout == "random":
                pos = nx.random_layout(G, seed=42)
            elif layout == "shell":
                pos = nx.shell_layout(G)
            elif layout == "kamada_kawai":
                # This often gives good results for supply chain networks
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    try:
                        pos = nx.kamada_kawai_layout(G)
                    except:
                        # Fall back to spring layout if kamada_kawai fails
                        pos = nx.spring_layout(G, seed=42)
            else:
                # Default to spring layout
                pos = nx.spring_layout(G, seed=42)
                
            # Get node sizes
            if node_size_column:
                # Use node attribute if available
                node_sizes = []
                for node in G.nodes():
                    size = G.nodes[node].get("size", 300)
                    # Ensure reasonable size
                    node_sizes.append(max(100, min(2000, size)))
            else:
                # Use degree centrality for size
                centrality = nx.degree_centrality(G)
                node_sizes = [centrality[node] * 5000 + 100 for node in G.nodes()]
                
            # Get node colors
            if node_color_column:
                # Use node attribute if available
                node_colors = []
                for node in G.nodes():
                    color = G.nodes[node].get("color")
                    if color:
                        node_colors.append(color)
                    else:
                        # Color based on node type
                        node_type = G.nodes[node].get("node_type", "other")
                        if node_type == "source":
                            node_colors.append(self.default_colors[0])
                        elif node_type == "target":
                            node_colors.append(self.default_colors[1])
                        else:
                            node_colors.append(self.default_colors[2])
            else:
                # Color based on node type
                node_colors = []
                for node in G.nodes():
                    node_type = G.nodes[node].get("node_type", "other")
                    if node_type == "source":
                        node_colors.append(self.default_colors[0])
                    elif node_type == "target":
                        node_colors.append(self.default_colors[1])
                    else:
                        node_colors.append(self.default_colors[2])
                        
            # Get edge weights
            if edge_weight_column:
                edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
                # Scale weights for visualization
                edge_weights = [max(1, min(5, w / max(edge_weights) * 5)) for w in edge_weights]
            else:
                edge_weights = [1] * len(G.edges())
                
            # Get node labels
            if node_label_column:
                node_labels = {}
                for node in G.nodes():
                    label = G.nodes[node].get("label")
                    if label:
                        node_labels[node] = label
                    else:
                        node_labels[node] = str(node)
            else:
                node_labels = {node: str(node) for node in G.nodes()}
                
            # Draw the network
            nodes = nx.draw_networkx_nodes(
                G, pos, 
                node_size=node_sizes,
                node_color=node_colors,
                alpha=0.8,
                ax=ax
            )
            
            edges = nx.draw_networkx_edges(
                G, pos,
                width=edge_weights,
                alpha=0.5,
                edge_color='gray',
                arrows=True,
                arrowsize=10,
                ax=ax
            )
            
            # Add labels
            nx.draw_networkx_labels(
                G, pos,
                labels=node_labels,
                font_size=8,
                font_family='sans-serif',
                ax=ax
            )
            
            # Remove axis
            ax.set_axis_off()
            
            # Set title
            ax.set_title(title)
            
            # Set margins
            plt.margins(0.1)
            
            # Tight layout
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Calculate network metrics
            metrics = {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "density": nx.density(G),
                "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes()
            }
            
            # Add more metrics if graph is not too large
            if G.number_of_nodes() < 500:  # Limit for computational efficiency
                try:
                    metrics["avg_shortest_path"] = nx.average_shortest_path_length(G)
                except (nx.NetworkXError, nx.NetworkXNoPath):
                    # Graph is not strongly connected
                    metrics["avg_shortest_path"] = None
                    
                try:
                    metrics["diameter"] = nx.diameter(G)
                except (nx.NetworkXError, nx.NetworkXNoPath):
                    metrics["diameter"] = None
                    
                # Calculate centrality for top nodes
                centrality = nx.degree_centrality(G)
                top_central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                metrics["top_central_nodes"] = [{"node": str(n), "centrality": c} for n, c in top_central_nodes]
            
            # Prepare node data for result
            node_data = []
            for node in G.nodes():
                node_info = {
                    "id": node,
                    "type": G.nodes[node].get("node_type", "other"),
                    "degree": G.degree(node),
                    "in_degree": G.in_degree(node),
                    "out_degree": G.out_degree(node)
                }
                
                # Add position
                if node in pos:
                    node_info["x"] = float(pos[node][0])
                    node_info["y"] = float(pos[node][1])
                    
                # Add other attributes
                for attr in G.nodes[node]:
                    if attr not in node_info:
                        node_info[attr] = G.nodes[node][attr]
                        
                node_data.append(node_info)
                
            # Prepare edge data for result
            edge_data = []
            for u, v in G.edges():
                edge_info = {
                    "source": u,
                    "target": v
                }
                
                # Add attributes
                for attr in G[u][v]:
                    edge_info[attr] = G[u][v][attr]
                    
                edge_data.append(edge_info)
            
            # Prepare result
            result = {
                "type": "network_graph",
                "title": title,
                "source_column": source_column,
                "target_column": target_column,
                "layout": layout,
                "nodes": node_data,
                "edges": edge_data,
                "metrics": metrics,
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating network graph: {str(e)}")
            raise
            
    def generate_supply_chain_network(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        source_column: str,
        target_column: str,
        tier_column: Optional[str] = None,
        material_column: Optional[str] = None,
        volume_column: Optional[str] = None,
        risk_column: Optional[str] = None,
        title: str = "Supply Chain Network",
        highlight_bottlenecks: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a supply chain network visualization with tiers.
        
        Args:
            data: DataFrame or list of dicts containing the data
            source_column: Column name for source nodes (suppliers)
            target_column: Column name for target nodes (customers)
            tier_column: Column name for supplier tier
            material_column: Column name for material type
            volume_column: Column name for volume or value
            risk_column: Column name for risk scores
            title: Chart title
            highlight_bottlenecks: Whether to highlight bottleneck nodes
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing chart data and metadata
        """
        try:
            # Convert to DataFrame if list
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            # Create network graph
            G = nx.DiGraph()
            
            # Add a company node if not in data
            company_node = "Company"
            G.add_node(company_node, node_type="company", tier=0, name="Company")
            
            # Add edges from data
            for _, row in df.iterrows():
                source = row[source_column]
                target = row[target_column]
                
                # Add nodes with attributes
                if source not in G:
                    node_attrs = {"node_type": "supplier", "name": source}
                    if tier_column and tier_column in row:
                        node_attrs["tier"] = row[tier_column]
                    G.add_node(source, **node_attrs)
                
                if target not in G:
                    node_attrs = {"node_type": "customer", "name": target}
                    if tier_column and tier_column in row:
                        node_attrs["tier"] = row[tier_column]
                    G.add_node(target, **node_attrs)
                
                # Add edge with attributes
                edge_attrs = {}
                if material_column and material_column in row:
                    edge_attrs["material"] = row[material_column]
                if volume_column and volume_column in row:
                    edge_attrs["volume"] = row[volume_column]
                    
                G.add_edge(source, target, **edge_attrs)
                
                # Connect tier 1 suppliers to company if they have no customers
                if tier_column and tier_column in row and row[tier_column] == 1:
                    if target not in G:
                        G.add_edge(source, company_node)
            
            # Add risk scores if available
            if risk_column:
                for _, row in df.iterrows():
                    node = row[source_column]
                    if node in G and risk_column in row:
                        G.nodes[node]["risk"] = row[risk_column]
            
            # Identify bottlenecks if requested
            if highlight_bottlenecks:
                # Calculate betweenness centrality
                try:
                    betweenness = nx.betweenness_centrality(G)
                    nx.set_node_attributes(G, betweenness, 'betweenness')
                    
                    # Identify nodes with high betweenness
                    threshold = np.percentile(list(betweenness.values()), 80)
                    for node, value in betweenness.items():
                        G.nodes[node]["is_bottleneck"] = value > threshold
                except:
                    # Fall back if centrality calculation fails
                    for node in G.nodes():
                        G.nodes[node]["is_bottleneck"] = False
            
            # Create a hierarchical layout based on tiers
            if tier_column:
                # Group nodes by tier
                tiers = {}
                for node in G.nodes():
                    tier = G.nodes[node].get("tier", 0)
                    if tier not in tiers:
                        tiers[tier] = []
                    tiers[tier].append(node)
                
                # Create position dictionary
                pos = {}
                
                # Place company at the center top
                pos[company_node] = (0, 1)
                
                # Place other tiers in layers
                tier_levels = sorted(tiers.keys())
                for i, tier in enumerate(tier_levels):
                    if tier == 0:
                        continue  # Skip company
                        
                    y_level = 1 - (i / max(1, len(tier_levels) - 1))
                    nodes = tiers[tier]
                    
                    # Place nodes horizontally
                    for j, node in enumerate(nodes):
                        x_pos = (j - len(nodes)/2) / max(1, len(nodes))
                        pos[node] = (x_pos, y_level)
            else:
                # Use kamada_kawai_layout if no tier information
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    try:
                        pos = nx.kamada_kawai_layout(G)
                    except:
                        # Fall back to spring layout
                        pos = nx.spring_layout(G, seed=42)
            
            # Create figure
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", self.default_figsize))
            
            # Set node colors based on type and risk
            node_colors = []
            for node in G.nodes():
                node_type = G.nodes[node].get("node_type", "supplier")
                
                if node_type == "company":
                    color = "tab:blue"
                elif G.nodes[node].get("is_bottleneck", False):
                    color = "tab:red"  # Highlight bottlenecks
                elif risk_column and "risk" in G.nodes[node]:
                    # Color based on risk (red = high risk, green = low risk)
                    risk = G.nodes[node]["risk"]
                    color = plt.cm.RdYlGn_r(risk)
                else:
                    # Color based on tier if available
                    tier = G.nodes[node].get("tier", 1)
                    tier_colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
                    color = tier_colors[min(tier, len(tier_colors) - 1)]
                
                node_colors.append(color)
            
            # Set node sizes
            node_sizes = []
            for node in G.nodes():
                if node == company_node:
                    size = 800  # Larger for company
                elif G.nodes[node].get("is_bottleneck", False):
                    size = 600  # Larger for bottlenecks
                else:
                    # Size based on connections
                    connections = G.degree(node)
                    size = 300 + (connections * 50)
                
                # Scale by volume if available
                if volume_column and "volume" in G.nodes[node]:
                    size = size * (1 + G.nodes[node]["volume"] / 100)
                    
                node_sizes.append(size)
            
            # Set edge widths
            edge_widths = []
            for u, v in G.edges():
                if volume_column and "volume" in G[u][v]:
                    # Scale by volume
                    volume = G[u][v]["volume"]
                    width = 1 + (volume / 10)
                else:
                    width = 1
                    
                edge_widths.append(width)
            
            # Draw the network
            nodes = nx.draw_networkx_nodes(
                G, pos, 
                node_size=node_sizes,
                node_color=node_colors,
                alpha=0.8,
                ax=ax
            )
            
            edges = nx.draw_networkx_edges(
                G, pos,
                width=edge_widths,
                alpha=0.5,
                edge_color='gray',
                arrows=True,
                arrowsize=10,
                ax=ax
            )
            
            # Add labels
            labels = {}
            for node in G.nodes():
                # Use name attribute if available, otherwise node ID
                labels[node] = G.nodes[node].get("name", str(node))
                
                # Add tier info for suppliers
                if tier_column and "tier" in G.nodes[node]:
                    tier = G.nodes[node]["tier"]
                    if tier > 0:  # Only for suppliers
                        labels[node] = f"{labels[node]}\n(Tier {tier})"
                        
                # Indicate bottlenecks
                if G.nodes[node].get("is_bottleneck", False):
                    labels[node] = f"{labels[node]}\n[BOTTLENECK]"
            
            nx.draw_networkx_labels(
                G, pos,
                labels=labels,
                font_size=8,
                font_family='sans-serif',
                ax=ax
            )
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=10, label='Company'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:green', markersize=10, label='Tier 1 Supplier'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:orange', markersize=10, label='Tier 2 Supplier')
            ]
            
            if highlight_bottlenecks:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', markersize=10, label='Bottleneck')
                )
                
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Remove axis
            ax.set_axis_off()
            
            # Set title
            ax.set_title(title)
            
            # Set margins
            plt.margins(0.1)
            
            # Tight layout
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Calculate supply chain metrics
            metrics = {
                "supplier_count": sum(1 for node in G.nodes() if G.nodes[node].get("node_type") == "supplier"),
                "tier1_count": sum(1 for node in G.nodes() if G.nodes[node].get("tier") == 1),
                "tier2_count": sum(1 for node in G.nodes() if G.nodes[node].get("tier") == 2),
                "tier3plus_count": sum(1 for node in G.nodes() if G.nodes[node].get("tier", 0) >= 3),
                "connection_count": G.number_of_edges(),
                "avg_connections": sum(dict(G.degree()).values()) / max(1, G.number_of_nodes() - 1),  # Excluding company
                "bottleneck_count": sum(1 for node in G.nodes() if G.nodes[node].get("is_bottleneck", False))
            }
            
            # Prepare node data for result
            node_data = []
            for node in G.nodes():
                node_info = {
                    "id": node,
                    "name": G.nodes[node].get("name", str(node)),
                    "type": G.nodes[node].get("node_type", "supplier"),
                    "connections": G.degree(node)
                }
                
                # Add tier if available
                if "tier" in G.nodes[node]:
                    node_info["tier"] = G.nodes[node]["tier"]
                    
                # Add risk if available
                if "risk" in G.nodes[node]:
                    node_info["risk"] = G.nodes[node]["risk"]
                    
                # Add bottleneck flag
                if "is_bottleneck" in G.nodes[node]:
                    node_info["is_bottleneck"] = G.nodes[node]["is_bottleneck"]
                    
                # Add position
                if node in pos:
                    node_info["x"] = float(pos[node][0])
                    node_info["y"] = float(pos[node][1])
                    
                node_data.append(node_info)
                
            # Prepare edge data for result
            edge_data = []
            for u, v in G.edges():
                edge_info = {
                    "source": u,
                    "source_name": G.nodes[u].get("name", str(u)),
                    "target": v,
                    "target_name": G.nodes[v].get("name", str(v))
                }
                
                # Add material if available
                if "material" in G[u][v]:
                    edge_info["material"] = G[u][v]["material"]
                    
                # Add volume if available
                if "volume" in G[u][v]:
                    edge_info["volume"] = G[u][v]["volume"]
                    
                edge_data.append(edge_info)
            
            # Prepare result
            result = {
                "type": "supply_chain_network",
                "title": title,
                "source_column": source_column,
                "target_column": target_column,
                "tier_column": tier_column,
                "material_column": material_column,
                "volume_column": volume_column,
                "risk_column": risk_column,
                "nodes": node_data,
                "edges": edge_data,
                "metrics": metrics,
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating supply chain network: {str(e)}")
            raise
            
    def generate_bottleneck_analysis(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        source_column: str,
        target_column: str,
        value_column: Optional[str] = None,
        tier_column: Optional[str] = None,
        risk_column: Optional[str] = None,
        title: str = "Bottleneck Analysis",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a bottleneck analysis visualization.
        
        Args:
            data: DataFrame or list of dicts containing the data
            source_column: Column name for source nodes
            target_column: Column name for target nodes
            value_column: Column name for flow values
            tier_column: Column name for supplier tier
            risk_column: Column name for risk scores
            title: Chart title
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing chart data and metadata
        """
        try:
            # Convert to DataFrame if list
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            # Create network graph
            G = nx.DiGraph()
            
            # Add edges from data
            for _, row in df.iterrows():
                source = row[source_column]
                target = row[target_column]
                
                # Add nodes with attributes
                node_attrs = {}
                if tier_column and tier_column in row:
                    node_attrs["tier"] = row[tier_column]
                if risk_column and risk_column in row:
                    node_attrs["risk"] = row[risk_column]
                    
                if source not in G:
                    G.add_node(source, **node_attrs)
                elif node_attrs:
                    # Update attributes if not already set
                    for attr, value in node_attrs.items():
                        if attr not in G.nodes[source]:
                            G.nodes[source][attr] = value
                
                if target not in G:
                    G.add_node(target, **node_attrs)
                elif node_attrs:
                    # Update attributes if not already set
                    for attr, value in node_attrs.items():
                        if attr not in G.nodes[target]:
                            G.nodes[target][attr] = value
                
                # Add edge with value if available
                edge_attrs = {}
                if value_column and value_column in row:
                    edge_attrs["value"] = row[value_column]
                    
                G.add_edge(source, target, **edge_attrs)
            
            # Calculate centrality measures to identify bottlenecks
            try:
                # Betweenness centrality
                betweenness = nx.betweenness_centrality(G)
                nx.set_node_attributes(G, betweenness, 'betweenness')
                
                # Degree centrality
                degree = nx.degree_centrality(G)
                nx.set_node_attributes(G, degree, 'degree_centrality')
                
                # Calculate downstream impact
                downstream_impact = {}
                for node in G.nodes():
                    try:
                        # Count number of nodes that depend on this node
                        descendants = list(nx.descendants(G, node))
                        impact = len(descendants)
                    except (nx.NetworkXError, nx.NetworkXNoPath):
                        impact = 0
                    downstream_impact[node] = impact
                
                nx.set_node_attributes(G, downstream_impact, 'downstream_impact')
                
                # Combine metrics to identify bottlenecks
                bottleneck_score = {}
                for node in G.nodes():
                    # Weighted combination of centrality metrics
                    b_score = (
                        betweenness[node] * 0.5 +
                        degree[node] * 0.2 +
                        downstream_impact[node] / max(1, max(downstream_impact.values())) * 0.3
                    )
                    
                    # Adjust by risk if available
                    if risk_column and "risk" in G.nodes[node]:
                        b_score *= (1 + G.nodes[node]["risk"])
                        
                    bottleneck_score[node] = b_score
                
                nx.set_node_attributes(G, bottleneck_score, 'bottleneck_score')
                
                # Flag top bottlenecks
                threshold = np.percentile(list(bottleneck_score.values()), 80)
                for node, score in bottleneck_score.items():
                    G.nodes[node]["is_bottleneck"] = score > threshold
            
            except Exception as e:
                logger.warning(f"Error calculating centrality: {str(e)}")
                # Set default values if calculation fails
                for node in G.nodes():
                    G.nodes[node]["betweenness"] = 0.0
                    G.nodes[node]["degree_centrality"] = G.degree(node) / max(1, G.number_of_nodes() - 1)
                    G.nodes[node]["downstream_impact"] = 0
                    G.nodes[node]["bottleneck_score"] = 0.0
                    G.nodes[node]["is_bottleneck"] = False
            
            # Create layout
            try:
                # Try kamada_kawai for better visualization of bottlenecks
                pos = nx.kamada_kawai_layout(G)
            except:
                # Fall back to spring layout
                pos = nx.spring_layout(G, seed=42)
            
            # Create figure
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", self.default_figsize))
            
            # Set node colors based on bottleneck score
            node_colors = []
            for node in G.nodes():
                if G.nodes[node].get("is_bottleneck", False):
                    # Red for bottlenecks
                    node_colors.append('red')
                else:
                    # Color based on centrality
                    score = G.nodes[node].get("bottleneck_score", 0)
                    # Use colormap: green to yellow to orange
                    node_colors.append(plt.cm.YlOrRd(score * 0.8))
            
            # Set node sizes based on bottleneck score
            node_sizes = []
            for node in G.nodes():
                score = G.nodes[node].get("bottleneck_score", 0)
                size = 100 + (score * 2000)  # Scale size
                node_sizes.append(size)
            
            # Set edge width based on value
            if value_column:
                edge_weights = []
                for u, v in G.edges():
                    if "value" in G[u][v]:
                        value = G[u][v]["value"]
                        width = 0.5 + (value / 10)  # Scale width
                    else:
                        width = 1.0
                    edge_weights.append(width)
            else:
                edge_weights = [1.0] * G.number_of_edges()
            
            # Draw the network
            nodes = nx.draw_networkx_nodes(
                G, pos, 
                node_size=node_sizes,
                node_color=node_colors,
                alpha=0.8,
                ax=ax
            )
            
            edges = nx.draw_networkx_edges(
                G, pos,
                width=edge_weights,
                alpha=0.5,
                edge_color='gray',
                arrows=True,
                arrowsize=10,
                connectionstyle="arc3,rad=0.1",  # Curved edges
                ax=ax
            )
            
            # Add labels for bottlenecks
            labels = {}
            for node in G.nodes():
                if G.nodes[node].get("is_bottleneck", False):
                    labels[node] = node
                else:
                    # Only label nodes with high scores
                    score = G.nodes[node].get("bottleneck_score", 0)
                    if score > threshold * 0.8:
                        labels[node] = node
            
            nx.draw_networkx_labels(
                G, pos,
                labels=labels,
                font_size=8,
                font_family='sans-serif',
                ax=ax
            )
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Critical Bottleneck'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.YlOrRd(0.8), markersize=10, label='High Centrality'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.YlOrRd(0.4), markersize=10, label='Medium Centrality'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.YlOrRd(0.1), markersize=10, label='Low Centrality')
            ]
            
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Remove axis
            ax.set_axis_off()
            
            # Set title
            ax.set_title(title)
            
            # Set margins
            plt.margins(0.1)
            
            # Tight layout
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Identify top bottlenecks
            top_bottlenecks = []
            for node in G.nodes():
                if G.nodes[node].get("is_bottleneck", False):
                    top_bottlenecks.append({
                        "node": node,
                        "bottleneck_score": G.nodes[node].get("bottleneck_score", 0),
                        "betweenness": G.nodes[node].get("betweenness", 0),
                        "degree_centrality": G.nodes[node].get("degree_centrality", 0),
                        "downstream_impact": G.nodes[node].get("downstream_impact", 0),
                        "risk": G.nodes[node].get("risk", None),
                        "tier": G.nodes[node].get("tier", None)
                    })
            
            # Sort by bottleneck score
            top_bottlenecks.sort(key=lambda x: x["bottleneck_score"], reverse=True)
            
            # Calculate overall metrics
            metrics = {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "bottleneck_count": len(top_bottlenecks),
                "avg_centrality": sum(nx.get_node_attributes(G, 'betweenness').values()) / G.number_of_nodes(),
                "network_density": nx.density(G),
                "top_bottlenecks": top_bottlenecks[:5]  # Top 5 bottlenecks
            }
            
            # Prepare node data for result
            node_data = []
            for node in G.nodes():
                node_info = {
                    "id": node,
                    "betweenness": G.nodes[node].get("betweenness", 0),
                    "degree_centrality": G.nodes[node].get("degree_centrality", 0),
                    "downstream_impact": G.nodes[node].get("downstream_impact", 0),
                    "bottleneck_score": G.nodes[node].get("bottleneck_score", 0),
                    "is_bottleneck": G.nodes[node].get("is_bottleneck", False)
                }
                
                # Add tier if available
                if "tier" in G.nodes[node]:
                    node_info["tier"] = G.nodes[node]["tier"]
                    
                # Add risk if available
                if "risk" in G.nodes[node]:
                    node_info["risk"] = G.nodes[node]["risk"]
                    
                # Add position
                if node in pos:
                    node_info["x"] = float(pos[node][0])
                    node_info["y"] = float(pos[node][1])
                    
                node_data.append(node_info)
            
            # Prepare result
            result = {
                "type": "bottleneck_analysis",
                "title": title,
                "source_column": source_column,
                "target_column": target_column,
                "value_column": value_column,
                "tier_column": tier_column,
                "risk_column": risk_column,
                "nodes": node_data,
                "top_bottlenecks": top_bottlenecks,
                "metrics": metrics,
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating bottleneck analysis: {str(e)}")
            raise