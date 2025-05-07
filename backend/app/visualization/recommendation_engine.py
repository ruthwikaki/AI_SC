"""
Chart recommendation engine for the supply chain LLM platform.

This module provides functionality to recommend appropriate chart types
based on data characteristics and query intent.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import re

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class ChartRecommendationEngine:
    """
    Recommends appropriate chart types based on data and query.
    
    This class analyzes data characteristics and query intent to provide
    recommendations for the most suitable visualization types.
    """
    
    def __init__(self):
        """Initialize the chart recommendation engine."""
        # Define chart types and their characteristics
        self.chart_types = {
            "bar_chart": {
                "categorical_axis": True,
                "numeric_values": True,
                "max_categories": 50,
                "time_series": False,
                "relationships": False,
                "distributions": False,
                "composition": True,
                "comparison": True
            },
            "line_chart": {
                "categorical_axis": True,
                "numeric_values": True,
                "max_categories": 100,
                "time_series": True,
                "relationships": True,
                "distributions": False,
                "composition": False,
                "comparison": True
            },
            "pie_chart": {
                "categorical_axis": True,
                "numeric_values": True,
                "max_categories": 7,  # Pie charts work best with few categories
                "time_series": False,
                "relationships": False,
                "distributions": False,
                "composition": True,
                "comparison": False
            },
            "donut_chart": {
                "categorical_axis": True,
                "numeric_values": True,
                "max_categories": 7,
                "time_series": False,
                "relationships": False,
                "distributions": False,
                "composition": True,
                "comparison": False
            },
            "heatmap": {
                "categorical_axis": True,
                "numeric_values": True,
                "max_categories": 50,
                "time_series": False,
                "relationships": True,
                "distributions": True,
                "composition": False,
                "comparison": True
            },
            "sankey_diagram": {
                "categorical_axis": True,
                "numeric_values": True,
                "max_categories": 30,
                "time_series": False,
                "relationships": True,
                "distributions": False,
                "composition": True,
                "comparison": False
            },
            "network_graph": {
                "categorical_axis": False,
                "numeric_values": False,
                "max_categories": 100,
                "time_series": False,
                "relationships": True,
                "distributions": False,
                "composition": False,
                "comparison": False
            },
            "scatter_plot": {
                "categorical_axis": False,
                "numeric_values": True,
                "max_categories": 1000,
                "time_series": False,
                "relationships": True,
                "distributions": True,
                "composition": False,
                "comparison": False
            },
            "histogram": {
                "categorical_axis": False,
                "numeric_values": True,
                "max_categories": 1,  # Just one numerical variable
                "time_series": False,
                "relationships": False,
                "distributions": True,
                "composition": False,
                "comparison": False
            },
            "stacked_bar": {
                "categorical_axis": True,
                "numeric_values": True,
                "max_categories": 30,
                "time_series": False,
                "relationships": True,
                "distributions": False,
                "composition": True,
                "comparison": True
            },
            "grouped_bar": {
                "categorical_axis": True,
                "numeric_values": True,
                "max_categories": 30,
                "time_series": False,
                "relationships": False,
                "distributions": False,
                "composition": False,
                "comparison": True
            },
            "area_chart": {
                "categorical_axis": True,
                "numeric_values": True,
                "max_categories": 100,
                "time_series": True,
                "relationships": True,
                "distributions": False,
                "composition": True,
                "comparison": True
            },
            "stacked_area": {
                "categorical_axis": True,
                "numeric_values": True,
                "max_categories": 30,
                "time_series": True,
                "relationships": True,
                "distributions": False,
                "composition": True,
                "comparison": True
            },
            "calendar_heatmap": {
                "categorical_axis": False,
                "numeric_values": True,
                "max_categories": 1,
                "time_series": True,
                "relationships": False,
                "distributions": True,
                "composition": False,
                "comparison": False
            },
            "correlation_heatmap": {
                "categorical_axis": True,
                "numeric_values": True,
                "max_categories": 30,
                "time_series": False,
                "relationships": True,
                "distributions": False,
                "composition": False,
                "comparison": True
            },
            "supply_chain_network": {
                "categorical_axis": False,
                "numeric_values": False,
                "max_categories": 100,
                "time_series": False,
                "relationships": True,
                "distributions": False,
                "composition": False,
                "comparison": False
            },
            "bottleneck_analysis": {
                "categorical_axis": False,
                "numeric_values": True,
                "max_categories": 100,
                "time_series": False,
                "relationships": True,
                "distributions": False,
                "composition": False,
                "comparison": False
            }
        }
        
        # Define keywords that suggest chart types
        self.intent_keywords = {
            "bar_chart": [
                "compare", "comparison", "comparing", "versus", "vs", "rank", "ranking",
                "largest", "smallest", "highest", "lowest", "bar chart", "bar graph"
            ],
            "line_chart": [
                "trend", "trends", "over time", "time series", "historical", "forecast",
                "projections", "change", "pattern", "patterns", "line chart", "line graph",
                "progression", "timeline"
            ],
            "pie_chart": [
                "proportion", "percentage", "breakdown", "composition", "share", "part of",
                "partition", "segment", "pie chart", "contribution", "makeup"
            ],
            "donut_chart": [
                "donut", "doughnut", "proportion", "percentage", "breakdown", "composition"
            ],
            "heatmap": [
                "correlation", "matrix", "grid", "intensity", "density", "heatmap", "heat map",
                "concentration", "hotspots", "cross-tabulation", "crosstab"
            ],
            "sankey_diagram": [
                "flow", "flows", "transfer", "movement", "path", "pathway", "sankey",
                "sequence", "from-to", "origin-destination", "transformation"
            ],
            "network_graph": [
                "network", "connections", "connected", "relationship", "relationships",
                "links", "linked", "interconnection", "nodes", "graph", "dependency",
                "dependencies", "ties", "web"
            ],
            "scatter_plot": [
                "correlation", "relationship", "scatter", "distribution", "spread",
                "clustered", "clusters", "outliers", "regression", "trend line"
            ],
            "histogram": [
                "distribution", "frequency", "histogram", "statistical", "spread",
                "range", "bin", "bins", "counts", "grouped"
            ],
            "stacked_bar": [
                "stacked", "stack", "cumulative", "components", "sub-categories",
                "parts", "composition", "breakdown"
            ],
            "grouped_bar": [
                "grouped", "group", "side by side", "multiple categories", "comparison",
                "sub-group", "subgroup", "multiple series"
            ],
            "area_chart": [
                "area", "filled", "cumulative", "trend", "over time", "stacked area",
                "mountain", "aggregate"
            ],
            "calendar_heatmap": [
                "calendar", "dates", "daily", "weekly", "monthly", "yearly", "by day",
                "by week", "by month", "seasonal", "periodicity", "day of week", "patterns",
                "dates"
            ],
            "supply_chain_network": [
                "supply chain", "supplier", "tier", "multi-tier", "network", "suppliers",
                "dependencies", "chain", "upstream", "downstream",
                "connections", "mapping"
            ],
            "bottleneck_analysis": [
                "bottleneck", "bottlenecks", "constraints", "single point", "failure",
                "critical", "risk", "vulnerability", "vulnerable", "weak link",
                "restricted", "limitation", "choke point"
            ]
        }
        
        # Define typical chart use cases for supply chain data
        self.supply_chain_chart_use_cases = {
            "inventory_analysis": [
                "bar_chart", "line_chart", "heatmap", "stacked_bar", "histogram", "area_chart"
            ],
            "supplier_performance": [
                "bar_chart", "line_chart", "heatmap", "scatter_plot", "network_graph", 
                "bottleneck_analysis", "radar_chart"
            ],
            "logistics_performance": [
                "bar_chart", "line_chart", "sankey_diagram", "heatmap", "calendar_heatmap"
            ],
            "multi_tier_supply_chain": [
                "network_graph", "sankey_diagram", "supply_chain_network", "bottleneck_analysis"
            ],
            "demand_forecasting": [
                "line_chart", "area_chart", "bar_chart", "histogram"
            ],
            "cost_analysis": [
                "pie_chart", "donut_chart", "stacked_bar", "bar_chart", "line_chart"
            ],
            "risk_analysis": [
                "heatmap", "network_graph", "scatter_plot", "bottleneck_analysis"
            ]
        }
        
    def recommend_chart_types(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        query: Optional[str] = None,
        columns: Optional[List[str]] = None,
        max_recommendations: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Recommend chart types based on data characteristics and query.
        
        Args:
            data: DataFrame or list of dicts containing the data
            query: Natural language query indicating intent
            columns: Optional list of specific columns to consider
            max_recommendations: Maximum number of chart types to recommend
            
        Returns:
            List of recommended chart types with reasons
        """
        try:
            # Convert to DataFrame if list
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            # Filter columns if specified
            if columns:
                df = df[columns]
                
            # Analyze data characteristics
            data_characteristics = self._analyze_data(df)
            
            # Analyze query intent
            intent_scores = {}
            if query:
                intent_scores = self._analyze_query_intent(query)
                
            # Get supply chain domain-specific context
            domain_context = self._get_domain_context(query)
            
            # Score chart types based on data and intent
            chart_scores = self._score_chart_types(data_characteristics, intent_scores, domain_context)
            
            # Get top recommendations
            recommendations = self._get_top_recommendations(chart_scores, max_recommendations, data_characteristics)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error recommending chart types: {str(e)}")
            # Return safe default recommendations
            return [
                {
                    "chart_type": "bar_chart",
                    "score": 0.9,
                    "reason": "Bar charts are versatile for displaying categorical data",
                    "appropriate_columns": {"x": None, "y": None}
                },
                {
                    "chart_type": "line_chart",
                    "score": 0.7,
                    "reason": "Line charts are good for showing trends over time",
                    "appropriate_columns": {"x": None, "y": None}
                }
            ]
    
    def _analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data characteristics to determine suitable chart types.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of data characteristics
        """
        characteristics = {}
        
        # Check if DataFrame is empty
        if df.empty:
            return {
                "empty": True,
                "categorical_columns": [],
                "numeric_columns": [],
                "datetime_columns": [],
                "category_counts": {},
                "has_time_series": False
            }
        
        # Identify column types
        categorical_columns = []
        numeric_columns = []
        datetime_columns = []
        category_counts = {}
        
        for col in df.columns:
            # Check if datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_columns.append(col)
                characteristics["has_time_series"] = True
                continue
                
            # Try to convert to datetime
            try:
                pd.to_datetime(df[col])
                datetime_columns.append(col)
                characteristics["has_time_series"] = True
                continue
            except:
                pass
            
            # Check if numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
                continue
                
            # Must be categorical
            categorical_columns.append(col)
            # Count unique categories
            category_counts[col] = df[col].nunique()
        
        characteristics["categorical_columns"] = categorical_columns
        characteristics["numeric_columns"] = numeric_columns
        characteristics["datetime_columns"] = datetime_columns
        characteristics["category_counts"] = category_counts
        
        # Check specific chart-relevant patterns
        
        # Time series data
        characteristics["has_time_series"] = len(datetime_columns) > 0
        
        # Network data (source-target pairs)
        source_target_pairs = [
            ("source", "target"),
            ("from", "to"),
            ("origin", "destination"),
            ("supplier", "customer")
        ]
        for src, tgt in source_target_pairs:
            if src in df.columns and tgt in df.columns:
                characteristics["has_network_data"] = True
                characteristics["network_columns"] = {"source": src, "target": tgt}
                break
        else:
            characteristics["has_network_data"] = False
        
        # Supply chain indicators
        supply_chain_keywords = ["supplier", "inventory", "product", "order", "logistics", "shipment", "material"]
        sc_columns = [col for col in df.columns if any(kw in col.lower() for kw in supply_chain_keywords)]
        characteristics["supply_chain_columns"] = sc_columns
        characteristics["is_supply_chain_data"] = len(sc_columns) > 0
        
        # Check for tier information (multi-tier supply chain)
        tier_columns = [col for col in df.columns if "tier" in col.lower()]
        characteristics["has_tier_data"] = len(tier_columns) > 0
        
        # Total rows and columns
        characteristics["row_count"] = len(df)
        characteristics["column_count"] = len(df.columns)
        
        return characteristics
        
    def _analyze_query_intent(self, query: str) -> Dict[str, float]:
        """
        Analyze query to determine chart intent.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary mapping chart types to intent scores
        """
        # Normalize query
        query = query.lower()
        
        # Score for each chart type based on keywords
        scores = {}
        
        for chart_type, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    # Exact match gets higher score
                    if re.search(r'\b' + re.escape(keyword) + r'\b', query):
                        score += 1.0
                    else:
                        # Partial match gets lower score
                        score += 0.25
            
            # Normalize score (0 to 1)
            if score > 0:
                scores[chart_type] = min(1.0, score / max(3, len(keywords) / 5))
                
        # Check for explicit chart requests
        explicit_patterns = [
            (r'\bbar\s*charts?\b', "bar_chart"),
            (r'\bline\s*charts?\b', "line_chart"),
            (r'\bpie\s*charts?\b', "pie_chart"),
            (r'\bdonut\s*charts?\b', "donut_chart"),
            (r'\bheat\s*maps?\b', "heatmap"),
            (r'\bsankey\s*diagrams?\b', "sankey_diagram"),
            (r'\bnetwork\s*graphs?\b', "network_graph"),
            (r'\bhistograms?\b', "histogram"),
            (r'\bscatter\s*plots?\b', "scatter_plot"),
            (r'\bstacked\s*bars?\b', "stacked_bar"),
            (r'\bgrouped\s*bars?\b', "grouped_bar"),
            (r'\barea\s*charts?\b', "area_chart")
        ]
        
        for pattern, chart_type in explicit_patterns:
            if re.search(pattern, query):
                # Explicit request gets high score
                scores[chart_type] = 1.0
                
        return scores
        
    def _get_domain_context(self, query: Optional[str]) -> Dict[str, float]:
        """
        Get domain-specific context from query.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary mapping domain contexts to scores
        """
        if not query:
            return {}
            
        # Normalize query
        query = query.lower()
        
        # Domain context keywords
        domain_keywords = {
            "inventory_analysis": [
                "inventory", "stock", "on-hand", "safety stock", "abc analysis",
                "stockout", "inventory level", "inventory turnover", "days of supply",
                "fill rate", "stock coverage"
            ],
            "supplier_performance": [
                "supplier", "vendor", "supplier performance", "scorecard", "on-time delivery",
                "quality", "supplier rating", "late delivery", "defect rate", "lead time"
            ],
            "logistics_performance": [
                "logistics", "shipping", "transportation", "delivery", "carrier",
                "freight", "transit time", "on-time delivery", "shipping cost",
                "route", "warehouse", "distribution"
            ],
            "multi_tier_supply_chain": [
                "multi-tier", "tier", "sub-supplier", "supply chain network",
                "supply chain mapping", "visibility", "upstream supplier", "supplier network"
            ],
            "demand_forecasting": [
                "forecast", "demand", "prediction", "projected", "future demand",
                "sales forecast", "forecast accuracy", "trend analysis"
            ],
            "cost_analysis": [
                "cost", "expense", "spending", "budget", "price", "financial",
                "cost breakdown", "cost analysis", "spending analysis", "cost per unit"
            ],
            "risk_analysis": [
                "risk", "vulnerability", "disruption", "bottleneck", "resilience",
                "risk score", "risk mitigation", "risk assessment", "impact analysis"
            ]
        }
        
        # Score each domain
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    # Exact match gets higher score
                    if re.search(r'\b' + re.escape(keyword) + r'\b', query):
                        score += 1.0
                    else:
                        # Partial match gets lower score
                        score += 0.25
            
            # Normalize score (0 to 1)
            if score > 0:
                domain_scores[domain] = min(1.0, score / max(3, len(keywords) / 5))
                
        return domain_scores
        
    def _score_chart_types(
        self,
        data_characteristics: Dict[str, Any],
        intent_scores: Dict[str, float],
        domain_context: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Score each chart type based on data and intent.
        
        Args:
            data_characteristics: Data characteristics
            intent_scores: Query intent scores
            domain_context: Domain context scores
            
        Returns:
            Dictionary mapping chart types to scores
        """
        chart_scores = {}
        
        # Score based on data characteristics
        for chart_type, characteristics in self.chart_types.items():
            score = 0.0
            
            # Check if data matches chart requirements
            
            # Categorical axis requirement
            if characteristics["categorical_axis"]:
                if len(data_characteristics["categorical_columns"]) > 0 or len(data_characteristics["datetime_columns"]) > 0:
                    score += 0.2
                else:
                    # Chart requires categorical axis but data doesn't have one
                    chart_scores[chart_type] = 0.0
                    continue
            
            # Numeric values requirement
            if characteristics["numeric_values"]:
                if len(data_characteristics["numeric_columns"]) > 0:
                    score += 0.2
                else:
                    # Chart requires numeric values but data doesn't have any
                    chart_scores[chart_type] = 0.0
                    continue
            
            # Time series requirement
            if characteristics["time_series"]:
                if data_characteristics["has_time_series"]:
                    score += 0.3
                else:
                    # Chart is for time series but data isn't time-based
                    # Don't immediately disqualify, but reduce score
                    score -= 0.2
            
            # Relationship data requirement
            if characteristics["relationships"]:
                if data_characteristics.get("has_network_data", False):
                    score += 0.3
                # Don't disqualify if missing, relationships can be derived
            
            # Category count limitations
            if characteristics["categorical_axis"]:
                # Get category counts for categorical columns
                category_counts = data_characteristics["category_counts"]
                if category_counts:
                    avg_categories = sum(category_counts.values()) / len(category_counts)
                    if avg_categories > characteristics["max_categories"]:
                        # Too many categories for this chart type
                        score -= 0.3
                    elif chart_type in ["pie_chart", "donut_chart"] and avg_categories > 7:
                        # Especially penalize too many categories for pie charts
                        score -= 0.5
            
            # Special case for network charts
            if chart_type in ["network_graph", "supply_chain_network", "bottleneck_analysis", "sankey_diagram"]:
                if data_characteristics.get("has_network_data", False):
                    score += 0.4
                else:
                    # These charts strongly require network data
                    score -= 0.5
            
            # Special case for supply chain specific charts
            if chart_type in ["supply_chain_network", "bottleneck_analysis"]:
                if data_characteristics.get("is_supply_chain_data", False):
                    score += 0.3
                if data_characteristics.get("has_tier_data", False):
                    score += 0.3
                    
            # Add intent score if available
            if chart_type in intent_scores:
                score += intent_scores[chart_type] * 0.4
                
            # Add domain context score
            for domain, domain_score in domain_context.items():
                if domain in self.supply_chain_chart_use_cases:
                    if chart_type in self.supply_chain_chart_use_cases[domain]:
                        score += domain_score * 0.3
            
            # Ensure score is between 0 and 1
            chart_scores[chart_type] = max(0.0, min(1.0, score))
        
        return chart_scores
        
    def _get_top_recommendations(
        self,
        chart_scores: Dict[str, float],
        max_recommendations: int,
        data_characteristics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get top chart recommendations with reasons.
        
        Args:
            chart_scores: Score for each chart type
            max_recommendations: Maximum number of recommendations
            data_characteristics: Data characteristics
            
        Returns:
            List of recommendation dictionaries
        """
        # Sort charts by score
        sorted_charts = sorted(chart_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out low-scoring charts
        filtered_charts = [(chart, score) for chart, score in sorted_charts if score > 0.3]
        
        # Take top N recommendations
        top_charts = filtered_charts[:max_recommendations]
        
        # Generate reasons and column suggestions
        recommendations = []
        
        for chart_type, score in top_charts:
            # Generate reason for recommendation
            reason = self._generate_recommendation_reason(chart_type, score, data_characteristics)
            
            # Suggest appropriate columns
            appropriate_columns = self._suggest_appropriate_columns(chart_type, data_characteristics)
            
            recommendations.append({
                "chart_type": chart_type,
                "score": round(score, 2),
                "reason": reason,
                "appropriate_columns": appropriate_columns
            })
        
        return recommendations
        
    def _generate_recommendation_reason(
        self,
        chart_type: str,
        score: float,
        data_characteristics: Dict[str, Any]
    ) -> str:
        """
        Generate explanation for chart recommendation.
        
        Args:
            chart_type: Recommended chart type
            score: Recommendation score
            data_characteristics: Data characteristics
            
        Returns:
            Explanation string
        """
        # Chart-specific reasons
        reasons = {
            "bar_chart": "Bar charts are effective for comparing categorical data and showing relative quantities.",
            "line_chart": "Line charts are ideal for showing trends over time and continuous data.",
            "pie_chart": "Pie charts work well for showing percentage or proportional data.",
            "donut_chart": "Donut charts are good for showing percentage or proportional data with space for summary metrics.",
            "heatmap": "Heatmaps are excellent for showing patterns and relationships between categories.",
            "sankey_diagram": "Sankey diagrams help visualize flows and transformations between categories.",
            "network_graph": "Network graphs show relationships and connections between entities.",
            "scatter_plot": "Scatter plots reveal relationships and patterns between two numeric variables.",
            "histogram": "Histograms show the distribution of a single numeric variable.",
            "stacked_bar": "Stacked bar charts show both totals and composition of categories.",
            "grouped_bar": "Grouped bar charts are good for comparing multiple categories across groups.",
            "area_chart": "Area charts show volumes over time and are good for showing trends.",
            "stacked_area": "Stacked area charts show how different categories contribute to a total over time.",
            "calendar_heatmap": "Calendar heatmaps show patterns and intensity across dates.",
            "correlation_heatmap": "Correlation heatmaps show relationships between multiple variables.",
            "supply_chain_network": "Supply chain network visualizations show multi-tier supplier relationships and flows.",
            "bottleneck_analysis": "Bottleneck analysis identifies critical nodes and constraints in a network."
        }
        
        # Get generic reason
        reason = reasons.get(chart_type, f"{chart_type.replace('_', ' ').title()} is appropriate for this data.")
        
        # Add data-specific insights
        if chart_type in ["bar_chart", "pie_chart", "donut_chart"]:
            if len(data_characteristics["categorical_columns"]) > 0:
                categorical_col = data_characteristics["categorical_columns"][0]
                if data_characteristics["category_counts"].get(categorical_col, 0) <= 7:
                    reason += f" Your data has a good number of categories for this visualization."
                
        if chart_type in ["line_chart", "area_chart", "stacked_area"]:
            if data_characteristics.get("has_time_series", False):
                reason += f" Your data contains date/time information which works well with this chart type."
                
        if chart_type in ["network_graph", "supply_chain_network", "bottleneck_analysis", "sankey_diagram"]:
            if data_characteristics.get("has_network_data", False):
                reason += f" Your data contains source-target pairs which is required for this visualization."
                
        if chart_type in ["heatmap", "correlation_heatmap"]:
            if len(data_characteristics["categorical_columns"]) >= 2 and len(data_characteristics["numeric_columns"]) > 0:
                reason += f" Your data has categorical dimensions and numeric values needed for this chart."
                
        return reason
        
    def _suggest_appropriate_columns(
        self,
        chart_type: str,
        data_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest appropriate columns for the chart type.
        
        Args:
            chart_type: Chart type
            data_characteristics: Data characteristics
            
        Returns:
            Dictionary of column suggestions
        """
        suggestions = {}
        
        # Get available columns by type
        cat_cols = data_characteristics["categorical_columns"]
        num_cols = data_characteristics["numeric_columns"]
        date_cols = data_characteristics["datetime_columns"]
        
        # Chart-specific suggestions
        if chart_type in ["bar_chart", "stacked_bar", "grouped_bar"]:
            # X-axis: categorical, Y-axis: numeric
            suggestions["x"] = cat_cols[0] if cat_cols else (date_cols[0] if date_cols else None)
            suggestions["y"] = num_cols[0] if num_cols else None
            
            if chart_type in ["stacked_bar", "grouped_bar"] and len(cat_cols) > 1:
                suggestions["group"] = cat_cols[1]
                
        elif chart_type in ["line_chart", "area_chart", "stacked_area"]:
            # X-axis: date or categorical, Y-axis: numeric
            suggestions["x"] = date_cols[0] if date_cols else (cat_cols[0] if cat_cols else None)
            suggestions["y"] = num_cols[0] if num_cols else None
            
            if chart_type in ["stacked_area"] and len(cat_cols) > 0:
                suggestions["group"] = cat_cols[0]
                
        elif chart_type in ["pie_chart", "donut_chart"]:
            # Category and value
            suggestions["category"] = cat_cols[0] if cat_cols else None
            suggestions["value"] = num_cols[0] if num_cols else None
            
        elif chart_type in ["heatmap", "correlation_heatmap"]:
            # X and Y categories, value for intensity
            if len(cat_cols) >= 2:
                suggestions["x"] = cat_cols[0]
                suggestions["y"] = cat_cols[1]
            elif len(cat_cols) == 1 and len(date_cols) >= 1:
                suggestions["x"] = cat_cols[0]
                suggestions["y"] = date_cols[0]
            else:
                suggestions["x"] = cat_cols[0] if cat_cols else None
                suggestions["y"] = date_cols[0] if date_cols else None
                
            suggestions["value"] = num_cols[0] if num_cols else None
            
        elif chart_type in ["sankey_diagram", "network_graph", "supply_chain_network", "bottleneck_analysis"]:
            # Source and target columns
            if data_characteristics.get("has_network_data", False):
                network_cols = data_characteristics["network_columns"]
                suggestions["source"] = network_cols["source"]
                suggestions["target"] = network_cols["target"]
                
                # Value column for flow thickness
                if len(num_cols) > 0:
                    suggestions["value"] = num_cols[0]
            else:
                # Try to guess source-target columns
                source_candidates = [col for col in cat_cols if col.lower() in 
                                    ["source", "from", "origin", "supplier", "sender"]]
                target_candidates = [col for col in cat_cols if col.lower() in 
                                    ["target", "to", "destination", "customer", "receiver"]]
                
                suggestions["source"] = source_candidates[0] if source_candidates else (cat_cols[0] if cat_cols else None)
                suggestions["target"] = target_candidates[0] if target_candidates else (cat_cols[1] if len(cat_cols) > 1 else None)
                
                if len(num_cols) > 0:
                    suggestions["value"] = num_cols[0]
                    
        elif chart_type == "histogram":
            # Just need a numeric column
            suggestions["value"] = num_cols[0] if num_cols else None
            
        elif chart_type == "scatter_plot":
            # Need two numeric columns
            if len(num_cols) >= 2:
                suggestions["x"] = num_cols[0]
                suggestions["y"] = num_cols[1]
            elif len(num_cols) == 1 and len(date_cols) >= 1:
                suggestions["x"] = date_cols[0]
                suggestions["y"] = num_cols[0]
            else:
                suggestions["x"] = num_cols[0] if num_cols else None
                suggestions["y"] = num_cols[1] if len(num_cols) > 1 else None
                
            # Optional category for color
            if len(cat_cols) > 0:
                suggestions["color"] = cat_cols[0]
                
        elif chart_type == "calendar_heatmap":
            # Need date column and value
            suggestions["date"] = date_cols[0] if date_cols else None
            suggestions["value"] = num_cols[0] if num_cols else None
        
        return suggestions