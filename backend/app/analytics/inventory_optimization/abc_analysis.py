"""
ABC Inventory Analysis Module

This module provides functions for performing ABC analysis on inventory items
to categorize them based on their importance (value, usage, etc.).
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import json

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class ABCAnalysis:
    """
    Performs ABC inventory analysis on products.
    """
    
    def __init__(
        self,
        class_a_threshold: float = 0.8,
        class_b_threshold: float = 0.95,
        class_c_threshold: float = 1.0
    ):
        """
        Initialize ABC Analysis with classification thresholds.
        
        Args:
            class_a_threshold: Cumulative percentage threshold for Class A (default: 80%)
            class_b_threshold: Cumulative percentage threshold for Class B (default: 95%)
            class_c_threshold: Cumulative percentage threshold for Class C (default: 100%)
        """
        self.class_a_threshold = class_a_threshold
        self.class_b_threshold = class_b_threshold
        self.class_c_threshold = class_c_threshold
    
    def perform_analysis(
        self,
        items: List[Dict[str, Any]],
        value_field: str = "annual_usage_value",
        id_field: str = "product_id",
        name_field: Optional[str] = "product_name"
    ) -> Dict[str, Any]:
        """
        Perform ABC analysis on a list of items.
        
        Args:
            items: List of item dictionaries
            value_field: Field name for the value to analyze
            id_field: Field name for the item ID
            name_field: Field name for the item name (optional)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Validate inputs
            if not items:
                return {
                    "error": "No items provided for analysis",
                    "class_a_items": [],
                    "class_b_items": [],
                    "class_c_items": []
                }
            
            # Create a DataFrame for easier processing
            df = pd.DataFrame(items)
            
            # Ensure required fields exist
            if value_field not in df.columns:
                return {
                    "error": f"Value field '{value_field}' not found in items",
                    "class_a_items": [],
                    "class_b_items": [],
                    "class_c_items": []
                }
            
            if id_field not in df.columns:
                return {
                    "error": f"ID field '{id_field}' not found in items",
                    "class_a_items": [],
                    "class_b_items": [],
                    "class_c_items": []
                }
            
            # Sort by value in descending order
            df = df.sort_values(by=value_field, ascending=False)
            
            # Calculate percentage and cumulative percentage
            total_value = df[value_field].sum()
            df['percentage'] = df[value_field] / total_value
            df['cumulative_percentage'] = df['percentage'].cumsum()
            
            # Assign ABC classes
            df['class'] = 'C'
            df.loc[df['cumulative_percentage'] <= self.class_a_threshold, 'class'] = 'A'
            df.loc[(df['cumulative_percentage'] > self.class_a_threshold) & 
                   (df['cumulative_percentage'] <= self.class_b_threshold), 'class'] = 'B'
            
            # Prepare results by class
            class_a_items = df[df['class'] == 'A'].to_dict(orient='records')
            class_b_items = df[df['class'] == 'B'].to_dict(orient='records')
            class_c_items = df[df['class'] == 'C'].to_dict(orient='records')
            
            # Calculate summary statistics
            class_a_count = len(class_a_items)
            class_b_count = len(class_b_items)
            class_c_count = len(class_c_items)
            total_count = len(items)
            
            class_a_value = sum(item[value_field] for item in class_a_items)
            class_b_value = sum(item[value_field] for item in class_b_items)
            class_c_value = sum(item[value_field] for item in class_c_items)
            
            # Prepare analysis summary
            analysis_summary = {
                "total_items": total_count,
                "total_value": total_value,
                "class_a": {
                    "count": class_a_count,
                    "percentage_of_items": round(class_a_count / total_count * 100, 2),
                    "value": class_a_value,
                    "percentage_of_value": round(class_a_value / total_value * 100, 2)
                },
                "class_b": {
                    "count": class_b_count,
                    "percentage_of_items": round(class_b_count / total_count * 100, 2),
                    "value": class_b_value,
                    "percentage_of_value": round(class_b_value / total_value * 100, 2)
                },
                "class_c": {
                    "count": class_c_count,
                    "percentage_of_items": round(class_c_count / total_count * 100, 2),
                    "value": class_c_value,
                    "percentage_of_value": round(class_c_value / total_value * 100, 2)
                }
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis_summary)
            
            return {
                "analysis_summary": analysis_summary,
                "class_a_items": class_a_items,
                "class_b_items": class_b_items,
                "class_c_items": class_c_items,
                "recommendations": recommendations,
                "parameters": {
                    "class_a_threshold": self.class_a_threshold,
                    "class_b_threshold": self.class_b_threshold,
                    "class_c_threshold": self.class_c_threshold,
                    "value_field": value_field
                }
            }
            
        except Exception as e:
            logger.error(f"Error performing ABC analysis: {str(e)}")
            return {
                "error": str(e),
                "class_a_items": [],
                "class_b_items": [],
                "class_c_items": []
            }
    
    @staticmethod
    def _generate_recommendations(analysis_summary: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate recommendations based on ABC analysis.
        
        Args:
            analysis_summary: Summary of ABC analysis
            
        Returns:
            Dictionary with recommendations for each class
        """
        recommendations = {
            "class_a": [],
            "class_b": [],
            "class_c": []
        }
        
        try:
            # Get key values from analysis summary
            class_a_percent_items = analysis_summary["class_a"]["percentage_of_items"]
            class_a_percent_value = analysis_summary["class_a"]["percentage_of_value"]
            
            # Recommendations for Class A
            recommendations["class_a"] = [
                "Implement tight inventory control with frequent cycle counts.",
                "Use precise demand forecasting methods for accurate planning.",
                "Review inventory levels weekly to ensure optimal stocking.",
                f"These items represent only {class_a_percent_items}% of items but {class_a_percent_value}% of value - prioritize supplier negotiations for these items.",
                "Consider safety stock for critical items to prevent stockouts.",
                "Monitor supplier performance closely for these high-value items."
            ]
            
            # Recommendations for Class B
            recommendations["class_b"] = [
                "Implement moderate controls with regular review cycles.",
                "Use standard forecasting methods for planning.",
                "Review inventory levels bi-weekly or monthly.",
                "Maintain adequate safety stock levels.",
                "Consider automated reordering systems with periodic review.",
                "Evaluate supplier performance quarterly."
            ]
            
            # Recommendations for Class C
            recommendations["class_c"] = [
                "Use simple controls with minimal manual intervention.",
                "Consider bulk purchasing to minimize ordering costs.",
                "Implement longer review cycles (monthly or quarterly).",
                "Use higher safety stock to reduce attention needed.",
                "Simple forecasting methods are adequate.",
                "Consider consolidating suppliers to reduce complexity."
            ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating ABC analysis recommendations: {str(e)}")
            return {
                "class_a": ["Error generating recommendations."],
                "class_b": ["Error generating recommendations."],
                "class_c": ["Error generating recommendations."]
            }
    
    @staticmethod
    async def get_product_data(
        client_id: str,
        connection_id: Optional[str] = None,
        criteria: str = "annual_usage_value",
        period: str = "last_12_months"
    ) -> List[Dict[str, Any]]:
        """
        Get product data for ABC analysis.
        
        Args:
            client_id: Client ID
            connection_id: Optional connection ID
            criteria: Analysis criteria (e.g., "annual_usage_value", "pick_frequency")
            period: Time period for analysis
            
        Returns:
            List of product data dictionaries
        """
        try:
            # Implementation depends on data sources
            # This is a placeholder implementation
            
            # Get data from database
            from app.db.interfaces.product_interface import ProductInterface
            
            # Create interface
            product_interface = ProductInterface(client_id=client_id, connection_id=connection_id)
            
            # Get product data based on criteria
            if criteria == "annual_usage_value":
                products = await product_interface.get_products_with_annual_usage_value(period=period)
            elif criteria == "pick_frequency":
                products = await product_interface.get_products_with_pick_frequency(period=period)
            else:
                products = await product_interface.get_products_with_custom_criteria(criteria=criteria, period=period)
            
            return products
            
        except Exception as e:
            logger.error(f"Error getting product data for ABC analysis: {str(e)}")
            
            # Generate mock data for demonstration or testing
            mock_products = ABCAnalysis._generate_mock_product_data(
                criteria=criteria,
                count=100
            )
            
            logger.warning(f"Using mock product data for ABC analysis: {len(mock_products)} products")
            return mock_products
    
    @staticmethod
    def _generate_mock_product_data(
        criteria: str = "annual_usage_value",
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate mock product data for testing.
        
        Args:
            criteria: Analysis criteria
            count: Number of products to generate
            
        Returns:
            List of mock product data
        """
        products = []
        
        # Generate products with a Pareto-like distribution
        # 20% of products will account for 80% of the value
        np.random.seed(42)  # For reproducibility
        
        # Generate primary values (e.g., annual_usage_value)
        # Use a power law distribution to simulate Pareto
        values = np.random.power(0.5, count) * 1000000
        values.sort()
        values = values[::-1]  # Sort in descending order
        
        for i in range(count):
            # Generate a mock product
            product = {
                "product_id": f"P{i+1:04d}",
                "product_name": f"Product {i+1}",
                "category": f"Category {(i % 10) + 1}",
                "unit_cost": round(np.random.uniform(10, 100), 2),
                "annual_usage_value": round(float(values[i]), 2),
                "annual_usage_units": int(values[i] / np.random.uniform(10, 100)),
                "pick_frequency": int(np.random.power(1.5, 1)[0] * 1000),
                "current_stock": int(np.random.uniform(10, 500)),
                "lead_time_days": int(np.random.normal(14, 3)),
                "is_mock_data": True
            }
            
            products.append(product)
        
        return products

    @staticmethod
    async def perform_multi_criteria_abc_analysis(
        client_id: str,
        connection_id: Optional[str] = None,
        criteria: List[str] = ["annual_usage_value", "pick_frequency"],
        weights: Optional[List[float]] = None,
        period: str = "last_12_months"
    ) -> Dict[str, Any]:
        """
        Perform multi-criteria ABC analysis.
        
        Args:
            client_id: Client ID
            connection_id: Optional connection ID
            criteria: List of criteria to analyze
            weights: Optional weights for each criterion (must sum to 1.0)
            period: Time period for analysis
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Validate inputs
            if not criteria:
                return {"error": "No criteria provided for analysis"}
            
            if weights and len(weights) != len(criteria):
                return {"error": "Number of weights must match number of criteria"}
            
            # Set default weights if not provided (equal weights)
            if not weights:
                weights = [1.0 / len(criteria)] * len(criteria)
            
            # Normalize weights to sum to 1.0
            weights = [w / sum(weights) for w in weights]
            
            # Get product data for all criteria
            all_product_data = {}
            for criterion in criteria:
                products = await ABCAnalysis.get_product_data(
                    client_id=client_id,
                    connection_id=connection_id,
                    criteria=criterion,
                    period=period
                )
                all_product_data[criterion] = products
            
            # Create a unified product list with all criteria
            product_ids = set()
            for criterion, products in all_product_data.items():
                product_ids.update(p["product_id"] for p in products)
            
            unified_products = []
            for product_id in product_ids:
                product = {"product_id": product_id}
                
                # Find this product in each criterion's data
                for criterion, products in all_product_data.items():
                    matching_products = [p for p in products if p["product_id"] == product_id]
                    if matching_products:
                        # Add all fields from the matching product
                        for key, value in matching_products[0].items():
                            if key not in product:
                                product[key] = value
                    else:
                        # If product not found for this criterion, set value to 0
                        product[criterion] = 0
                
                unified_products.append(product)
            
            # Calculate normalized scores for each criterion
            for criterion, products in all_product_data.items():
                # Get all values for this criterion
                values = [p.get(criterion, 0) for p in products]
                if not values or max(values) == 0:
                    continue
                
                # Calculate normalization factor
                max_value = max(values)
                
                # Update unified products with normalized scores
                for product in unified_products:
                    normalized_key = f"{criterion}_normalized"
                    product[normalized_key] = product.get(criterion, 0) / max_value
            
            # Calculate weighted score for each product
            for product in unified_products:
                weighted_score = 0
                for i, criterion in enumerate(criteria):
                    normalized_key = f"{criterion}_normalized"
                    if normalized_key in product:
                        weighted_score += product[normalized_key] * weights[i]
                
                product["weighted_score"] = weighted_score
            
            # Perform ABC analysis on weighted scores
            abc_analyzer = ABCAnalysis()
            analysis_result = abc_analyzer.perform_analysis(
                items=unified_products,
                value_field="weighted_score",
                id_field="product_id",
                name_field="product_name" if "product_name" in unified_products[0] else None
            )
            
            # Add analysis metadata
            analysis_result["parameters"] = {
                "criteria": criteria,
                "weights": weights,
                "period": period
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error performing multi-criteria ABC analysis: {str(e)}")
            return {
                "error": str(e),
                "class_a_items": [],
                "class_b_items": [],
                "class_c_items": []
            }