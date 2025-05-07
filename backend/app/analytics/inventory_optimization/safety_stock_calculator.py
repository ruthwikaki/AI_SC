"""
Safety Stock Calculator Module for Inventory Optimization

This module provides functions for calculating safety stock levels
based on various methods and service level requirements.
"""

from typing import Dict, List, Any, Optional, Union
import math
import numpy as np
from datetime import datetime, timedelta
import json

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Standard Z-scores for common service levels
SERVICE_LEVEL_Z_SCORES = {
    0.50: 0.000,  # 50% service level
    0.75: 0.674,  # 75% service level
    0.80: 0.842,  # 80% service level
    0.85: 1.036,  # 85% service level
    0.90: 1.282,  # 90% service level
    0.95: 1.645,  # 95% service level
    0.96: 1.751,  # 96% service level
    0.97: 1.881,  # 97% service level
    0.98: 2.054,  # 98% service level
    0.99: 2.326,  # 99% service level
    0.995: 2.576, # 99.5% service level
    0.999: 3.090  # 99.9% service level
}

class SafetyStockCalculator:
    """
    Calculates safety stock levels based on demand and lead time variability.
    """
    
    @staticmethod
    def get_z_score(service_level: float) -> float:
        """
        Get Z-score for a given service level.
        
        Args:
            service_level: Service level as a decimal (e.g., 0.95 for 95%)
            
        Returns:
            Z-score corresponding to the service level
        """
        # Check if exact service level exists in the dictionary
        if service_level in SERVICE_LEVEL_Z_SCORES:
            return SERVICE_LEVEL_Z_SCORES[service_level]
        
        # Find closest service level
        closest_service_level = min(SERVICE_LEVEL_Z_SCORES.keys(), 
                                   key=lambda x: abs(x - service_level))
        
        logger.warning(f"Exact service level {service_level} not found, using closest: {closest_service_level}")
        return SERVICE_LEVEL_Z_SCORES[closest_service_level]
    
    @staticmethod
    def calculate_basic_safety_stock(
        demand_std_dev: float,
        lead_time_avg: float,
        service_level: float
    ) -> float:
        """
        Calculate safety stock using the basic formula.
        
        Safety Stock = Z × σ × √L
        
        Args:
            demand_std_dev: Standard deviation of demand (per time unit)
            lead_time_avg: Average lead time (in time units)
            service_level: Service level as a decimal (e.g., 0.95 for 95%)
            
        Returns:
            Safety stock quantity
        """
        try:
            # Get Z-score for service level
            z_score = SafetyStockCalculator.get_z_score(service_level)
            
            # Calculate safety stock
            safety_stock = z_score * demand_std_dev * math.sqrt(lead_time_avg)
            
            return max(0, safety_stock)  # Safety stock can't be negative
            
        except Exception as e:
            logger.error(f"Error calculating basic safety stock: {str(e)}")
            return 0
    
    @staticmethod
    def calculate_advanced_safety_stock(
        demand_std_dev: float,
        lead_time_avg: float,
        lead_time_std_dev: float,
        demand_avg: float,
        service_level: float
    ) -> float:
        """
        Calculate safety stock considering both demand and lead time variability.
        
        Safety Stock = Z × √(L × σ_d² + D² × σ_l²)
        
        Args:
            demand_std_dev: Standard deviation of demand (per time unit)
            lead_time_avg: Average lead time (in time units)
            lead_time_std_dev: Standard deviation of lead time
            demand_avg: Average demand (per time unit)
            service_level: Service level as a decimal (e.g., 0.95 for 95%)
            
        Returns:
            Safety stock quantity
        """
        try:
            # Get Z-score for service level
            z_score = SafetyStockCalculator.get_z_score(service_level)
            
            # Calculate safety stock considering both demand and lead time variability
            term1 = lead_time_avg * (demand_std_dev ** 2)
            term2 = (demand_avg ** 2) * (lead_time_std_dev ** 2)
            
            safety_stock = z_score * math.sqrt(term1 + term2)
            
            return max(0, safety_stock)  # Safety stock can't be negative
            
        except Exception as e:
            logger.error(f"Error calculating advanced safety stock: {str(e)}")
            return 0
    
    @staticmethod
    def calculate_safety_stock_from_historical_data(
        historical_demand: List[float],
        historical_lead_times: List[float],
        service_level: float,
        forecast_period: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate safety stock from historical demand and lead time data.
        
        Args:
            historical_demand: List of historical demand values
            historical_lead_times: List of historical lead time values
            service_level: Service level as a decimal (e.g., 0.95 for 95%)
            forecast_period: Number of time periods in forecast horizon
            
        Returns:
            Dictionary with safety stock and related statistics
        """
        try:
            # Calculate demand statistics
            demand_avg = np.mean(historical_demand)
            demand_std_dev = np.std(historical_demand, ddof=1)  # ddof=1 for sample standard deviation
            
            # Calculate lead time statistics
            lead_time_avg = np.mean(historical_lead_times)
            lead_time_std_dev = np.std(historical_lead_times, ddof=1)
            
            # Get Z-score for service level
            z_score = SafetyStockCalculator.get_z_score(service_level)
            
            # Calculate basic safety stock
            basic_safety_stock = SafetyStockCalculator.calculate_basic_safety_stock(
                demand_std_dev=demand_std_dev,
                lead_time_avg=lead_time_avg,
                service_level=service_level
            )
            
            # Calculate advanced safety stock
            advanced_safety_stock = SafetyStockCalculator.calculate_advanced_safety_stock(
                demand_std_dev=demand_std_dev,
                lead_time_avg=lead_time_avg,
                lead_time_std_dev=lead_time_std_dev,
                demand_avg=demand_avg,
                service_level=service_level
            )
            
            # Calculate safety stock for forecast period
            if forecast_period > 1:
                forecast_safety_stock = advanced_safety_stock * math.sqrt(forecast_period)
            else:
                forecast_safety_stock = advanced_safety_stock
            
            # Prepare results
            result = {
                "safety_stock": round(advanced_safety_stock, 2),
                "forecast_period_safety_stock": round(forecast_safety_stock, 2),
                "basic_safety_stock": round(basic_safety_stock, 2),
                "demand_stats": {
                    "average": round(demand_avg, 2),
                    "std_dev": round(demand_std_dev, 2),
                    "cv": round(demand_std_dev / demand_avg, 4) if demand_avg > 0 else 0
                },
                "lead_time_stats": {
                    "average": round(lead_time_avg, 2),
                    "std_dev": round(lead_time_std_dev, 2),
                    "cv": round(lead_time_std_dev / lead_time_avg, 4) if lead_time_avg > 0 else 0
                },
                "service_level": service_level,
                "z_score": z_score
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating safety stock from historical data: {str(e)}")
            return {
                "error": str(e),
                "safety_stock": 0
            }
    
    @staticmethod
    async def get_safety_stock_recommendations(
        product_id: str,
        service_level: float = 0.95,
        data_source: Optional[str] = None,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get safety stock recommendations for a product.
        
        Args:
            product_id: Product ID
            service_level: Service level as a decimal (e.g., 0.95 for 95%)
            data_source: Optional data source name
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Dictionary with safety stock recommendations
        """
        try:
            # Get historical data (implementation depends on data sources)
            historical_data = await SafetyStockCalculator._get_historical_data(
                product_id=product_id,
                data_source=data_source,
                client_id=client_id,
                connection_id=connection_id
            )
            
            # Calculate safety stock
            safety_stock_result = SafetyStockCalculator.calculate_safety_stock_from_historical_data(
                historical_demand=historical_data["demand"],
                historical_lead_times=historical_data["lead_times"],
                service_level=service_level
            )
            
            # Add recommendations
            recommendations = SafetyStockCalculator._generate_recommendations(
                safety_stock_result=safety_stock_result,
                historical_data=historical_data,
                product_id=product_id
            )
            
            return {
                **safety_stock_result,
                "recommendations": recommendations,
                "product_id": product_id
            }
            
        except Exception as e:
            logger.error(f"Error getting safety stock recommendations: {str(e)}")
            return {
                "error": str(e),
                "product_id": product_id,
                "safety_stock": 0,
                "recommendations": ["Error calculating safety stock recommendations."]
            }
    
    @staticmethod
    async def _get_historical_data(
        product_id: str,
        data_source: Optional[str] = None,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get historical demand and lead time data for a product.
        
        Args:
            product_id: Product ID
            data_source: Optional data source name
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Dictionary with historical data
        """
        try:
            # Implementation depends on data sources
            # This is a placeholder implementation
            
            if data_source == "database" and client_id and connection_id:
                # Get data from database
                from app.db.interfaces.product_interface import ProductInterface
                
                # Create interface
                product_interface = ProductInterface(client_id=client_id, connection_id=connection_id)
                
                # Get historical demand
                demand_data = await product_interface.get_historical_demand(product_id=product_id, periods=12)
                
                # Get historical lead times
                lead_time_data = await product_interface.get_historical_lead_times(product_id=product_id, periods=12)
                
                return {
                    "demand": demand_data["values"],
                    "lead_times": lead_time_data["values"],
                    "demand_periods": demand_data["periods"],
                    "lead_time_periods": lead_time_data["periods"]
                }
            
            else:
                # Use mock data for demonstration
                # In a real implementation, you would fetch this from a database or API
                
                # Generate some random data for demonstration
                demand = np.random.normal(loc=100, scale=20, size=12).tolist()
                lead_times = np.random.normal(loc=14, scale=3, size=12).tolist()
                
                # Ensure no negative values
                demand = [max(0, d) for d in demand]
                lead_times = [max(1, lt) for lt in lead_times]
                
                # Generate period labels (last 12 months)
                today = datetime.now()
                periods = []
                for i in range(12, 0, -1):
                    month = today.month - (i % 12)
                    year = today.year - (i // 12)
                    if month <= 0:
                        month += 12
                        year -= 1
                    periods.append(f"{year}-{month:02d}")
                
                return {
                    "demand": demand,
                    "lead_times": lead_times,
                    "periods": periods,
                    "is_mock_data": True
                }
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            # Return some reasonable default data to avoid breaking calculations
            return {
                "demand": [100] * 12,
                "lead_times": [14] * 12,
                "periods": ["N/A"] * 12,
                "error": str(e),
                "is_mock_data": True
            }
    
    @staticmethod
    def _generate_recommendations(
        safety_stock_result: Dict[str, Any],
        historical_data: Dict[str, Any],
        product_id: str
    ) -> List[str]:
        """
        Generate recommendations based on safety stock calculation.
        
        Args:
            safety_stock_result: Safety stock calculation result
            historical_data: Historical data
            product_id: Product ID
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Get key values from the safety stock result
            safety_stock = safety_stock_result.get("safety_stock", 0)
            demand_cv = safety_stock_result.get("demand_stats", {}).get("cv", 0)
            lead_time_cv = safety_stock_result.get("lead_time_stats", {}).get("cv", 0)
            
            # Add recommendations based on demand variability
            if demand_cv > 0.5:
                recommendations.append(
                    f"High demand variability detected (CV={demand_cv:.2f}). Consider demand smoothing or forecasting improvements."
                )
                
            if lead_time_cv > 0.3:
                recommendations.append(
                    f"High lead time variability (CV={lead_time_cv:.2f}). Consider working with suppliers to improve consistency."
                )
            
            # Add recommendation for safety stock implementation
            recommendations.append(
                f"Recommended safety stock level is {safety_stock:.2f} units based on calculated service level."
            )
            
            # Add recommendation for review frequency
            if demand_cv > 0.3 or lead_time_cv > 0.2:
                recommendations.append(
                    "Due to high variability, review safety stock levels monthly."
                )
            else:
                recommendations.append(
                    "Variability is within reasonable levels. Review safety stock quarterly."
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating safety stock recommendations."]