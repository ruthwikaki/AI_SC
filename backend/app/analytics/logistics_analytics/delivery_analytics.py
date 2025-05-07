"""
Delivery Analytics Module

This module provides functionality for analyzing delivery performance,
tracking metrics, and identifying improvement opportunities.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import math

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class DeliveryAnalytics:
    """
    Analyzes delivery performance and identifies improvement opportunities.
    """
    
    # Performance threshold levels
    PERFORMANCE_THRESHOLDS = {
        "on_time_delivery": {
            "excellent": 95,
            "good": 90,
            "average": 85,
            "below_average": 80,
            "poor": 0
        },
        "delivery_accuracy": {
            "excellent": 98,
            "good": 95,
            "average": 92,
            "below_average": 90,
            "poor": 0
        },
        "order_cycle_time": {
            "excellent": 0,  # Lower is better, thresholds are reversed
            "good": 1.1,
            "average": 1.2,
            "below_average": 1.4,
            "poor": float('inf')
        }
    }
    
    def __init__(self):
        """Initialize the delivery analytics module."""
        pass
    
    async def analyze_delivery_performance(
        self,
        time_period: str = "last_30_days",
        group_by: str = "none",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze delivery performance metrics.
        
        Args:
            time_period: Time period for analysis
            group_by: Group by dimension (none, carrier, region, customer, product)
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with delivery performance analysis
        """
        try:
            # Get delivery data
            delivery_data = await self._get_delivery_data(
                time_period=time_period,
                client_id=client_id,
                connection_id=connection_id,
                filters=filters
            )
            
            # Calculate summary metrics
            summary_metrics = self._calculate_summary_metrics(delivery_data)
            
            # Group data if requested
            grouped_metrics = {}
            if group_by != "none" and group_by in delivery_data:
                grouped_metrics = self._group_delivery_data(
                    delivery_data=delivery_data,
                    group_by=group_by
                )
            
            # Calculate trends
            trends = self._calculate_trends(delivery_data)
            
            # Generate insights
            insights = self._generate_insights(
                summary_metrics=summary_metrics,
                grouped_metrics=grouped_metrics,
                trends=trends,
                group_by=group_by
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                summary_metrics=summary_metrics,
                grouped_metrics=grouped_metrics,
                trends=trends,
                insights=insights
            )
            
            # Prepare result
            result = {
                "time_period": time_period,
                "summary_metrics": summary_metrics,
                "trends": trends,
                "insights": insights,
                "recommendations": recommendations
            }
            
            if grouped_metrics:
                result["grouped_metrics"] = grouped_metrics
                result["group_by"] = group_by
            
            if filters:
                result["filters"] = filters
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing delivery performance: {str(e)}")
            return {
                "error": str(e),
                "time_period": time_period,
                "summary_metrics": {}
            }
    
    async def analyze_delivery_exceptions(
        self,
        time_period: str = "last_30_days",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze delivery exceptions and issues.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with delivery exceptions analysis
        """
        try:
            # Get delivery exception data
            exception_data = await self._get_delivery_exceptions(
                time_period=time_period,
                client_id=client_id,
                connection_id=connection_id,
                filters=filters
            )
            
            # Calculate exception metrics
            exception_metrics = self._calculate_exception_metrics(exception_data)
            
            # Group exceptions by type
            exceptions_by_type = self._group_exceptions_by_type(exception_data)
            
            # Group exceptions by cause
            exceptions_by_cause = self._group_exceptions_by_cause(exception_data)
            
            # Calculate exception trends
            exception_trends = self._calculate_exception_trends(exception_data)
            
            # Generate insights
            insights = self._generate_exception_insights(
                exception_metrics=exception_metrics,
                exceptions_by_type=exceptions_by_type,
                exceptions_by_cause=exceptions_by_cause,
                exception_trends=exception_trends
            )
            
            # Generate recommendations
            recommendations = self._generate_exception_recommendations(
                exception_metrics=exception_metrics,
                exceptions_by_type=exceptions_by_type,
                exceptions_by_cause=exceptions_by_cause,
                exception_trends=exception_trends,
                insights=insights
            )
            
            # Prepare result
            result = {
                "time_period": time_period,
                "exception_metrics": exception_metrics,
                "exceptions_by_type": exceptions_by_type,
                "exceptions_by_cause": exceptions_by_cause,
                "exception_trends": exception_trends,
                "insights": insights,
                "recommendations": recommendations
            }
            
            if filters:
                result["filters"] = filters
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing delivery exceptions: {str(e)}")
            return {
                "error": str(e),
                "time_period": time_period,
                "exception_metrics": {}
            }
    
    async def analyze_delivery_costs(
        self,
        time_period: str = "last_30_days",
        group_by: str = "none",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze delivery costs.
        
        Args:
            time_period: Time period for analysis
            group_by: Group by dimension (none, carrier, region, customer, product)
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with delivery cost analysis
        """
        try:
           # Get delivery cost data
            cost_data = await self._get_delivery_cost_data(
                time_period=time_period,
                client_id=client_id,
                connection_id=connection_id,
                filters=filters
            )
            
            # Calculate cost metrics
            cost_metrics = self._calculate_cost_metrics(cost_data)
            
            # Group costs if requested
            grouped_costs = {}
            if group_by != "none" and group_by in cost_data:
                grouped_costs = self._group_cost_data(
                    cost_data=cost_data,
                    group_by=group_by
                )
            
            # Calculate cost trends
            cost_trends = self._calculate_cost_trends(cost_data)
            
            # Generate insights
            insights = self._generate_cost_insights(
                cost_metrics=cost_metrics,
                grouped_costs=grouped_costs,
                cost_trends=cost_trends,
                group_by=group_by
            )
            
            # Generate recommendations
            recommendations = self._generate_cost_recommendations(
                cost_metrics=cost_metrics,
                grouped_costs=grouped_costs,
                cost_trends=cost_trends,
                insights=insights
            )
            
            # Prepare result
            result = {
                "time_period": time_period,
                "cost_metrics": cost_metrics,
                "cost_trends": cost_trends,
                "insights": insights,
                "recommendations": recommendations
            }
            
            if grouped_costs:
                result["grouped_costs"] = grouped_costs
                result["group_by"] = group_by
            
            if filters:
                result["filters"] = filters
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing delivery costs: {str(e)}")
            return {
                "error": str(e),
                "time_period": time_period,
                "cost_metrics": {}
            }
    
    async def analyze_last_mile_performance(
        self,
        time_period: str = "last_30_days",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze last mile delivery performance.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with last mile performance analysis
        """
        try:
            # Get last mile delivery data
            last_mile_data = await self._get_last_mile_data(
                time_period=time_period,
                client_id=client_id,
                connection_id=connection_id,
                filters=filters
            )
            
            # Calculate last mile metrics
            last_mile_metrics = self._calculate_last_mile_metrics(last_mile_data)
            
            # Group by region
            performance_by_region = self._group_last_mile_by_region(last_mile_data)
            
            # Calculate last mile trends
            last_mile_trends = self._calculate_last_mile_trends(last_mile_data)
            
            # Generate insights
            insights = self._generate_last_mile_insights(
                last_mile_metrics=last_mile_metrics,
                performance_by_region=performance_by_region,
                last_mile_trends=last_mile_trends
            )
            
            # Generate recommendations
            recommendations = self._generate_last_mile_recommendations(
                last_mile_metrics=last_mile_metrics,
                performance_by_region=performance_by_region,
                last_mile_trends=last_mile_trends,
                insights=insights
            )
            
            # Prepare result
            result = {
                "time_period": time_period,
                "last_mile_metrics": last_mile_metrics,
                "performance_by_region": performance_by_region,
                "last_mile_trends": last_mile_trends,
                "insights": insights,
                "recommendations": recommendations
            }
            
            if filters:
                result["filters"] = filters
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing last mile performance: {str(e)}")
            return {
                "error": str(e),
                "time_period": time_period,
                "last_mile_metrics": {}
            }
    
    async def _get_delivery_data(
        self,
        time_period: str = "last_30_days",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get delivery data for analysis.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with delivery data
        """
        try:
            # Get data from database
            if client_id and connection_id:
                # Import interfaces
                from app.db.interfaces.delivery_interface import DeliveryInterface
                
                # Create interface
                delivery_interface = DeliveryInterface(client_id=client_id, connection_id=connection_id)
                
                # Get delivery data
                delivery_data = await delivery_interface.get_delivery_data(
                    time_period=time_period,
                    filters=filters
                )
                
                return delivery_data
            
            # Generate mock data for demonstration or testing
            delivery_data = self._generate_mock_delivery_data(
                time_period=time_period,
                filters=filters
            )
            
            return delivery_data
            
        except Exception as e:
            logger.error(f"Error getting delivery data: {str(e)}")
            
            # Generate mock data if an error occurs
            return self._generate_mock_delivery_data(
                time_period=time_period,
                filters=filters,
                error=True
            )
    
    async def _get_delivery_exceptions(
        self,
        time_period: str = "last_30_days",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get delivery exception data for analysis.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with delivery exception data
        """
        try:
            # Get data from database
            if client_id and connection_id:
                # Import interfaces
                from app.db.interfaces.delivery_interface import DeliveryInterface
                
                # Create interface
                delivery_interface = DeliveryInterface(client_id=client_id, connection_id=connection_id)
                
                # Get exception data
                exception_data = await delivery_interface.get_exception_data(
                    time_period=time_period,
                    filters=filters
                )
                
                return exception_data
            
            # Generate mock data for demonstration or testing
            exception_data = self._generate_mock_exception_data(
                time_period=time_period,
                filters=filters
            )
            
            return exception_data
            
        except Exception as e:
            logger.error(f"Error getting delivery exception data: {str(e)}")
            
            # Generate mock data if an error occurs
            return self._generate_mock_exception_data(
                time_period=time_period,
                filters=filters,
                error=True
            )
    
    async def _get_delivery_cost_data(
        self,
        time_period: str = "last_30_days",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get delivery cost data for analysis.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with delivery cost data
        """
        try:
            # Get data from database
            if client_id and connection_id:
                # Import interfaces
                from app.db.interfaces.delivery_interface import DeliveryInterface
                
                # Create interface
                delivery_interface = DeliveryInterface(client_id=client_id, connection_id=connection_id)
                
                # Get cost data
                cost_data = await delivery_interface.get_cost_data(
                    time_period=time_period,
                    filters=filters
                )
                
                return cost_data
            
            # Generate mock data for demonstration or testing
            cost_data = self._generate_mock_cost_data(
                time_period=time_period,
                filters=filters
            )
            
            return cost_data
            
        except Exception as e:
            logger.error(f"Error getting delivery cost data: {str(e)}")
            
            # Generate mock data if an error occurs
            return self._generate_mock_cost_data(
                time_period=time_period,
                filters=filters,
                error=True
            )
    
    async def _get_last_mile_data(
        self,
        time_period: str = "last_30_days",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get last mile delivery data for analysis.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with last mile delivery data
        """
        try:
            # Get data from database
            if client_id and connection_id:
                # Import interfaces
                from app.db.interfaces.delivery_interface import DeliveryInterface
                
                # Create interface
                delivery_interface = DeliveryInterface(client_id=client_id, connection_id=connection_id)
                
                # Get last mile data
                last_mile_data = await delivery_interface.get_last_mile_data(
                    time_period=time_period,
                    filters=filters
                )
                
                return last_mile_data
            
            # Generate mock data for demonstration or testing
            last_mile_data = self._generate_mock_last_mile_data(
                time_period=time_period,
                filters=filters
            )
            
            return last_mile_data
            
        except Exception as e:
            logger.error(f"Error getting last mile delivery data: {str(e)}")
            
            # Generate mock data if an error occurs
            return self._generate_mock_last_mile_data(
                time_period=time_period,
                filters=filters,
                error=True
            )
    
    def _calculate_summary_metrics(
        self,
        delivery_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate summary metrics from delivery data.
        
        Args:
            delivery_data: Delivery data
            
        Returns:
            Dictionary with summary metrics
        """
        summary_metrics = {}
        
        # Extract key metrics
        if "deliveries" in delivery_data:
            deliveries = delivery_data["deliveries"]
            
            # On-time delivery rate
            if "total" in deliveries and "on_time" in deliveries:
                total_deliveries = deliveries["total"]
                on_time_deliveries = deliveries["on_time"]
                
                on_time_rate = (on_time_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
                
                summary_metrics["on_time_delivery"] = {
                    "value": round(on_time_rate, 1),
                    "total_deliveries": total_deliveries,
                    "on_time_deliveries": on_time_deliveries,
                    "unit": "%",
                    "performance_level": self._get_performance_level(
                        "on_time_delivery", on_time_rate
                    )
                }
            
            # Delivery accuracy
            if "total" in deliveries and "accurate" in deliveries:
                total_deliveries = deliveries["total"]
                accurate_deliveries = deliveries["accurate"]
                
                accuracy_rate = (accurate_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
                
                summary_metrics["delivery_accuracy"] = {
                    "value": round(accuracy_rate, 1),
                    "total_deliveries": total_deliveries,
                    "accurate_deliveries": accurate_deliveries,
                    "unit": "%",
                    "performance_level": self._get_performance_level(
                        "delivery_accuracy", accuracy_rate
                    )
                }
        
        # Cycle time metrics
        if "cycle_times" in delivery_data:
            cycle_times = delivery_data["cycle_times"]
            
            # Order cycle time
            if "avg_actual" in cycle_times and "avg_target" in cycle_times:
                avg_actual = cycle_times["avg_actual"]
                avg_target = cycle_times["avg_target"]
                
                # Calculate ratio of actual to target
                cycle_time_ratio = avg_actual / avg_target if avg_target > 0 else 1
                
                summary_metrics["order_cycle_time"] = {
                    "value": round(avg_actual, 1),
                    "target": round(avg_target, 1),
                    "ratio": round(cycle_time_ratio, 2),
                    "unit": "days",
                    "performance_level": self._get_performance_level(
                        "order_cycle_time", cycle_time_ratio
                    )
                }
        
        # Cost metrics
        if "costs" in delivery_data:
            costs = delivery_data["costs"]
            
            # Cost per delivery
            if "total_cost" in costs and "total_deliveries" in costs:
                total_cost = costs["total_cost"]
                total_deliveries = costs["total_deliveries"]
                
                cost_per_delivery = total_cost / total_deliveries if total_deliveries > 0 else 0
                
                summary_metrics["cost_per_delivery"] = {
                    "value": round(cost_per_delivery, 2),
                    "total_cost": round(total_cost, 2),
                    "total_deliveries": total_deliveries,
                    "unit": "$"
                }
        
        # Exception metrics
        if "exceptions" in delivery_data:
            exceptions = delivery_data["exceptions"]
            
            # Exception rate
            if "total_deliveries" in exceptions and "total_exceptions" in exceptions:
                total_deliveries = exceptions["total_deliveries"]
                total_exceptions = exceptions["total_exceptions"]
                
                exception_rate = (total_exceptions / total_deliveries * 100) if total_deliveries > 0 else 0
                
                summary_metrics["exception_rate"] = {
                    "value": round(exception_rate, 1),
                    "total_deliveries": total_deliveries,
                    "total_exceptions": total_exceptions,
                    "unit": "%"
                }
        
        return summary_metrics
    
    def _group_delivery_data(
        self,
        delivery_data: Dict[str, Any],
        group_by: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group delivery data by specified dimension.
        
        Args:
            delivery_data: Delivery data
            group_by: Dimension to group by
            
        Returns:
            Dictionary with grouped metrics
        """
        grouped_metrics = {}
        
        # Extract group data
        if group_by in delivery_data and "by_group" in delivery_data:
            group_data = delivery_data["by_group"].get(group_by, {})
            
            for group_name, group_values in group_data.items():
                # Calculate metrics for this group
                group_metrics = {}
                
                # On-time delivery rate
                if "total_deliveries" in group_values and "on_time_deliveries" in group_values:
                    total_deliveries = group_values["total_deliveries"]
                    on_time_deliveries = group_values["on_time_deliveries"]
                    
                    on_time_rate = (on_time_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
                    
                    group_metrics["on_time_delivery"] = {
                        "value": round(on_time_rate, 1),
                        "total_deliveries": total_deliveries,
                        "on_time_deliveries": on_time_deliveries,
                        "unit": "%",
                        "performance_level": self._get_performance_level(
                            "on_time_delivery", on_time_rate
                        )
                    }
                
                # Delivery accuracy
                if "total_deliveries" in group_values and "accurate_deliveries" in group_values:
                    total_deliveries = group_values["total_deliveries"]
                    accurate_deliveries = group_values["accurate_deliveries"]
                    
                    accuracy_rate = (accurate_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
                    
                    group_metrics["delivery_accuracy"] = {
                        "value": round(accuracy_rate, 1),
                        "total_deliveries": total_deliveries,
                        "accurate_deliveries": accurate_deliveries,
                        "unit": "%",
                        "performance_level": self._get_performance_level(
                            "delivery_accuracy", accuracy_rate
                        )
                    }
                
                # Cost per delivery
                if "total_cost" in group_values and "total_deliveries" in group_values:
                    total_cost = group_values["total_cost"]
                    total_deliveries = group_values["total_deliveries"]
                    
                    cost_per_delivery = total_cost / total_deliveries if total_deliveries > 0 else 0
                    
                    group_metrics["cost_per_delivery"] = {
                        "value": round(cost_per_delivery, 2),
                        "total_cost": round(total_cost, 2),
                        "total_deliveries": total_deliveries,
                        "unit": "$"
                    }
                
                # Add group metrics
                grouped_metrics[group_name] = group_metrics
        
        return grouped_metrics
    
    def _calculate_trends(
        self,
        delivery_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate trends from delivery data.
        
        Args:
            delivery_data: Delivery data
            
        Returns:
            Dictionary with trend data
        """
        trends = {}
        
        # Extract trend data
        if "trends" in delivery_data:
            trend_data = delivery_data["trends"]
            
            # On-time delivery trend
            if "on_time_delivery" in trend_data:
                on_time_trend = trend_data["on_time_delivery"]
                
                if "values" in on_time_trend and "periods" in on_time_trend:
                    values = on_time_trend["values"]
                    periods = on_time_trend["periods"]
                    
                    # Calculate trend direction and magnitude
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values)
                    
                    trends["on_time_delivery"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # Delivery accuracy trend
            if "delivery_accuracy" in trend_data:
                accuracy_trend = trend_data["delivery_accuracy"]
                
                if "values" in accuracy_trend and "periods" in accuracy_trend:
                    values = accuracy_trend["values"]
                    periods = accuracy_trend["periods"]
                    
                    # Calculate trend direction and magnitude
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values)
                    
                    trends["delivery_accuracy"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # Order cycle time trend
            if "order_cycle_time" in trend_data:
                cycle_time_trend = trend_data["order_cycle_time"]
                
                if "values" in cycle_time_trend and "periods" in cycle_time_trend:
                    values = cycle_time_trend["values"]
                    periods = cycle_time_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for cycle time, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    trends["order_cycle_time"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # Cost per delivery trend
            if "cost_per_delivery" in trend_data:
                cost_trend = trend_data["cost_per_delivery"]
                
                if "values" in cost_trend and "periods" in cost_trend:
                    values = cost_trend["values"]
                    periods = cost_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for cost, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    trends["cost_per_delivery"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
        
        return trends
    
    def _calculate_trend_direction(
        self,
        values: List[float],
        decreasing_is_positive: bool = False
    ) -> Tuple[str, float]:
        """
        Calculate trend direction and magnitude.
        
        Args:
            values: List of metric values
            decreasing_is_positive: Whether decreasing trend is positive
            
        Returns:
            Tuple of (direction, magnitude)
        """
        if not values or len(values) < 2:
            return "stable", 0.0
        
        # Calculate simple linear regression
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Calculate percentage change
        first_value = values[0]
        last_value = values[-1]
        
        percent_change = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0
        
        # Determine direction
        if abs(percent_change) < 2:
            direction = "stable"
        elif percent_change > 0:
            direction = "improving" if not decreasing_is_positive else "declining"
        else:
            direction = "declining" if not decreasing_is_positive else "improving"
        
        return direction, abs(percent_change)
    
    def _generate_insights(
        self,
        summary_metrics: Dict[str, Any],
        grouped_metrics: Dict[str, Dict[str, Any]],
        trends: Dict[str, Any],
        group_by: str
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from delivery metrics.
        
        Args:
            summary_metrics: Summary metrics
            grouped_metrics: Grouped metrics
            trends: Trend data
            group_by: Group by dimension
            
        Returns:
            List of insights
        """
        insights = []
        
        # Generate overall performance insights
        if "on_time_delivery" in summary_metrics:
            on_time_value = summary_metrics["on_time_delivery"]["value"]
            performance_level = summary_metrics["on_time_delivery"]["performance_level"]
            
            if performance_level in ("excellent", "good"):
                insights.append({
                    "type": "positive",
                    "category": "on_time_delivery",
                    "insight": f"Strong on-time delivery performance at {on_time_value}%"
                })
            elif performance_level in ("below_average", "poor"):
                insights.append({
                    "type": "negative",
                    "category": "on_time_delivery",
                    "insight": f"Poor on-time delivery performance at {on_time_value}%"
                })
        
        if "delivery_accuracy" in summary_metrics:
            accuracy_value = summary_metrics["delivery_accuracy"]["value"]
            performance_level = summary_metrics["delivery_accuracy"]["performance_level"]
            
            if performance_level in ("excellent", "good"):
                insights.append({
                    "type": "positive",
                    "category": "delivery_accuracy",
                    "insight": f"High delivery accuracy at {accuracy_value}%"
                })
            elif performance_level in ("below_average", "poor"):
                insights.append({
                    "type": "negative",
                    "category": "delivery_accuracy",
                    "insight": f"Low delivery accuracy at {accuracy_value}%"
                })
        
        if "order_cycle_time" in summary_metrics:
            cycle_time = summary_metrics["order_cycle_time"]["value"]
            target = summary_metrics["order_cycle_time"]["target"]
            ratio = summary_metrics["order_cycle_time"]["ratio"]
            performance_level = summary_metrics["order_cycle_time"]["performance_level"]
            
            if performance_level in ("excellent", "good"):
                insights.append({
                    "type": "positive",
                    "category": "order_cycle_time",
                    "insight": f"Fast order cycle time at {cycle_time} days ({ratio:.2f}x target)"
                })
            elif performance_level in ("below_average", "poor"):
                insights.append({
                    "type": "negative",
                    "category": "order_cycle_time",
                    "insight": f"Slow order cycle time at {cycle_time} days ({ratio:.2f}x target)"
                })
        
        # Generate trend insights
        if "on_time_delivery" in trends:
            direction = trends["on_time_delivery"]["direction"]
            magnitude = trends["on_time_delivery"]["magnitude"]
            
            if direction == "improving" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "on_time_delivery",
                    "insight": f"On-time delivery improving by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "on_time_delivery",
                    "insight": f"On-time delivery declining by {magnitude:.1f}%"
                })
        
        if "delivery_accuracy" in trends:
            direction = trends["delivery_accuracy"]["direction"]
            magnitude = trends["delivery_accuracy"]["magnitude"]
            
            if direction == "improving" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "delivery_accuracy",
                    "insight": f"Delivery accuracy improving by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "delivery_accuracy",
                    "insight": f"Delivery accuracy declining by {magnitude:.1f}%"
                })
        
        if "order_cycle_time" in trends:
            direction = trends["order_cycle_time"]["direction"]
            magnitude = trends["order_cycle_time"]["magnitude"]
            
            if direction == "improving" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "order_cycle_time",
                    "insight": f"Order cycle time improving by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "order_cycle_time",
                    "insight": f"Order cycle time declining by {magnitude:.1f}%"
                })
        
        # Generate grouped insights
        if grouped_metrics and group_by != "none":
            # Find best and worst performers
            best_performers = {}
            worst_performers = {}
            
            for metric in ["on_time_delivery", "delivery_accuracy", "cost_per_delivery"]:
                if not any(metric in group_metrics for group_metrics in grouped_metrics.values()):
                    continue
                
                # Find groups with this metric
                groups_with_metric = {
                    group_name: group_metrics[metric]["value"]
                    for group_name, group_metrics in grouped_metrics.items()
                    if metric in group_metrics
                }
                
                if not groups_with_metric:
                    continue
                
                # Find best performer
                best_value = max(groups_with_metric.values()) if metric != "cost_per_delivery" else min(groups_with_metric.values())
                best_group = next(
                    group_name for group_name, value in groups_with_metric.items()
                    if value == best_value
                )
                
                best_performers[metric] = {
                    "group": best_group,
                    "value": best_value
                }
                
                # Find worst performer
                worst_value = min(groups_with_metric.values()) if metric != "cost_per_delivery" else max(groups_with_metric.values())
                worst_group = next(
                    group_name for group_name, value in groups_with_metric.items()
                    if value == worst_value
                )
                
                worst_performers[metric] = {
                    "group": worst_group,
                    "value": worst_value
                }
            
            # Add insights for best performers
            for metric, performer in best_performers.items():
                if metric == "on_time_delivery":
                    insights.append({
                        "type": "group",
                        "category": "on_time_delivery",
                        "insight": f"Best on-time delivery: {performer['group']} at {performer['value']}%"
                    })
                elif metric == "delivery_accuracy":
                    insights.append({
                        "type": "group",
                        "category": "delivery_accuracy",
                        "insight": f"Best delivery accuracy: {performer['group']} at {performer['value']}%"
                    })
                elif metric == "cost_per_delivery":
                    insights.append({
                        "type": "group",
                        "category": "cost",
                        "insight": f"Lowest cost per delivery: {performer['group']} at ${performer['value']}"
                    })
            
            # Add insights for worst performers
            for metric, performer in worst_performers.items():
                if metric == "on_time_delivery" and performer["value"] < 80:
                    insights.append({
                        "type": "group",
                        "category": "on_time_delivery",
                        "insight": f"Worst on-time delivery: {performer['group']} at {performer['value']}%"
                    })
                elif metric == "delivery_accuracy" and performer["value"] < 90:
                    insights.append({
                        "type": "group",
                        "category": "delivery_accuracy",
                        "insight": f"Worst delivery accuracy: {performer['group']} at {performer['value']}%"
                    })
                elif metric == "cost_per_delivery":
                    # Only add if the cost is significantly higher than average
                    if "cost_per_delivery" in summary_metrics:
                        avg_cost = summary_metrics["cost_per_delivery"]["value"]
                        if performer["value"] > avg_cost * 1.2:  # 20% higher than average
                            insights.append({
                                "type": "group",
                                "category": "cost",
                                "insight": f"Highest cost per delivery: {performer['group']} at ${performer['value']}"
                            })
        
        return insights
    
    def _generate_recommendations(
        self,
        summary_metrics: Dict[str, Any],
        grouped_metrics: Dict[str, Dict[str, Any]],
        trends: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on delivery metrics and insights.
        
        Args:
            summary_metrics: Summary metrics
            grouped_metrics: Grouped metrics
            trends: Trend data
            insights: Generated insights
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate recommendations based on overall performance
        if "on_time_delivery" in summary_metrics:
            on_time_value = summary_metrics["on_time_delivery"]["value"]
            performance_level = summary_metrics["on_time_delivery"]["performance_level"]
            
            if performance_level in ("below_average", "poor"):
                recommendations.append({
                    "priority": "high",
                    "category": "on_time_delivery",
                    "recommendation": "Implement daily carrier performance reviews to address on-time delivery issues",
                    "expected_impact": "Potential improvement of 5-10% in on-time delivery rate"
                })
        
        if "delivery_accuracy" in summary_metrics:
            accuracy_value = summary_metrics["delivery_accuracy"]["value"]
            performance_level = summary_metrics["delivery_accuracy"]["performance_level"]
            
            if performance_level in ("below_average", "poor"):
                recommendations.append({
                    "priority": "high",
                    "category": "delivery_accuracy",
                    "recommendation": "Implement barcode scanning at all handoff points to improve delivery accuracy",
                    "expected_impact": "Potential improvement of 3-8% in delivery accuracy"
                })
        
        if "order_cycle_time" in summary_metrics:
            cycle_time = summary_metrics["order_cycle_time"]["value"]
            target = summary_metrics["order_cycle_time"]["target"]
            ratio = summary_metrics["order_cycle_time"]["ratio"]
            performance_level = summary_metrics["order_cycle_time"]["performance_level"]
            
            if performance_level in ("below_average", "poor"):
                recommendations.append({
                    "priority": "medium",
                    "category": "order_cycle_time",
                    "recommendation": "Review and optimize order processing workflows to reduce cycle time",
                    "expected_impact": "Potential reduction of 10-15% in order cycle time"
                })
        
        # Generate recommendations based on trends
        for metric, trend_data in trends.items():
            direction = trend_data["direction"]
            magnitude = trend_data["magnitude"]
            
            if direction == "declining" and magnitude >= 5:
                if metric == "on_time_delivery":
                    recommendations.append({
                        "priority": "high",
                        "category": "on_time_delivery",
                        "recommendation": "Investigate root causes of declining on-time delivery performance",
                        "expected_impact": "Identify and address factors causing the negative trend"
                    })
                elif metric == "delivery_accuracy":
                    recommendations.append({
                        "priority": "high",
                        "category": "delivery_accuracy",
                        "recommendation": "Audit delivery processes to identify causes of declining accuracy",
                        "expected_impact": "Identify and address factors causing the negative trend"
                    })
                elif metric == "order_cycle_time" and direction == "declining":  # Note: for cycle time, declining means increasing time
                    recommendations.append({
                        "priority": "medium",
                        "category": "order_cycle_time",
                        "recommendation": "Analyze bottlenecks in order fulfillment process",
                        "expected_impact": "Identify and address delays in the process"
                    })
        
        # Generate recommendations based on grouped insights
        negative_group_insights = [
            insight for insight in insights
            if insight["type"] == "group" and "Worst" in insight["insight"]
        ]
        
        for insight in negative_group_insights:
            category = insight["category"]
            # Extract group name from insight text
            group_name = insight["insight"].split(": ")[1].split(" at ")[0]
            
            if category == "on_time_delivery":
                recommendations.append({
                    "priority": "medium",
                    "category": "on_time_delivery",
                    "recommendation": f"Develop improvement plan for {group_name} to address on-time delivery performance",
                    "expected_impact": "Target 5-10% improvement in on-time delivery for this group"
                })
            elif category == "delivery_accuracy":
                recommendations.append({
                    "priority": "medium",
                    "category": "delivery_accuracy",
                    "recommendation": f"Provide additional training and resources for {group_name} to improve delivery accuracy",
                    "expected_impact": "Target 5-8% improvement in delivery accuracy for this group"
                })
            elif category == "cost":
                recommendations.append({
                    "priority": "medium",
                    "category": "cost",
                    "recommendation": f"Review delivery costs for {group_name} to identify cost reduction opportunities",
                    "expected_impact": "Potential 10-15% cost reduction for this group"
                })
        
        # Add general recommendations
        recommendations.append({
            "priority": "low",
            "category": "general",
            "recommendation": "Implement regular delivery performance reviews with all carriers",
            "expected_impact": "Improved visibility and accountability across delivery operations"
        })
        
        recommendations.append({
            "priority": "low",
            "category": "general",
            "recommendation": "Enhance delivery tracking and visibility for customers",
            "expected_impact": "Improved customer experience and reduced WISMO (Where Is My Order) inquiries"
        })
        
        return recommendations
    
    def _calculate_exception_metrics(
        self,
        exception_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate exception metrics from delivery exception data.
        
        Args:
            exception_data: Delivery exception data
            
        Returns:
            Dictionary with exception metrics
        """
        exception_metrics = {}
        
        # Extract key metrics
        if "summary" in exception_data:
            summary = exception_data["summary"]
            
            # Overall exception rate
            if "total_deliveries" in summary and "total_exceptions" in summary:
                total_deliveries = summary["total_deliveries"]
                total_exceptions = summary["total_exceptions"]
                
                exception_rate = (total_exceptions / total_deliveries * 100) if total_deliveries > 0 else 0
                
                exception_metrics["exception_rate"] = {
                    "value": round(exception_rate, 1),
                    "total_deliveries": total_deliveries,
                    "total_exceptions": total_exceptions,
                    "unit": "%"
                }
            
            # Average resolution time
            if "avg_resolution_time" in summary:
                avg_resolution_time = summary["avg_resolution_time"]
                
                exception_metrics["avg_resolution_time"] = {
                    "value": round(avg_resolution_time, 1),
                    "unit": "hours"
                }
            
            # Unresolved exceptions
            if "unresolved_exceptions" in summary and "total_exceptions" in summary:
                unresolved_exceptions = summary["unresolved_exceptions"]
                total_exceptions = summary["total_exceptions"]
                
                unresolved_rate = (unresolved_exceptions / total_exceptions * 100) if total_exceptions > 0 else 0
                
                exception_metrics["unresolved_rate"] = {
                    "value": round(unresolved_rate, 1),
                    "unresolved_exceptions": unresolved_exceptions,
                    "total_exceptions": total_exceptions,
                    "unit": "%"
                }
            
            # Customer impact metrics
            if "customer_impact" in summary:
                customer_impact = summary["customer_impact"]
                
                exception_metrics["customer_impact"] = customer_impact
        
        return exception_metrics
    
    def _group_exceptions_by_type(
        self,
        exception_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group exceptions by type.
        
        Args:
            exception_data: Exception data
            
        Returns:
            Dictionary with exceptions grouped by type
        """
        grouped_exceptions = {}
        
        # Extract exception types
        if "by_type" in exception_data:
            by_type = exception_data["by_type"]
            
            for exception_type, type_data in by_type.items():
                # Extract key metrics for this type
                type_metrics = {}
                
                if "count" in type_data and "total_exceptions" in type_data:
                    count = type_data["count"]
                    total_exceptions = type_data["total_exceptions"]
                    
                    percentage = (count / total_exceptions * 100) if total_exceptions > 0 else 0
                    
                    type_metrics["count"] = count
                    type_metrics["percentage"] = round(percentage, 1)
                    type_metrics["unit"] = "%"
                
                if "avg_resolution_time" in type_data:
                    type_metrics["avg_resolution_time"] = round(type_data["avg_resolution_time"], 1)
                
                if "description" in type_data:
                    type_metrics["description"] = type_data["description"]
                
                grouped_exceptions[exception_type] = type_metrics
        
        return grouped_exceptions
    
    def _group_exceptions_by_cause(
        self,
        exception_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group exceptions by root cause.
        
        Args:
            exception_data: Exception data
            
        Returns:
            Dictionary with exceptions grouped by cause
        """
        grouped_by_cause = {}
        
        # Extract exception causes
        if "by_cause" in exception_data:
            by_cause = exception_data["by_cause"]
            
            for cause, cause_data in by_cause.items():
                # Extract key metrics for this cause
                cause_metrics = {}
                
                if "count" in cause_data and "total_exceptions" in cause_data:
                    count = cause_data["count"]
                    total_exceptions = cause_data["total_exceptions"]
                    
                    percentage = (count / total_exceptions * 100) if total_exceptions > 0 else 0
                    
                    cause_metrics["count"] = count
                    cause_metrics["percentage"] = round(percentage, 1)
                    cause_metrics["unit"] = "%"
                
                if "description" in cause_data:
                    cause_metrics["description"] = cause_data["description"]
                
                if "preventable" in cause_data:
                    cause_metrics["preventable"] = cause_data["preventable"]
                
                grouped_by_cause[cause] = cause_metrics
        
        return grouped_by_cause
    
    def _calculate_exception_trends(
        self,
        exception_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate exception trends.
        
        Args:
            exception_data: Exception data
            
        Returns:
            Dictionary with exception trends
        """
        exception_trends = {}
        
        # Extract trend data
        if "trends" in exception_data:
            trend_data = exception_data["trends"]
            
            # Exception rate trend
            if "exception_rate" in trend_data:
                rate_trend = trend_data["exception_rate"]
                
                if "values" in rate_trend and "periods" in rate_trend:
                    values = rate_trend["values"]
                    periods = rate_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for exception rate, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    exception_trends["exception_rate"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # Resolution time trend
            if "resolution_time" in trend_data:
                time_trend = trend_data["resolution_time"]
                
                if "values" in time_trend and "periods" in time_trend:
                    values = time_trend["values"]
                    periods = time_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for resolution time, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    exception_trends["resolution_time"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
        
        return exception_trends
    
    def _generate_exception_insights(
        self,
        exception_metrics: Dict[str, Any],
        exceptions_by_type: Dict[str, Dict[str, Any]],
        exceptions_by_cause: Dict[str, Dict[str, Any]],
        exception_trends: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from exception data.
        
        Args:
            exception_metrics: Exception metrics
            exceptions_by_type: Exceptions grouped by type
            exceptions_by_cause: Exceptions grouped by cause
            exception_trends: Exception trends
            
        Returns:
            List of insights
        """
        insights = []
        
        # Generate insights based on overall metrics
        if "exception_rate" in exception_metrics:
            exception_rate = exception_metrics["exception_rate"]["value"]
            
            if exception_rate > 10:
                insights.append({
                    "type": "negative",
                    "category": "exception_rate",
                    "insight": f"High overall exception rate of {exception_rate}%"
                })
            elif exception_rate < 3:
                insights.append({
                    "type": "positive",
                    "category": "exception_rate",
                    "insight": f"Low overall exception rate of {exception_rate}%"
                })
        
        if "avg_resolution_time" in exception_metrics:
            resolution_time = exception_metrics["avg_resolution_time"]["value"]
            
            if resolution_time > 24:
                insights.append({
                    "type": "negative",
                    "category": "resolution_time",
                    "insight": f"Long average exception resolution time of {resolution_time} hours"
                })
            elif resolution_time < 4:
                insights.append({
                    "type": "positive",
                    "category": "resolution_time",
                    "insight": f"Fast average exception resolution time of {resolution_time} hours"
                })
        
        if "unresolved_rate" in exception_metrics:
            unresolved_rate = exception_metrics["unresolved_rate"]["value"]
            
            if unresolved_rate > 15:
                insights.append({
                    "type": "negative",
                    "category": "unresolved_rate",
                    "insight": f"High rate of unresolved exceptions at {unresolved_rate}%"
                })
            elif unresolved_rate < 5:
                insights.append({
                    "type": "positive",
                    "category": "unresolved_rate",
                    "insight": f"Low rate of unresolved exceptions at {unresolved_rate}%"
                })
        
        # Generate insights about top exception types
        if exceptions_by_type:
            # Sort exception types by count
            sorted_types = sorted(
                exceptions_by_type.items(),
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )
            
            # Get top exception types
            top_types = sorted_types[:3]
            
            for exception_type, type_data in top_types:
                percentage = type_data.get("percentage", 0)
                
                if percentage >= 15:
                    insights.append({
                        "type": "type",
                        "category": "exception_type",
                        "insight": f"'{exception_type}' is a major exception type, representing {percentage}% of all exceptions"
                    })
        
        # Generate insights about top exception causes
        if exceptions_by_cause:
            # Sort exception causes by count
            sorted_causes = sorted(
                exceptions_by_cause.items(),
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )
            
            # Get top preventable causes
            top_preventable_causes = [
                (cause, cause_data) for cause, cause_data in sorted_causes
                if cause_data.get("preventable", False)
            ][:3]
            
            for cause, cause_data in top_preventable_causes:
                percentage = cause_data.get("percentage", 0)
                
                if percentage >= 10:
                    insights.append({
                        "type": "cause",
                        "category": "preventable_cause",
                        "insight": f"'{cause}' is a major preventable cause, accounting for {percentage}% of exceptions"
                    })
        
        # Generate trend insights
        if "exception_rate" in exception_trends:
            direction = exception_trends["exception_rate"]["direction"]
            magnitude = exception_trends["exception_rate"]["magnitude"]
            
            if direction == "improving" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "exception_rate",
                    "insight": f"Exception rate improving (decreasing) by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "exception_rate",
                    "insight": f"Exception rate declining (increasing) by {magnitude:.1f}%"
                })
        
        if "resolution_time" in exception_trends:
            direction = exception_trends["resolution_time"]["direction"]
            magnitude = exception_trends["resolution_time"]["magnitude"]
            
            if direction == "improving" and magnitude >= 10:
                insights.append({
                    "type": "trend",
                    "category": "resolution_time",
                    "insight": f"Resolution time improving (decreasing) by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 10:
                insights.append({
                    "type": "trend",
                    "category": "resolution_time",
                    "insight": f"Resolution time declining (increasing) by {magnitude:.1f}%"
                })
        
        return insights
    
    def _generate_exception_recommendations(
        self,
        exception_metrics: Dict[str, Any],
        exceptions_by_type: Dict[str, Dict[str, Any]],
        exceptions_by_cause: Dict[str, Dict[str, Any]],
        exception_trends: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on exception data.
        
        Args:
            exception_metrics: Exception metrics
            exceptions_by_type: Exceptions grouped by type
            exceptions_by_cause: Exceptions grouped by cause
            exception_trends: Exception trends
            insights: Generated insights
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate recommendations based on overall metrics
        if "exception_rate" in exception_metrics:
            exception_rate = exception_metrics["exception_rate"]["value"]
            
            if exception_rate > 8:
                recommendations.append({
                    "priority": "high",
                    "category": "exception_rate",
                    "recommendation": "Implement daily exception review process to address high exception rate",
                    "expected_impact": "Potential reduction of 20-30% in exception rate"
                })
        
        if "avg_resolution_time" in exception_metrics:
            resolution_time = exception_metrics["avg_resolution_time"]["value"]
            
            if resolution_time > 12:
                recommendations.append({
                    "priority": "medium",
                    "category": "resolution_time",
                    "recommendation": "Establish Service Level Agreements (SLAs) for exception resolution",
                    "expected_impact": "Reduce average resolution time by 30-40%"
                })
        
        if "unresolved_rate" in exception_metrics:
            unresolved_rate = exception_metrics["unresolved_rate"]["value"]
            
            if unresolved_rate > 10:
                recommendations.append({
                    "priority": "high",
                    "category": "unresolved_rate",
                    "recommendation": "Implement escalation process for exceptions unresolved after 24 hours",
                    "expected_impact": "Reduce unresolved exception rate by 50-60%"
                })
        
        # Generate recommendations based on top exception types
        if exceptions_by_type:
            # Sort exception types by count
            sorted_types = sorted(
                exceptions_by_type.items(),
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )
            
            # Get top exception type
            if sorted_types:
                top_type, top_type_data = sorted_types[0]
                percentage = top_type_data.get("percentage", 0)
                
                if percentage >= 20:
                    recommendations.append({
                        "priority": "high",
                        "category": "exception_type",
                        "recommendation": f"Develop specific action plan to address '{top_type}' exceptions",
                        "expected_impact": f"Potential reduction of 30-40% in '{top_type}' exceptions"
                    })
        
        # Generate recommendations based on preventable causes
        if exceptions_by_cause:
            # Find preventable causes
            preventable_causes = [
                (cause, cause_data) for cause, cause_data in exceptions_by_cause.items()
                if cause_data.get("preventable", False)
            ]
            
            # Sort by count
            sorted_preventable = sorted(
                preventable_causes,
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )
            
            # Recommend addressing top preventable causes
            for cause, cause_data in sorted_preventable[:2]:
                percentage = cause_data.get("percentage", 0)
                
                if percentage >= 5:
                    recommendations.append({
                        "priority": "medium",
                        "category": "preventable_cause",
                        "recommendation": f"Implement process improvements to address '{cause}' exceptions",
                        "expected_impact": f"Potential elimination of up to 80% of '{cause}' exceptions"
                    })
        
        # Generate recommendations based on trends
        if "exception_rate" in exception_trends:
            direction = exception_trends["exception_rate"]["direction"]
            
            if direction == "declining":
                recommendations.append({
                    "priority": "high",
                    "category": "exception_rate",
                    "recommendation": "Investigate root causes of increasing exception rate",
                    "expected_impact": "Identify and address factors causing the negative trend"
                })
        
        # Add general recommendations
        recommendations.append({
            "priority": "low",
            "category": "general",
            "recommendation": "Implement exception prevention training for warehouse and delivery staff",
            "expected_impact": "Improve awareness and reduce preventable exceptions"
        })
        
        recommendations.append({
            "priority": "low",
            "category": "general",
            "recommendation": "Develop exception resolution playbooks for common exception types",
            "expected_impact": "Standardize resolution process and reduce resolution time"
        })
        
        return recommendations
    
    def _calculate_cost_metrics(
        self,
        cost_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate cost metrics from delivery cost data.
        
        Args:
            cost_data: Delivery cost data
            
        Returns:
            Dictionary with cost metrics
        """
        cost_metrics = {}
        
        # Extract key metrics
        if "summary" in cost_data:
            summary = cost_data["summary"]
            
            # Cost per delivery
            if "total_cost" in summary and "total_deliveries" in summary:
                total_cost = summary["total_cost"]
                total_deliveries = summary["total_deliveries"]
                
                cost_per_delivery = total_cost / total_deliveries if total_deliveries > 0 else 0
                
                cost_metrics["cost_per_delivery"] = {
                    "value": round(cost_per_delivery, 2),
                    "total_cost": round(total_cost, 2),
                    "total_deliveries": total_deliveries,
                    "unit": "$"
                }
            
            # Cost per mile
            if "total_cost" in summary and "total_miles" in summary:
                total_cost = summary["total_cost"]
                total_miles = summary["total_miles"]
                
                cost_per_mile = total_cost / total_miles if total_miles > 0 else 0
                
                cost_metrics["cost_per_mile"] = {
                    "value": round(cost_per_mile, 2),
                    "total_cost": round(total_cost, 2),
                    "total_miles": total_miles,
                    "unit": "$"
                }
            
            # Cost breakdown
            if "cost_breakdown" in summary:
                cost_metrics["cost_breakdown"] = summary["cost_breakdown"]
        
        return cost_metrics
    
    def _group_cost_data(
        self,
        cost_data: Dict[str, Any],
        group_by: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group cost data by specified dimension.
        
        Args:
            cost_data: Cost data
            group_by: Dimension to group by
            
        Returns:
            Dictionary with grouped cost data
        """
        grouped_costs = {}
        
        # Extract group data
        if group_by in cost_data and "by_group" in cost_data:
            group_data = cost_data["by_group"].get(group_by, {})
            
            for group_name, group_values in group_data.items():
                # Calculate metrics for this group
                group_metrics = {}
                
                # Cost per delivery
                if "total_cost" in group_values and "total_deliveries" in group_values:
                    total_cost = group_values["total_cost"]
                    total_deliveries = group_values["total_deliveries"]
                    
                    cost_per_delivery = total_cost / total_deliveries if total_deliveries > 0 else 0
                    
                    group_metrics["cost_per_delivery"] = {
                        "value": round(cost_per_delivery, 2),
                        "total_cost": round(total_cost, 2),
                        "total_deliveries": total_deliveries,
                        "unit": "$"
                    }
                
                # Cost per mile
                if "total_cost" in group_values and "total_miles" in group_values:
                    total_cost = group_values["total_cost"]
                    total_miles = group_values["total_miles"]
                    
                    cost_per_mile = total_cost / total_miles if total_miles > 0 else 0
                    
                    group_metrics["cost_per_mile"] = {
                        "value": round(cost_per_mile, 2),
                        "total_cost": round(total_cost, 2),
                        "total_miles": total_miles,
                        "unit": "$"
                    }
                
                # Cost breakdown
                if "cost_breakdown" in group_values:
                    group_metrics["cost_breakdown"] = group_values["cost_breakdown"]
                
                grouped_costs[group_name] = group_metrics
        
        return grouped_costs
    
    def _calculate_cost_trends(
        self,
        cost_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate cost trends from delivery cost data.
        
        Args:
            cost_data: Delivery cost data
            
        Returns:
            Dictionary with cost trends
        """
        cost_trends = {}
        
        # Extract trend data
        if "trends" in cost_data:
            trend_data = cost_data["trends"]
            
            # Cost per delivery trend
            if "cost_per_delivery" in trend_data:
                delivery_trend = trend_data["cost_per_delivery"]
                
                if "values" in delivery_trend and "periods" in delivery_trend:
                    values = delivery_trend["values"]
                    periods = delivery_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for cost, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    cost_trends["cost_per_delivery"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # Cost per mile trend
            if "cost_per_mile" in trend_data:
                mile_trend = trend_data["cost_per_mile"]
                
                if "values" in mile_trend and "periods" in mile_trend:
                    values = mile_trend["values"]
                    periods = mile_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for cost, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    cost_trends["cost_per_mile"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
        
        return cost_trends
    
    def _generate_cost_insights(
        self,
        cost_metrics: Dict[str, Any],
        grouped_costs: Dict[str, Dict[str, Any]],
        cost_trends: Dict[str, Any],
        group_by: str
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from delivery cost data.
        
        Args:
            cost_metrics: Cost metrics
            grouped_costs: Grouped cost data
            cost_trends: Cost trends
            group_by: Group by dimension
            
        Returns:
            List of insights
        """
        insights = []
        
        # Generate insights based on overall costs
        if "cost_per_delivery" in cost_metrics:
            cost_per_delivery = cost_metrics["cost_per_delivery"]["value"]
            
            insights.append({
                "type": "metric",
                "category": "cost_per_delivery",
                "insight": f"Average cost per delivery is ${cost_per_delivery}"
            })
        
        if "cost_per_mile" in cost_metrics:
            cost_per_mile = cost_metrics["cost_per_mile"]["value"]
            
            insights.append({
                "type": "metric",
                "category": "cost_per_mile",
                "insight": f"Average cost per mile is ${cost_per_mile}"
            })
        
        # Generate insights based on cost breakdown
        if "cost_breakdown" in cost_metrics:
            breakdown = cost_metrics["cost_breakdown"]
            
            # Find highest cost category
            highest_category = max(breakdown.items(), key=lambda x: x[1]["value"])
            category_name, category_data = highest_category
            percentage = category_data.get("percentage", 0)
            
            if percentage > 40:
                insights.append({
                    "type": "breakdown",
                    "category": "cost_category",
                    "insight": f"'{category_name}' represents {percentage}% of total delivery costs"
                })
        
        # Generate insights based on cost trends
        if "cost_per_delivery" in cost_trends:
            direction = cost_trends["cost_per_delivery"]["direction"]
            magnitude = cost_trends["cost_per_delivery"]["magnitude"]
            
            if direction == "improving" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "cost_per_delivery",
                    "insight": f"Cost per delivery decreasing by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "cost_per_delivery",
                    "insight": f"Cost per delivery increasing by {magnitude:.1f}%"
                })
        
        # Generate insights based on grouped costs
        if grouped_costs and group_by != "none":
            # Find highest and lowest cost groups
            if any("cost_per_delivery" in group_metrics for group_metrics in grouped_costs.values()):
                # Create mapping of group to cost per delivery
                group_costs = {
                    group_name: group_metrics["cost_per_delivery"]["value"]
                    for group_name, group_metrics in grouped_costs.items()
                    if "cost_per_delivery" in group_metrics
                }
                
                if group_costs:
                    # Find highest cost group
                    highest_group = max(group_costs.items(), key=lambda x: x[1])
                    group_name, group_cost = highest_group
                    
                    # Find lowest cost group
                    lowest_group = min(group_costs.items(), key=lambda x: x[1])
                    low_group_name, low_group_cost = lowest_group
                    
                    # Calculate cost difference percentage
                    if low_group_cost > 0:
                        diff_percent = ((group_cost - low_group_cost) / low_group_cost * 100)
                    else:
                        diff_percent = 0
                    
                    if diff_percent >= 30:
                        insights.append({
                            "type": "group",
                            "category": "cost_variation",
                            "insight": f"'{group_name}' has {diff_percent:.1f}% higher cost per delivery than '{low_group_name}'"
                        })
        
        return insights
    
    def _generate_cost_recommendations(
        self,
        cost_metrics: Dict[str, Any],
        grouped_costs: Dict[str, Dict[str, Any]],
        cost_trends: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on delivery cost data.
        
        Args:
            cost_metrics: Cost metrics
            grouped_costs: Grouped cost data
            cost_trends: Cost trends
            insights: Generated insights
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate recommendations based on cost trends
        if "cost_per_delivery" in cost_trends:
            direction = cost_trends["cost_per_delivery"]["direction"]
            magnitude = cost_trends["cost_per_delivery"]["magnitude"]
            
            if direction == "declining" and magnitude >= 8:
                recommendations.append({
                    "priority": "high",
                    "category": "cost_trend",
                    "recommendation": "Investigate drivers of increasing delivery costs",
                    "expected_impact": "Identify cost drivers and implement targeted cost reduction measures"
                })
        
        # Generate recommendations based on cost breakdown
        if "cost_breakdown" in cost_metrics:
            breakdown = cost_metrics["cost_breakdown"]
            
            # Find highest cost category
            highest_category = max(breakdown.items(), key=lambda x: x[1]["value"])
            category_name, category_data = highest_category
            percentage = category_data.get("percentage", 0)
            
            if percentage > 40:
                recommendations.append({
                    "priority": "high",
                    "category": "cost_category",
                    "recommendation": f"Develop cost reduction strategy for '{category_name}' costs",
                    "expected_impact": f"Target 10-15% reduction in '{category_name}' costs"
                })
        
        # Generate recommendations based on grouped costs
        high_cost_groups = []
        if grouped_costs and any("cost_per_delivery" in group_metrics for group_metrics in grouped_costs.values()):
            # Calculate average cost per delivery
            avg_cost = 0
            if "cost_per_delivery" in cost_metrics:
                avg_cost = cost_metrics["cost_per_delivery"]["value"]
            else:
                group_costs = [
                    group_metrics["cost_per_delivery"]["value"]
                    for group_metrics in grouped_costs.values()
                    if "cost_per_delivery" in group_metrics
                ]
                if group_costs:
                    avg_cost = sum(group_costs) / len(group_costs)
            
            # Find groups with significantly higher cost than average
            for group_name, group_metrics in grouped_costs.items():
                if "cost_per_delivery" in group_metrics:
                    group_cost = group_metrics["cost_per_delivery"]["value"]
                    
                    # If cost is 20% or more above average, add to high cost groups
                    if avg_cost > 0 and group_cost > avg_cost * 1.2:
                        high_cost_groups.append((group_name, group_cost, (group_cost - avg_cost) / avg_cost * 100))
            
            # Sort high cost groups by percentage above average
            high_cost_groups.sort(key=lambda x: x[2], reverse=True)
            
            # Generate recommendations for high cost groups
            for group_name, group_cost, percent_above in high_cost_groups[:3]:
                recommendations.append({
                    "priority": "medium",
                    "category": "high_cost_group",
                    "recommendation": f"Analyze and optimize delivery operations for '{group_name}'",
                    "expected_impact": f"Potential to reduce '{group_name}' delivery costs by 15-20%"
                })
        
        # Add general cost-saving recommendations
        recommendations.append({
            "priority": "medium",
            "category": "cost_saving",
            "recommendation": "Implement route optimization to reduce total miles traveled",
            "expected_impact": "Potential 10-15% reduction in transportation costs"
        })
        
        recommendations.append({
            "priority": "medium",
            "category": "cost_saving",
            "recommendation": "Review carrier contracts and negotiate volume-based discounts",
            "expected_impact": "Potential 5-10% reduction in carrier fees"
        })
        
        recommendations.append({
            "priority": "low",
            "category": "cost_saving",
            "recommendation": "Implement regular cost variance analysis",
            "expected_impact": "Better visibility into cost drivers and opportunities for savings"
        })
        
        return recommendations
    
    def _calculate_last_mile_metrics(
        self,
        last_mile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate last mile delivery metrics.
        
        Args:
            last_mile_data: Last mile delivery data
            
        Returns:
            Dictionary with last mile metrics
        """
        last_mile_metrics = {}
        
        # Extract key metrics
        if "summary" in last_mile_data:
            summary = last_mile_data["summary"]
            
            # On-time delivery rate
            if "total_deliveries" in summary and "on_time_deliveries" in summary:
                total_deliveries = summary["total_deliveries"]
                on_time_deliveries = summary["on_time_deliveries"]
                
                on_time_rate = (on_time_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
                
                last_mile_metrics["on_time_delivery"] = {
                    "value": round(on_time_rate, 1),
                    "total_deliveries": total_deliveries,
                    "on_time_deliveries": on_time_deliveries,
                    "unit": "%",
                    "performance_level": self._get_performance_level(
                        "on_time_delivery", on_time_rate
                    )
                }
            
            # First attempt delivery rate
            if "total_deliveries" in summary and "first_attempt_success" in summary:
                total_deliveries = summary["total_deliveries"]
                first_attempt_success = summary["first_attempt_success"]
                
                first_attempt_rate = (first_attempt_success / total_deliveries * 100) if total_deliveries > 0 else 0
                
                last_mile_metrics["first_attempt_rate"] = {
                    "value": round(first_attempt_rate, 1),
                    "total_deliveries": total_deliveries,
                    "first_attempt_success": first_attempt_success,
                    "unit": "%"
                }
            
            # Average time per delivery
            if "avg_delivery_time" in summary:
                avg_delivery_time = summary["avg_delivery_time"]
                
                last_mile_metrics["avg_delivery_time"] = {
                    "value": round(avg_delivery_time, 1),
                    "unit": "minutes"
                }
            
            # Average stops per route
            if "avg_stops_per_route" in summary:
                avg_stops_per_route = summary["avg_stops_per_route"]
                
                last_mile_metrics["avg_stops_per_route"] = {
                    "value": round(avg_stops_per_route, 1),
                    "unit": "stops"
                }
            
            # Customer satisfaction
            if "customer_satisfaction" in summary:
                customer_satisfaction = summary["customer_satisfaction"]
                
                last_mile_metrics["customer_satisfaction"] = {
                    "value": round(customer_satisfaction, 1),
                    "unit": "score (0-10)"
                }
        
        return last_mile_metrics
    
    def _group_last_mile_by_region(
        self,
        last_mile_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group last mile performance by region.
        
        Args:
            last_mile_data: Last mile delivery data
            
        Returns:
            Dictionary with performance by region
        """
        performance_by_region = {}
        
        # Extract region data
        if "by_region" in last_mile_data:
            region_data = last_mile_data["by_region"]
            
            for region_name, region_values in region_data.items():
                # Calculate metrics for this region
                region_metrics = {}
                
                # On-time delivery rate
                if "total_deliveries" in region_values and "on_time_deliveries" in region_values:
                    total_deliveries = region_values["total_deliveries"]
                    on_time_deliveries = region_values["on_time_deliveries"]
                    
                    on_time_rate = (on_time_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
                    
                    region_metrics["on_time_delivery"] = {
                        "value": round(on_time_rate, 1),
                        "total_deliveries": total_deliveries,
                        "on_time_deliveries": on_time_deliveries,
                        "unit": "%",
                        "performance_level": self._get_performance_level(
                            "on_time_delivery", on_time_rate
                        )
                    }
                
                # First attempt delivery rate
                if "total_deliveries" in region_values and "first_attempt_success" in region_values:
                    total_deliveries = region_values["total_deliveries"]
                    first_attempt_success = region_values["first_attempt_success"]
                    
                    first_attempt_rate = (first_attempt_success / total_deliveries * 100) if total_deliveries > 0 else 0
                    
                    region_metrics["first_attempt_rate"] = {
                        "value": round(first_attempt_rate, 1),
                        "total_deliveries": total_deliveries,
                        "first_attempt_success": first_attempt_success,
                        "unit": "%"
                    }
                
                # Average delivery time
                if "avg_delivery_time" in region_values:
                    region_metrics["avg_delivery_time"] = {
                        "value": round(region_values["avg_delivery_time"], 1),
                        "unit": "minutes"
                    }
                
                # Add metrics for this region
                performance_by_region[region_name] = region_metrics
        
        return performance_by_region
    
    def _calculate_last_mile_trends(
        self,
        last_mile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate last mile delivery trends.
        
        Args:
            last_mile_data: Last mile delivery data
            
        Returns:
            Dictionary with last mile trends
        """
        last_mile_trends = {}
        
        # Extract trend data
        if "trends" in last_mile_data:
            trend_data = last_mile_data["trends"]
            
            # On-time delivery trend
            if "on_time_delivery" in trend_data:
                on_time_trend = trend_data["on_time_delivery"]
                
                if "values" in on_time_trend and "periods" in on_time_trend:
                    values = on_time_trend["values"]
                    periods = on_time_trend["periods"]
                    
                    # Calculate trend direction and magnitude
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values)
                    
                    last_mile_trends["on_time_delivery"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # First attempt delivery trend
            if "first_attempt_rate" in trend_data:
                first_attempt_trend = trend_data["first_attempt_rate"]
                
                if "values" in first_attempt_trend and "periods" in first_attempt_trend:
                    values = first_attempt_trend["values"]
                    periods = first_attempt_trend["periods"]
                    
                    # Calculate trend direction and magnitude
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values)
                    
                    last_mile_trends["first_attempt_rate"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # Average delivery time trend
            if "avg_delivery_time" in trend_data:
                time_trend = trend_data["avg_delivery_time"]
                
                if "values" in time_trend and "periods" in time_trend:
                    values = time_trend["values"]
                    periods = time_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for time, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    last_mile_trends["avg_delivery_time"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
        
        return last_mile_trends
    
    def _generate_last_mile_insights(
        self,
        last_mile_metrics: Dict[str, Any],
        performance_by_region: Dict[str, Dict[str, Any]],
        last_mile_trends: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from last mile delivery data.
        
        Args:
            last_mile_metrics: Last mile metrics
            performance_by_region: Performance by region
            last_mile_trends: Last mile trends
            
        Returns:
            List of insights
        """
        insights = []
        
        # Generate insights based on overall metrics
        if "on_time_delivery" in last_mile_metrics:
            on_time_value = last_mile_metrics["on_time_delivery"]["value"]
            performance_level = last_mile_metrics["on_time_delivery"]["performance_level"]
            
            if performance_level in ("excellent", "good"):
                insights.append({
                    "type": "positive",
                    "category": "on_time_delivery",
                    "insight": f"Strong last mile on-time delivery at {on_time_value}%"
                })
            elif performance_level in ("below_average", "poor"):
                insights.append({
                    "type": "negative",
                    "category": "on_time_delivery",
                    "insight": f"Poor last mile on-time delivery at {on_time_value}%"
                })
        
        if "first_attempt_rate" in last_mile_metrics:
            first_attempt_rate = last_mile_metrics["first_attempt_rate"]["value"]
            
            if first_attempt_rate >= 90:
                insights.append({
                    "type": "positive",
                    "category": "first_attempt_rate",
                    "insight": f"High first attempt delivery success rate at {first_attempt_rate}%"
                })
            elif first_attempt_rate < 80:
                insights.append({
                    "type": "negative",
                    "category": "first_attempt_rate",
                    "insight": f"Low first attempt delivery success rate at {first_attempt_rate}%"
                })
        
        if "customer_satisfaction" in last_mile_metrics:
            satisfaction = last_mile_metrics["customer_satisfaction"]["value"]
            
            if satisfaction >= 8.5:
                insights.append({
                    "type": "positive",
                    "category": "customer_satisfaction",
                    "insight": f"High customer satisfaction with last mile delivery at {satisfaction}/10"
                })
            elif satisfaction < 7:
                insights.append({
                    "type": "negative",
                    "category": "customer_satisfaction",
                    "insight": f"Low customer satisfaction with last mile delivery at {satisfaction}/10"
                })
        
        # Generate insights based on trends
        if "on_time_delivery" in last_mile_trends:
            direction = last_mile_trends["on_time_delivery"]["direction"]
            magnitude = last_mile_trends["on_time_delivery"]["magnitude"]
            
            if direction == "improving" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "on_time_delivery",
                    "insight": f"Last mile on-time delivery improving by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "on_time_delivery",
                    "insight": f"Last mile on-time delivery declining by {magnitude:.1f}%"
                })
        
        # Generate insights by region
        if performance_by_region:
            # Find best and worst performing regions
            on_time_by_region = {
                region: metrics["on_time_delivery"]["value"]
                for region, metrics in performance_by_region.items()
                if "on_time_delivery" in metrics
            }
            
            if on_time_by_region:
                best_region = max(on_time_by_region.items(), key=lambda x: x[1])
                worst_region = min(on_time_by_region.items(), key=lambda x: x[1])
                
                region_gap = best_region[1] - worst_region[1]
                
                if region_gap >= 10:
                    insights.append({
                        "type": "region",
                        "category": "regional_variation",
                        "insight": f"Large on-time delivery gap between regions: {best_region[0]} ({best_region[1]}%) vs {worst_region[0]} ({worst_region[1]}%)"
                    })
            
            # Find regions with particularly low first attempt rates
            first_attempt_by_region = {
                region: metrics["first_attempt_rate"]["value"]
                for region, metrics in performance_by_region.items()
                if "first_attempt_rate" in metrics
            }
            
            if first_attempt_by_region:
                low_regions = [
                    (region, rate) for region, rate in first_attempt_by_region.items()
                    if rate < 75
                ]
                
                for region, rate in low_regions[:2]:  # Limit to top 2
                    insights.append({
                        "type": "region",
                        "category": "first_attempt_rate",
                        "insight": f"Low first attempt delivery rate in {region} at {rate}%"
                    })
        
        return insights
    
    def _generate_last_mile_recommendations(
        self,
        last_mile_metrics: Dict[str, Any],
        performance_by_region: Dict[str, Dict[str, Any]],
        last_mile_trends: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on last mile delivery data.
        
        Args:
            last_mile_metrics: Last mile metrics
            performance_by_region: Performance by region
            last_mile_trends: Last mile trends
            insights: Generated insights
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate recommendations based on overall metrics
        if "on_time_delivery" in last_mile_metrics:
            on_time_value = last_mile_metrics["on_time_delivery"]["value"]
            performance_level = last_mile_metrics["on_time_delivery"]["performance_level"]
            
            if performance_level in ("below_average", "poor"):
                recommendations.append({
                    "priority": "high",
                    "category": "on_time_delivery",
                    "recommendation": "Implement real-time tracking and dynamic routing for last mile delivery",
                    "expected_impact": "Potential improvement of 10-15% in on-time delivery rate"
                })
        
        if "first_attempt_rate" in last_mile_metrics:
            first_attempt_rate = last_mile_metrics["first_attempt_rate"]["value"]
            
            if first_attempt_rate < 85:
                recommendations.append({
                    "priority": "high",
                    "category": "first_attempt_rate",
                    "recommendation": "Implement delivery scheduling and notification system for customers",
                    "expected_impact": "Potential increase of 10-20% in first attempt delivery success"
                })
        
        if "avg_delivery_time" in last_mile_metrics:
            avg_delivery_time = last_mile_metrics["avg_delivery_time"]["value"]
            
            if avg_delivery_time > 10:
                recommendations.append({
                    "priority": "medium",
                    "category": "delivery_time",
                    "recommendation": "Optimize package sorting and loading processes to reduce delivery time",
                    "expected_impact": "Potential reduction of 15-25% in average delivery time"
                })
        
        # Generate recommendations based on trends
        for metric, trend_data in last_mile_trends.items():
            direction = trend_data["direction"]
            magnitude = trend_data["magnitude"]
            
            if direction == "declining" and magnitude >= 5:
                if metric == "on_time_delivery":
                    recommendations.append({
                        "priority": "high",
                        "category": "on_time_delivery",
                        "recommendation": "Investigate causes of declining on-time delivery performance",
                        "expected_impact": "Identify and address factors causing the negative trend"
                    })
                elif metric == "first_attempt_rate":
                    recommendations.append({
                        "priority": "high",
                        "category": "first_attempt_rate",
                        "recommendation": "Analyze patterns in failed delivery attempts",
                        "expected_impact": "Identify key factors reducing first attempt success rate"
                    })
        
        # Generate region-specific recommendations
        if performance_by_region:
            # Identify poorly performing regions
            poor_regions = []
            for region, metrics in performance_by_region.items():
                if "on_time_delivery" in metrics:
                    on_time = metrics["on_time_delivery"]["value"]
                    performance_level = metrics["on_time_delivery"]["performance_level"]
                    
                    if performance_level in ("below_average", "poor"):
                        poor_regions.append((region, on_time))
            
            # Sort by on-time delivery rate (worst first)
            poor_regions.sort(key=lambda x: x[1])
            
            # Make recommendations for worst performing regions
            for region, on_time in poor_regions[:2]:  # Limit to top 2
                recommendations.append({
                    "priority": "medium",
                    "category": "regional",
                    "recommendation": f"Develop region-specific improvement plan for {region}",
                    "expected_impact": f"Address unique challenges in {region} to improve performance"
                })
        
        # Add general recommendations
        recommendations.append({
            "priority": "medium",
            "category": "technology",
            "recommendation": "Implement mobile proof of delivery app with geolocation verification",
            "expected_impact": "Improve delivery accuracy and provide better visibility"
        })
        
        recommendations.append({
            "priority": "low",
            "category": "training",
            "recommendation": "Enhance training program for last mile delivery drivers",
            "expected_impact": "Improve delivery quality and customer interaction"
        })
        
        return recommendations
    
    def _get_performance_level(
        self,
        metric_name: str,
        value: float
    ) -> str:
        """
        Get performance level based on metric value.
        
        Args:
            metric_name: Metric name
            value: Metric value
            
        Returns:
            Performance level (excellent, good, average, below_average, poor)
        """
        if metric_name not in self.PERFORMANCE_THRESHOLDS:
            return "unknown"
        
        thresholds = self.PERFORMANCE_THRESHOLDS[metric_name]
        
        # Check each threshold
        for level, threshold in thresholds.items():
            # For metrics where higher is better (default)
            if metric_name not in ["order_cycle_time"]:
                if value >= threshold:
                    return level
            # For metrics where lower is better
            else:
                if value <= threshold:
                    return level
        
        return "unknown"
    
    def _generate_mock_delivery_data(
        self,
        time_period: str = "last_30_days",
        filters: Optional[Dict[str, Any]] = None,
        error: bool = False
    ) -> Dict[str, Any]:
        """
        Generate mock delivery data for testing.
        
        Args:
            time_period: Time period
            filters: Optional filters
            error: Whether to simulate an error
            
        Returns:
            Dictionary with mock delivery data
        """
        if error:
            return {
                "error": "Failed to retrieve delivery data",
                "time_period": time_period,
                "is_mock_data": True
            }
        
        # Generate a deterministic random seed
        seed = hash(time_period) % 10000
        np.random.seed(seed)
        
        # Generate mock deliveries data
        total_deliveries = np.random.randint(5000, 20000)
        on_time_rate = np.random.uniform(0.82, 0.95)
        on_time_deliveries = int(total_deliveries * on_time_rate)
        
        accurate_rate = np.random.uniform(0.90, 0.98)
        accurate_deliveries = int(total_deliveries * accurate_rate)
        
        # Generate mock cycle time data
        avg_target_cycle_time = np.random.uniform(2.5, 4.0)
        actual_ratio = np.random.uniform(0.9, 1.3)
        avg_actual_cycle_time = avg_target_cycle_time * actual_ratio
        
        # Generate mock cost data
        total_cost = total_deliveries * np.random.uniform(8, 15)
        total_miles = total_deliveries * np.random.uniform(5, 12)
        
        # Generate mock exception data
        exception_rate = np.random.uniform(0.03, 0.08)
        total_exceptions = int(total_deliveries * exception_rate)
        
        # Generate mock trend data
        periods = 12
        trend_periods = []
        
        # Create period labels
        if time_period == "last_30_days":
            # Daily periods
            for i in range(periods):
                days_ago = periods - i - 1
                trend_periods.append(f"Day {days_ago + 1}")
        elif time_period == "last_12_months":
            # Monthly periods
            current_month = datetime.now().month
            current_year = datetime.now().year
            
            for i in range(periods):
                month = current_month - i
                year = current_year
                if month <= 0:
                    month += 12
                    year -= 1
                trend_periods.append(f"{year}-{month:02d}")
        else:
            # Generic periods
            for i in range(periods):
                trend_periods.append(f"Period {i+1}")
        
        # Generate trend values with some randomness and trend
        def generate_trend_values(base: float, trend_factor: float, noise: float) -> List[float]:
            return [
                max(0, base * (1 + trend_factor * (i / periods)) + np.random.normal(0, noise * base))
                for i in range(periods)
            ]
        
        on_time_trend = generate_trend_values(on_time_rate * 100, np.random.uniform(-0.1, 0.1), 0.03)
        accuracy_trend = generate_trend_values(accurate_rate * 100, np.random.uniform(-0.05, 0.05), 0.02)
        cycle_time_trend = generate_trend_values(avg_actual_cycle_time, np.random.uniform(-0.15, 0.15), 0.05)
        cost_trend = generate_trend_values(total_cost / total_deliveries, np.random.uniform(-0.05, 0.15), 0.04)
        
        # Generate mock grouped data
        group_dimensions = ["carrier", "region", "customer", "product"]
        grouped_data = {}
        
        for dimension in group_dimensions:
            group_count = np.random.randint(3, 8)
            dimension_data = {}
            
            # Generate data for each group
            for i in range(group_count):
                group_name = f"{dimension.title()} {i+1}"
                group_deliveries = np.random.randint(total_deliveries // (group_count * 2), total_deliveries // group_count)
                
                group_on_time_rate = on_time_rate * np.random.uniform(0.85, 1.15)
                group_on_time_deliveries = int(group_deliveries * group_on_time_rate)
                
                group_accuracy_rate = accurate_rate * np.random.uniform(0.9, 1.1)
                group_accurate_deliveries = int(group_deliveries * group_accuracy_rate)
                
                group_cost = group_deliveries * np.random.uniform(7, 18)
                
                dimension_data[group_name] = {
                    "total_deliveries": group_deliveries,
                    "on_time_deliveries": group_on_time_deliveries,
                    "accurate_deliveries": group_accurate_deliveries,
                    "total_cost": group_cost
                }
            
            grouped_data[dimension] = dimension_data
        
        # Compile delivery data
        delivery_data = {
            "time_period": time_period,
            "deliveries": {
                "total": total_deliveries,
                "on_time": on_time_deliveries,
                "accurate": accurate_deliveries
            },
            "cycle_times": {
                "avg_target": avg_target_cycle_time,
                "avg_actual": avg_actual_cycle_time
            },
            "costs": {
                "total_cost": total_cost,
                "total_deliveries": total_deliveries,
                "total_miles": total_miles
            },
            "exceptions": {
                "total_deliveries": total_deliveries,
                "total_exceptions": total_exceptions
            },
            "trends": {
                "on_time_delivery": {
                    "values": on_time_trend,
                    "periods": trend_periods
                },
                "delivery_accuracy": {
                    "values": accuracy_trend,
                    "periods": trend_periods
                },
                "order_cycle_time": {
                    "values": cycle_time_trend,
                    "periods": trend_periods
                },
                "cost_per_delivery": {
                    "values": cost_trend,
                    "periods": trend_periods
                }
            },
            "by_group": grouped_data,
            "is_mock_data": True
        }
        
        # Apply filters if provided
        if filters:
            # This is a placeholder for filter logic
            # In a real implementation, you would filter the data based on criteria
            delivery_data["applied_filters"] = filters
        
        return delivery_data
    
    def _generate_mock_exception_data(
        self,
        time_period: str = "last_30_days",
        filters: Optional[Dict[str, Any]] = None,
        error: bool = False
    ) -> Dict[str, Any]:
        """
        Generate mock exception data for testing.
        
        Args:
            time_period: Time period
            filters: Optional filters
            error: Whether to simulate an error
            
        Returns:
            Dictionary with mock exception data
        """
        if error:
            return {
                "error": "Failed to retrieve exception data",
                "time_period": time_period,
                "is_mock_data": True
            }
        
        # Generate a deterministic random seed
        seed = hash(time_period) % 10000
        np.random.seed(seed)
        
        # Generate mock summary data
        total_deliveries = np.random.randint(5000, 20000)
        exception_rate = np.random.uniform(0.03, 0.08)
        total_exceptions = int(total_deliveries * exception_rate)
        
        avg_resolution_time = np.random.uniform(4, 24)
        
        unresolved_rate = np.random.uniform(0.05, 0.15)
        unresolved_exceptions = int(total_exceptions * unresolved_rate)
        
        # Define exception types
        exception_types = [
            "Address Not Found",
            "Customer Not Available",
            "Damaged in Transit",
            "Weather Delay",
            "Vehicle Breakdown",
            "Package Lost",
            "Delivery Refused",
            "Access Issues"
        ]
        
        # Generate exception counts by type
        exception_counts = {}
        remaining_exceptions = total_exceptions
        
        for i, exception_type in enumerate(exception_types):
            # Last type gets all remaining exceptions
            if i == len(exception_types) - 1:
                count = remaining_exceptions
            else:
                # Generate random count for this type
                max_count = min(remaining_exceptions, total_exceptions // 2)
                count = np.random.randint(max_count // 5, max_count)
                remaining_exceptions -= count
            
            # Skip if count is 0
            if count == 0:
                continue
            
            # Generate resolution time for this type
            type_resolution_time = avg_resolution_time * np.random.uniform(0.7, 1.3)
            
            exception_counts[exception_type] = {
                "count": count,
                "total_exceptions": total_exceptions,
                "avg_resolution_time": type_resolution_time,
                "description": f"Exceptions where {exception_type.lower()}"
            }
        
        # Define exception causes
        exception_causes = [
            "Incorrect Address",
            "Customer Unavailable",
            "Weather Conditions",
            "Carrier Error",
            "Warehouse Error",
            "Technical Issues",
            "Traffic Conditions"
        ]
        
        # Generate exception counts by cause
        cause_counts = {}
        remaining_exceptions = total_exceptions
        
        for i, cause in enumerate(exception_causes):
            # Last cause gets all remaining exceptions
            if i == len(exception_causes) - 1:
                count = remaining_exceptions
            else:
                # Generate random count for this cause
                max_count = min(remaining_exceptions, total_exceptions // 2)
                count = np.random.randint(max_count // 5, max_count)
                remaining_exceptions -= count
            
            # Skip if count is 0
            if count == 0:
                continue
            
            # Determine if cause is preventable
            preventable = cause in ["Incorrect Address", "Carrier Error", "Warehouse Error", "Technical Issues"]
            
            cause_counts[cause] = {
                "count": count,
                "total_exceptions": total_exceptions,
                "description": f"Exceptions caused by {cause.lower()}",
                "preventable": preventable
            }
        
        # Generate trend data
        periods = 12
        trend_periods = []
        
        # Create period labels
        if time_period == "last_30_days":
            # Daily periods
            for i in range(periods):
                days_ago = periods - i - 1
                trend_periods.append(f"Day {days_ago + 1}")
        elif time_period == "last_12_months":
            # Monthly periods
            current_month = datetime.now().month
            current_year = datetime.now().year
            
            for i in range(periods):
                month = current_month - i
                year = current_year
                if month <= 0:
                    month += 12
                    year -= 1
                trend_periods.append(f"{year}-{month:02d}")
        else:
            # Generic periods
            for i in range(periods):
                trend_periods.append(f"Period {i+1}")
        
        # Generate trend values with some randomness and trend
        def generate_trend_values(base: float, trend_factor: float, noise: float) -> List[float]:
            return [
                max(0, base * (1 + trend_factor * (i / periods)) + np.random.normal(0, noise * base))
                for i in range(periods)
            ]
        
        exception_rate_trend = generate_trend_values(exception_rate * 100, np.random.uniform(-0.15, 0.15), 0.05)
        resolution_time_trend = generate_trend_values(avg_resolution_time, np.random.uniform(-0.1, 0.1), 0.07)
        
        # Compile exception data
        exception_data = {
            "time_period": time_period,
            "summary": {
                "total_deliveries": total_deliveries,
                "total_exceptions": total_exceptions,
                "exception_rate": exception_rate * 100,
                "avg_resolution_time": avg_resolution_time,
                "unresolved_exceptions": unresolved_exceptions,
                "customer_impact": {
                    "satisfaction_impact": np.random.uniform(-1.5, -0.5),
                    "repeat_order_impact": np.random.uniform(-0.15, -0.05)
                }
            },
            "by_type": exception_counts,
            "by_cause": cause_counts,
            "trends": {
                "exception_rate": {
                    "values": exception_rate_trend,
                    "periods": trend_periods
                },
                "resolution_time": {
                    "values": resolution_time_trend,
                    "periods": trend_periods
                }
            },
            "is_mock_data": True
        }
        
        # Apply filters if provided
        if filters:
            # This is a placeholder for filter logic
            exception_data["applied_filters"] = filters
        
        return exception_data
    
    def _generate_mock_cost_data(
        self,
        time_period: str = "last_30_days",
        filters: Optional[Dict[str, Any]] = None,
        error: bool = False
    ) -> Dict[str, Any]:
        """
        Generate mock cost data for testing.
        
        Args:
            time_period: Time period
            filters: Optional filters
            error: Whether to simulate an error
            
        Returns:
            Dictionary with mock cost data
        """
        if error:
            return {
                "error": "Failed to retrieve cost data",
                "time_period": time_period,
                "is_mock_data": True
            }
        
        # Generate a deterministic random seed
        seed = hash(time_period) % 10000
        np.random.seed(seed)
        
        # Generate mock summary data
        total_deliveries = np.random.randint(5000, 20000)
        cost_per_delivery = np.random.uniform(8, 15)
        total_cost = total_deliveries * cost_per_delivery
        
        total_miles = total_deliveries * np.random.uniform(5, 12)
        cost_per_mile = total_cost / total_miles
        
        # Generate cost breakdown
        cost_categories = [
            "Fuel",
            "Labor",
            "Vehicle Maintenance",
            "Packaging",
            "Insurance",
            "Miscellaneous"
        ]
        
        cost_breakdown = {}
        remaining_cost = total_cost
        
        for i, category in enumerate(cost_categories):
            # Last category gets all remaining cost
            if i == len(cost_categories) - 1:
                category_cost = remaining_cost
            else:
                # Generate random cost for this category
                if category == "Labor":
                    # Labor typically high percentage
                    category_cost = total_cost * np.random.uniform(0.35, 0.45)
                elif category == "Fuel":
                    # Fuel another significant cost
                    category_cost = total_cost * np.random.uniform(0.15, 0.25)
                else:
                    category_cost = total_cost * np.random.uniform(0.05, 0.15)
                
                remaining_cost -= category_cost
            
            percentage = (category_cost / total_cost * 100)
            
            cost_breakdown[category] = {
                "value": round(category_cost, 2),
                "percentage": round(percentage, 1),
                "unit": "$"
            }
        
        # Generate trend data
        periods = 12
        trend_periods = []
        
        # Create period labels
        if time_period == "last_30_days":
            # Daily periods
            for i in range(periods):
                days_ago = periods - i - 1
                trend_periods.append(f"Day {days_ago + 1}")
        elif time_period == "last_12_months":
            # Monthly periods
            current_month = datetime.now().month
            current_year = datetime.now().year
            
            for i in range(periods):
                month = current_month - i
                year = current_year
                if month <= 0:
                    month += 12
                    year -= 1
                trend_periods.append(f"{year}-{month:02d}")
        else:
            # Generic periods
            for i in range(periods):
                trend_periods.append(f"Period {i+1}")
        
        # Generate trend values with some randomness and trend
        def generate_trend_values(base: float, trend_factor: float, noise: float) -> List[float]:
            return [
                max(0, base * (1 + trend_factor * (i / periods)) + np.random.normal(0, noise * base))
                for i in range(periods)
            ]
        
        # Generate cost trends (slightly increasing)
        cost_per_delivery_trend = generate_trend_values(cost_per_delivery, np.random.uniform(0.0, 0.1), 0.03)
        cost_per_mile_trend = generate_trend_values(cost_per_mile, np.random.uniform(0.0, 0.1), 0.04)
        
        # Generate mock grouped data
        group_dimensions = ["carrier", "region", "customer", "product"]
        grouped_data = {}
        
        for dimension in group_dimensions:
            group_count = np.random.randint(3, 8)
            dimension_data = {}
            
            # Generate data for each group
            for i in range(group_count):
                group_name = f"{dimension.title()} {i+1}"
                group_deliveries = np.random.randint(total_deliveries // (group_count * 2), total_deliveries // group_count)
                
                group_cost_per_delivery = cost_per_delivery * np.random.uniform(0.85, 1.2)
                group_total_cost = group_deliveries * group_cost_per_delivery
                
                group_miles = group_deliveries * np.random.uniform(4, 13)
                
                dimension_data[group_name] = {
                    "total_deliveries": group_deliveries,
                    "total_cost": group_total_cost,
                    "total_miles": group_miles
                }
            
            grouped_data[dimension] = dimension_data
        
        # Compile cost data
        cost_data = {
            "time_period": time_period,
            "summary": {
                "total_deliveries": total_deliveries,
                "total_cost": total_cost,
                "cost_per_delivery": cost_per_delivery,
                "total_miles": total_miles,
                "cost_per_mile": cost_per_mile,
                "cost_breakdown": cost_breakdown
            },
            "trends": {
                "cost_per_delivery": {
                    "values": cost_per_delivery_trend,
                    "periods": trend_periods
                },
                "cost_per_mile": {
                    "values": cost_per_mile_trend,
                    "periods": trend_periods
                }
            },
            "by_group": grouped_data,
            "is_mock_data": True
        }
        
        # Apply filters if provided
        if filters:
            # This is a placeholder for filter logic
            cost_data["applied_filters"] = filters
        
        return cost_data
    
    def _generate_mock_last_mile_data(
        self,
        time_period: str = "last_30_days",
        filters: Optional[Dict[str, Any]] = None,
        error: bool = False
    ) -> Dict[str, Any]:
        """
        Generate mock last mile delivery data for testing.
        
        Args:
            time_period: Time period
            filters: Optional filters
            error: Whether to simulate an error
            
        Returns:
            Dictionary with mock last mile data
        """
        if error:
            return {
                "error": "Failed to retrieve last mile data",
                "time_period": time_period,
                "is_mock_data": True
            }
        
        # Generate a deterministic random seed
        seed = hash(time_period) % 10000
        np.random.seed(seed)
        
        # Generate mock summary data
        total_deliveries = np.random.randint(5000, 20000)
        on_time_rate = np.random.uniform(0.8, 0.93)
        on_time_deliveries = int(total_deliveries * on_time_rate)
        
        first_attempt_rate = np.random.uniform(0.82, 0.95)
        first_attempt_success = int(total_deliveries * first_attempt_rate)
        
        avg_delivery_time = np.random.uniform(5, 12)  # minutes
        avg_stops_per_route = np.random.uniform(25, 40)
        customer_satisfaction = np.random.uniform(7.5, 9.0)  # 0-10 scale
        
        # Generate data by region
        regions = [
            "Northeast",
            "Southeast",
            "Midwest",
            "Southwest",
            "West",
            "Northwest"
        ]
        
        region_data = {}
        for region in regions:
            region_deliveries = np.random.randint(total_deliveries // (len(regions) * 2), total_deliveries // len(regions))
            
            region_on_time_rate = on_time_rate * np.random.uniform(0.9, 1.1)
            region_on_time_deliveries = int(region_deliveries * region_on_time_rate)
            
            region_first_attempt_rate = first_attempt_rate * np.random.uniform(0.9, 1.1)
            region_first_attempt_success = int(region_deliveries * region_first_attempt_rate)
            
            region_delivery_time = avg_delivery_time * np.random.uniform(0.85, 1.15)
            
            region_data[region] = {
                "total_deliveries": region_deliveries,
                "on_time_deliveries": region_on_time_deliveries,
                "first_attempt_success": region_first_attempt_success,
                "avg_delivery_time": region_delivery_time
            }
        
        # Generate trend data
        periods = 12
        trend_periods = []
        
        # Create period labels
        if time_period == "last_30_days":
            # Daily periods
            for i in range(periods):
                days_ago = periods - i - 1
                trend_periods.append(f"Day {days_ago + 1}")
        elif time_period == "last_12_months":
            # Monthly periods
            current_month = datetime.now().month
            current_year = datetime.now().year
            
            for i in range(periods):
                month = current_month - i
                year = current_year
                if month <= 0:
                    month += 12
                    year -= 1
                trend_periods.append(f"{year}-{month:02d}")
        else:
            # Generic periods
            for i in range(periods):
                trend_periods.append(f"Period {i+1}")
        
        # Generate trend values with some randomness and trend
        def generate_trend_values(base: float, trend_factor: float, noise: float) -> List[float]:
            return [
                max(0, base * (1 + trend_factor * (i / periods)) + np.random.normal(0, noise * base))
                for i in range(periods)
            ]
        
        on_time_trend = generate_trend_values(on_time_rate * 100, np.random.uniform(-0.05, 0.05), 0.03)
        first_attempt_trend = generate_trend_values(first_attempt_rate * 100, np.random.uniform(-0.05, 0.05), 0.03)
        delivery_time_trend = generate_trend_values(avg_delivery_time, np.random.uniform(-0.1, 0.1), 0.05)
        
        # Compile last mile data
        last_mile_data = {
            "time_period": time_period,
            "summary": {
                "total_deliveries": total_deliveries,
                "on_time_deliveries": on_time_deliveries,
                "first_attempt_success": first_attempt_success,
                "avg_delivery_time": avg_delivery_time,
                "avg_stops_per_route": avg_stops_per_route,
                "customer_satisfaction": customer_satisfaction
            },
            "by_region": region_data,
            "trends": {
                "on_time_delivery": {
                    "values": on_time_trend,
                    "periods": trend_periods
                },
                "first_attempt_rate": {
                    "values": first_attempt_trend,
                    "periods": trend_periods
                },
                "avg_delivery_time": {
                    "values": delivery_time_trend,
                    "periods": trend_periods
                }
            },
            "is_mock_data": True
        }
        
        # Apply filters if provided
        if filters:
            # This is a placeholder for filter logic
            last_mile_data["applied_filters"] = filters
        
        return last_mile_data"""
Delivery Analytics Module

This module provides functionality for analyzing delivery performance,
tracking metrics, and identifying improvement opportunities.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import math

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class DeliveryAnalytics:
    """
    Analyzes delivery performance and identifies improvement opportunities.
    """
    
    # Performance threshold levels
    PERFORMANCE_THRESHOLDS = {
        "on_time_delivery": {
            "excellent": 95,
            "good": 90,
            "average": 85,
            "below_average": 80,
            "poor": 0
        },
        "delivery_accuracy": {
            "excellent": 98,
            "good": 95,
            "average": 92,
            "below_average": 90,
            "poor": 0
        },
        "order_cycle_time": {
            "excellent": 0,  # Lower is better, thresholds are reversed
            "good": 1.1,
            "average": 1.2,
            "below_average": 1.4,
            "poor": float('inf')
        }
    }
    
    def __init__(self):
        """Initialize the delivery analytics module."""
        pass
    
    async def analyze_delivery_performance(
        self,
        time_period: str = "last_30_days",
        group_by: str = "none",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze delivery performance metrics.
        
        Args:
            time_period: Time period for analysis
            group_by: Group by dimension (none, carrier, region, customer, product)
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with delivery performance analysis
        """
        try:
            # Get delivery data
            delivery_data = await self._get_delivery_data(
                time_period=time_period,
                client_id=client_id,
                connection_id=connection_id,
                filters=filters
            )
            
            # Calculate summary metrics
            summary_metrics = self._calculate_summary_metrics(delivery_data)
            
            # Group data if requested
            grouped_metrics = {}
            if group_by != "none" and group_by in delivery_data:
                grouped_metrics = self._group_delivery_data(
                    delivery_data=delivery_data,
                    group_by=group_by
                )
            
            # Calculate trends
            trends = self._calculate_trends(delivery_data)
            
            # Generate insights
            insights = self._generate_insights(
                summary_metrics=summary_metrics,
                grouped_metrics=grouped_metrics,
                trends=trends,
                group_by=group_by
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                summary_metrics=summary_metrics,
                grouped_metrics=grouped_metrics,
                trends=trends,
                insights=insights
            )
            
            # Prepare result
            result = {
                "time_period": time_period,
                "summary_metrics": summary_metrics,
                "trends": trends,
                "insights": insights,
                "recommendations": recommendations
            }
            
            if grouped_metrics:
                result["grouped_metrics"] = grouped_metrics
                result["group_by"] = group_by
            
            if filters:
                result["filters"] = filters
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing delivery performance: {str(e)}")
            return {
                "error": str(e),
                "time_period": time_period,
                "summary_metrics": {}
            }
    
    async def analyze_delivery_exceptions(
        self,
        time_period: str = "last_30_days",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze delivery exceptions and issues.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with delivery exceptions analysis
        """
        try:
            # Get delivery exception data
            exception_data = await self._get_delivery_exceptions(
                time_period=time_period,
                client_id=client_id,
                connection_id=connection_id,
                filters=filters
            )
            
            # Calculate exception metrics
            exception_metrics = self._calculate_exception_metrics(exception_data)
            
            # Group exceptions by type
            exceptions_by_type = self._group_exceptions_by_type(exception_data)
            
            # Group exceptions by cause
            exceptions_by_cause = self._group_exceptions_by_cause(exception_data)
            
            # Calculate exception trends
            exception_trends = self._calculate_exception_trends(exception_data)
            
            # Generate insights
            insights = self._generate_exception_insights(
                exception_metrics=exception_metrics,
                exceptions_by_type=exceptions_by_type,
                exceptions_by_cause=exceptions_by_cause,
                exception_trends=exception_trends
            )
            
            # Generate recommendations
            recommendations = self._generate_exception_recommendations(
                exception_metrics=exception_metrics,
                exceptions_by_type=exceptions_by_type,
                exceptions_by_cause=exceptions_by_cause,
                exception_trends=exception_trends,
                insights=insights
            )
            
            # Prepare result
            result = {
                "time_period": time_period,
                "exception_metrics": exception_metrics,
                "exceptions_by_type": exceptions_by_type,
                "exceptions_by_cause": exceptions_by_cause,
                "exception_trends": exception_trends,
                "insights": insights,
                "recommendations": recommendations
            }
            
            if filters:
                result["filters"] = filters
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing delivery exceptions: {str(e)}")
            return {
                "error": str(e),
                "time_period": time_period,
                "exception_metrics": {}
            }
    
   async def analyze_delivery_costs(
        self,
        time_period: str = "last_30_days",
        group_by: str = "none",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze delivery costs.
        
        Args:
            time_period: Time period for analysis
            group_by: Group by dimension (none, carrier, region, customer, product)
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with delivery cost analysis
        """
        try:
            # Get delivery cost data
            cost_data = await self._get_delivery_cost_data(
                time_period=time_period,
                client_id=client_id,
                connection_id=connection_id,
                filters=filters
            )
            
            # Calculate cost metrics
            cost_metrics = self._calculate_cost_metrics(cost_data)
            
            # Group costs if requested
            grouped_costs = {}
            if group_by != "none" and group_by in cost_data:
                grouped_costs = self._group_cost_data(
                    cost_data=cost_data,
                    group_by=group_by
                )
            
            # Calculate cost trends
            cost_trends = self._calculate_cost_trends(cost_data)
            
            # Generate insights
            insights = self._generate_cost_insights(
                cost_metrics=cost_metrics,
                grouped_costs=grouped_costs,
                cost_trends=cost_trends,
                group_by=group_by
            )
            
            # Generate recommendations
            recommendations = self._generate_cost_recommendations(
                cost_metrics=cost_metrics,
                grouped_costs=grouped_costs,
                cost_trends=cost_trends,
                insights=insights
            )
            
            # Prepare result
            result = {
                "time_period": time_period,
                "cost_metrics": cost_metrics,
                "cost_trends": cost_trends,
                "insights": insights,
                "recommendations": recommendations
            }
            
            if grouped_costs:
                result["grouped_costs"] = grouped_costs
                result["group_by"] = group_by
            
            if filters:
                result["filters"] = filters
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing delivery costs: {str(e)}")
            return {
                "error": str(e),
                "time_period": time_period,
                "cost_metrics": {}
            }
    
    async def analyze_last_mile_performance(
        self,
        time_period: str = "last_30_days",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze last mile delivery performance.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with last mile performance analysis
        """
        try:
            # Get last mile delivery data
            last_mile_data = await self._get_last_mile_data(
                time_period=time_period,
                client_id=client_id,
                connection_id=connection_id,
                filters=filters
            )
            
            # Calculate last mile metrics
            last_mile_metrics = self._calculate_last_mile_metrics(last_mile_data)
            
            # Group by region
            performance_by_region = self._group_last_mile_by_region(last_mile_data)
            
            # Calculate last mile trends
            last_mile_trends = self._calculate_last_mile_trends(last_mile_data)
            
            # Generate insights
            insights = self._generate_last_mile_insights(
                last_mile_metrics=last_mile_metrics,
                performance_by_region=performance_by_region,
                last_mile_trends=last_mile_trends
            )
            
            # Generate recommendations
            recommendations = self._generate_last_mile_recommendations(
                last_mile_metrics=last_mile_metrics,
                performance_by_region=performance_by_region,
                last_mile_trends=last_mile_trends,
                insights=insights
            )
            
            # Prepare result
            result = {
                "time_period": time_period,
                "last_mile_metrics": last_mile_metrics,
                "performance_by_region": performance_by_region,
                "last_mile_trends": last_mile_trends,
                "insights": insights,
                "recommendations": recommendations
            }
            
            if filters:
                result["filters"] = filters
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing last mile performance: {str(e)}")
            return {
                "error": str(e),
                "time_period": time_period,
                "last_mile_metrics": {}
            }
    
    async def _get_delivery_data(
        self,
        time_period: str = "last_30_days",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get delivery data for analysis.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with delivery data
        """
        try:
            # Get data from database
            if client_id and connection_id:
                # Import interfaces
                from app.db.interfaces.delivery_interface import DeliveryInterface
                
                # Create interface
                delivery_interface = DeliveryInterface(client_id=client_id, connection_id=connection_id)
                
                # Get delivery data
                delivery_data = await delivery_interface.get_delivery_data(
                    time_period=time_period,
                    filters=filters
                )
                
                return delivery_data
            
            # Generate mock data for demonstration or testing
            delivery_data = self._generate_mock_delivery_data(
                time_period=time_period,
                filters=filters
            )
            
            return delivery_data
            
        except Exception as e:
            logger.error(f"Error getting delivery data: {str(e)}")
            
            # Generate mock data if an error occurs
            return self._generate_mock_delivery_data(
                time_period=time_period,
                filters=filters,
                error=True
            )
    
    async def _get_delivery_exceptions(
        self,
        time_period: str = "last_30_days",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get delivery exception data for analysis.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with delivery exception data
        """
        try:
            # Get data from database
            if client_id and connection_id:
                # Import interfaces
                from app.db.interfaces.delivery_interface import DeliveryInterface
                
                # Create interface
                delivery_interface = DeliveryInterface(client_id=client_id, connection_id=connection_id)
                
                # Get exception data
                exception_data = await delivery_interface.get_exception_data(
                    time_period=time_period,
                    filters=filters
                )
                
                return exception_data
            
            # Generate mock data for demonstration or testing
            exception_data = self._generate_mock_exception_data(
                time_period=time_period,
                filters=filters
            )
            
            return exception_data
            
        except Exception as e:
            logger.error(f"Error getting delivery exception data: {str(e)}")
            
            # Generate mock data if an error occurs
            return self._generate_mock_exception_data(
                time_period=time_period,
                filters=filters,
                error=True
            )
    
    async def _get_delivery_cost_data(
        self,
        time_period: str = "last_30_days",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get delivery cost data for analysis.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with delivery cost data
        """
        try:
            # Get data from database
            if client_id and connection_id:
                # Import interfaces
                from app.db.interfaces.delivery_interface import DeliveryInterface
                
                # Create interface
                delivery_interface = DeliveryInterface(client_id=client_id, connection_id=connection_id)
                
                # Get cost data
                cost_data = await delivery_interface.get_cost_data(
                    time_period=time_period,
                    filters=filters
                )
                
                return cost_data
            
            # Generate mock data for demonstration or testing
            cost_data = self._generate_mock_cost_data(
                time_period=time_period,
                filters=filters
            )
            
            return cost_data
            
        except Exception as e:
            logger.error(f"Error getting delivery cost data: {str(e)}")
            
            # Generate mock data if an error occurs
            return self._generate_mock_cost_data(
                time_period=time_period,
                filters=filters,
                error=True
            )
    
    async def _get_last_mile_data(
        self,
        time_period: str = "last_30_days",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get last mile delivery data for analysis.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            filters: Optional filters to apply
            
        Returns:
            Dictionary with last mile delivery data
        """
        try:
            # Get data from database
            if client_id and connection_id:
                # Import interfaces
                from app.db.interfaces.delivery_interface import DeliveryInterface
                
                # Create interface
                delivery_interface = DeliveryInterface(client_id=client_id, connection_id=connection_id)
                
                # Get last mile data
                last_mile_data = await delivery_interface.get_last_mile_data(
                    time_period=time_period,
                    filters=filters
                )
                
                return last_mile_data
            
            # Generate mock data for demonstration or testing
            last_mile_data = self._generate_mock_last_mile_data(
                time_period=time_period,
                filters=filters
            )
            
            return last_mile_data
            
        except Exception as e:
            logger.error(f"Error getting last mile delivery data: {str(e)}")
            
            # Generate mock data if an error occurs
            return self._generate_mock_last_mile_data(
                time_period=time_period,
                filters=filters,
                error=True
            )
    
    def _calculate_summary_metrics(
        self,
        delivery_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate summary metrics from delivery data.
        
        Args:
            delivery_data: Delivery data
            
        Returns:
            Dictionary with summary metrics
        """
        summary_metrics = {}
        
        # Extract key metrics
        if "deliveries" in delivery_data:
            deliveries = delivery_data["deliveries"]
            
            # On-time delivery rate
            if "total" in deliveries and "on_time" in deliveries:
                total_deliveries = deliveries["total"]
                on_time_deliveries = deliveries["on_time"]
                
                on_time_rate = (on_time_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
                
                summary_metrics["on_time_delivery"] = {
                    "value": round(on_time_rate, 1),
                    "total_deliveries": total_deliveries,
                    "on_time_deliveries": on_time_deliveries,
                    "unit": "%",
                    "performance_level": self._get_performance_level(
                        "on_time_delivery", on_time_rate
                    )
                }
            
            # Delivery accuracy
            if "total" in deliveries and "accurate" in deliveries:
                total_deliveries = deliveries["total"]
                accurate_deliveries = deliveries["accurate"]
                
                accuracy_rate = (accurate_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
                
                summary_metrics["delivery_accuracy"] = {
                    "value": round(accuracy_rate, 1),
                    "total_deliveries": total_deliveries,
                    "accurate_deliveries": accurate_deliveries,
                    "unit": "%",
                    "performance_level": self._get_performance_level(
                        "delivery_accuracy", accuracy_rate
                    )
                }
        
        # Cycle time metrics
        if "cycle_times" in delivery_data:
            cycle_times = delivery_data["cycle_times"]
            
            # Order cycle time
            if "avg_actual" in cycle_times and "avg_target" in cycle_times:
                avg_actual = cycle_times["avg_actual"]
                avg_target = cycle_times["avg_target"]
                
                # Calculate ratio of actual to target
                cycle_time_ratio = avg_actual / avg_target if avg_target > 0 else 1
                
                summary_metrics["order_cycle_time"] = {
                    "value": round(avg_actual, 1),
                    "target": round(avg_target, 1),
                    "ratio": round(cycle_time_ratio, 2),
                    "unit": "days",
                    "performance_level": self._get_performance_level(
                        "order_cycle_time", cycle_time_ratio
                    )
                }
        
        # Cost metrics
        if "costs" in delivery_data:
            costs = delivery_data["costs"]
            
            # Cost per delivery
            if "total_cost" in costs and "total_deliveries" in costs:
                total_cost = costs["total_cost"]
                total_deliveries = costs["total_deliveries"]
                
                cost_per_delivery = total_cost / total_deliveries if total_deliveries > 0 else 0
                
                summary_metrics["cost_per_delivery"] = {
                    "value": round(cost_per_delivery, 2),
                    "total_cost": round(total_cost, 2),
                    "total_deliveries": total_deliveries,
                    "unit": "$"
                }
        
        # Exception metrics
        if "exceptions" in delivery_data:
            exceptions = delivery_data["exceptions"]
            
            # Exception rate
            if "total_deliveries" in exceptions and "total_exceptions" in exceptions:
                total_deliveries = exceptions["total_deliveries"]
                total_exceptions = exceptions["total_exceptions"]
                
                exception_rate = (total_exceptions / total_deliveries * 100) if total_deliveries > 0 else 0
                
                summary_metrics["exception_rate"] = {
                    "value": round(exception_rate, 1),
                    "total_deliveries": total_deliveries,
                    "total_exceptions": total_exceptions,
                    "unit": "%"
                }
        
        return summary_metrics
    
    def _group_delivery_data(
        self,
        delivery_data: Dict[str, Any],
        group_by: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group delivery data by specified dimension.
        
        Args:
            delivery_data: Delivery data
            group_by: Dimension to group by
            
        Returns:
            Dictionary with grouped metrics
        """
        grouped_metrics = {}
        
        # Extract group data
        if group_by in delivery_data and "by_group" in delivery_data:
            group_data = delivery_data["by_group"].get(group_by, {})
            
            for group_name, group_values in group_data.items():
                # Calculate metrics for this group
                group_metrics = {}
                
                # On-time delivery rate
                if "total_deliveries" in group_values and "on_time_deliveries" in group_values:
                    total_deliveries = group_values["total_deliveries"]
                    on_time_deliveries = group_values["on_time_deliveries"]
                    
                    on_time_rate = (on_time_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
                    
                    group_metrics["on_time_delivery"] = {
                        "value": round(on_time_rate, 1),
                        "total_deliveries": total_deliveries,
                        "on_time_deliveries": on_time_deliveries,
                        "unit": "%",
                        "performance_level": self._get_performance_level(
                            "on_time_delivery", on_time_rate
                        )
                    }
                
                # Delivery accuracy
                if "total_deliveries" in group_values and "accurate_deliveries" in group_values:
                    total_deliveries = group_values["total_deliveries"]
                    accurate_deliveries = group_values["accurate_deliveries"]
                    
                    accuracy_rate = (accurate_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
                    
                    group_metrics["delivery_accuracy"] = {
                        "value": round(accuracy_rate, 1),
                        "total_deliveries": total_deliveries,
                        "accurate_deliveries": accurate_deliveries,
                        "unit": "%",
                        "performance_level": self._get_performance_level(
                            "delivery_accuracy", accuracy_rate
                        )
                    }
                
                # Cost per delivery
                if "total_cost" in group_values and "total_deliveries" in group_values:
                    total_cost = group_values["total_cost"]
                    total_deliveries = group_values["total_deliveries"]
                    
                    cost_per_delivery = total_cost / total_deliveries if total_deliveries > 0 else 0
                    
                    group_metrics["cost_per_delivery"] = {
                        "value": round(cost_per_delivery, 2),
                        "total_cost": round(total_cost, 2),
                        "total_deliveries": total_deliveries,
                        "unit": "$"
                    }
                
                # Add group metrics
                grouped_metrics[group_name] = group_metrics
        
        return grouped_metrics
    
    def _calculate_trends(
        self,
        delivery_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate trends from delivery data.
        
        Args:
            delivery_data: Delivery data
            
        Returns:
            Dictionary with trend data
        """
        trends = {}
        
        # Extract trend data
        if "trends" in delivery_data:
            trend_data = delivery_data["trends"]
            
            # On-time delivery trend
            if "on_time_delivery" in trend_data:
                on_time_trend = trend_data["on_time_delivery"]
                
                if "values" in on_time_trend and "periods" in on_time_trend:
                    values = on_time_trend["values"]
                    periods = on_time_trend["periods"]
                    
                    # Calculate trend direction and magnitude
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values)
                    
                    trends["on_time_delivery"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # Delivery accuracy trend
            if "delivery_accuracy" in trend_data:
                accuracy_trend = trend_data["delivery_accuracy"]
                
                if "values" in accuracy_trend and "periods" in accuracy_trend:
                    values = accuracy_trend["values"]
                    periods = accuracy_trend["periods"]
                    
                    # Calculate trend direction and magnitude
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values)
                    
                    trends["delivery_accuracy"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # Order cycle time trend
            if "order_cycle_time" in trend_data:
                cycle_time_trend = trend_data["order_cycle_time"]
                
                if "values" in cycle_time_trend and "periods" in cycle_time_trend:
                    values = cycle_time_trend["values"]
                    periods = cycle_time_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for cycle time, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    trends["order_cycle_time"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # Cost per delivery trend
            if "cost_per_delivery" in trend_data:
                cost_trend = trend_data["cost_per_delivery"]
                
                if "values" in cost_trend and "periods" in cost_trend:
                    values = cost_trend["values"]
                    periods = cost_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for cost, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    trends["cost_per_delivery"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
        
        return trends
    
    def _calculate_trend_direction(
        self,
        values: List[float],
        decreasing_is_positive: bool = False
    ) -> Tuple[str, float]:
        """
        Calculate trend direction and magnitude.
        
        Args:
            values: List of metric values
            decreasing_is_positive: Whether decreasing trend is positive
            
        Returns:
            Tuple of (direction, magnitude)
        """
        if not values or len(values) < 2:
            return "stable", 0.0
        
        # Calculate simple linear regression
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Calculate percentage change
        first_value = values[0]
        last_value = values[-1]
        
        percent_change = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0
        
        # Determine direction
        if abs(percent_change) < 2:
            direction = "stable"
        elif percent_change > 0:
            direction = "improving" if not decreasing_is_positive else "declining"
        else:
            direction = "declining" if not decreasing_is_positive else "improving"
        
        return direction, abs(percent_change)
    
    def _generate_insights(
        self,
        summary_metrics: Dict[str, Any],
        grouped_metrics: Dict[str, Dict[str, Any]],
        trends: Dict[str, Any],
        group_by: str
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from delivery metrics.
        
        Args:
            summary_metrics: Summary metrics
            grouped_metrics: Grouped metrics
            trends: Trend data
            group_by: Group by dimension
            
        Returns:
            List of insights
        """
        insights = []
        
        # Generate overall performance insights
        if "on_time_delivery" in summary_metrics:
            on_time_value = summary_metrics["on_time_delivery"]["value"]
            performance_level = summary_metrics["on_time_delivery"]["performance_level"]
            
            if performance_level in ("excellent", "good"):
                insights.append({
                    "type": "positive",
                    "category": "on_time_delivery",
                    "insight": f"Strong on-time delivery performance at {on_time_value}%"
                })
            elif performance_level in ("below_average", "poor"):
                insights.append({
                    "type": "negative",
                    "category": "on_time_delivery",
                    "insight": f"Poor on-time delivery performance at {on_time_value}%"
                })
        
        if "delivery_accuracy" in summary_metrics:
            accuracy_value = summary_metrics["delivery_accuracy"]["value"]
            performance_level = summary_metrics["delivery_accuracy"]["performance_level"]
            
            if performance_level in ("excellent", "good"):
                insights.append({
                    "type": "positive",
                    "category": "delivery_accuracy",
                    "insight": f"High delivery accuracy at {accuracy_value}%"
                })
            elif performance_level in ("below_average", "poor"):
                insights.append({
                    "type": "negative",
                    "category": "delivery_accuracy",
                    "insight": f"Low delivery accuracy at {accuracy_value}%"
                })
        
        if "order_cycle_time" in summary_metrics:
            cycle_time = summary_metrics["order_cycle_time"]["value"]
            target = summary_metrics["order_cycle_time"]["target"]
            ratio = summary_metrics["order_cycle_time"]["ratio"]
            performance_level = summary_metrics["order_cycle_time"]["performance_level"]
            
            if performance_level in ("excellent", "good"):
                insights.append({
                    "type": "positive",
                    "category": "order_cycle_time",
                    "insight": f"Fast order cycle time at {cycle_time} days ({ratio:.2f}x target)"
                })
            elif performance_level in ("below_average", "poor"):
                insights.append({
                    "type": "negative",
                    "category": "order_cycle_time",
                    "insight": f"Slow order cycle time at {cycle_time} days ({ratio:.2f}x target)"
                })
        
        # Generate trend insights
        if "on_time_delivery" in trends:
            direction = trends["on_time_delivery"]["direction"]
            magnitude = trends["on_time_delivery"]["magnitude"]
            
            if direction == "improving" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "on_time_delivery",
                    "insight": f"On-time delivery improving by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "on_time_delivery",
                    "insight": f"On-time delivery declining by {magnitude:.1f}%"
                })
        
        if "delivery_accuracy" in trends:
            direction = trends["delivery_accuracy"]["direction"]
            magnitude = trends["delivery_accuracy"]["magnitude"]
            
            if direction == "improving" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "delivery_accuracy",
                    "insight": f"Delivery accuracy improving by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "delivery_accuracy",
                    "insight": f"Delivery accuracy declining by {magnitude:.1f}%"
                })
        
        if "order_cycle_time" in trends:
            direction = trends["order_cycle_time"]["direction"]
            magnitude = trends["order_cycle_time"]["magnitude"]
            
            if direction == "improving" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "order_cycle_time",
                    "insight": f"Order cycle time improving by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "order_cycle_time",
                    "insight": f"Order cycle time declining by {magnitude:.1f}%"
                })
        
        # Generate grouped insights
        if grouped_metrics and group_by != "none":
            # Find best and worst performers
            best_performers = {}
            worst_performers = {}
            
            for metric in ["on_time_delivery", "delivery_accuracy", "cost_per_delivery"]:
                if not any(metric in group_metrics for group_metrics in grouped_metrics.values()):
                    continue
                
                # Find groups with this metric
                groups_with_metric = {
                    group_name: group_metrics[metric]["value"]
                    for group_name, group_metrics in grouped_metrics.items()
                    if metric in group_metrics
                }
                
                if not groups_with_metric:
                    continue
                
                # Find best performer
                best_value = max(groups_with_metric.values()) if metric != "cost_per_delivery" else min(groups_with_metric.values())
                best_group = next(
                    group_name for group_name, value in groups_with_metric.items()
                    if value == best_value
                )
                
                best_performers[metric] = {
                    "group": best_group,
                    "value": best_value
                }
                
                # Find worst performer
                worst_value = min(groups_with_metric.values()) if metric != "cost_per_delivery" else max(groups_with_metric.values())
                worst_group = next(
                    group_name for group_name, value in groups_with_metric.items()
                    if value == worst_value
                )
                
                worst_performers[metric] = {
                    "group": worst_group,
                    "value": worst_value
                }
            
            # Add insights for best performers
            for metric, performer in best_performers.items():
                if metric == "on_time_delivery":
                    insights.append({
                        "type": "group",
                        "category": "on_time_delivery",
                        "insight": f"Best on-time delivery: {performer['group']} at {performer['value']}%"
                    })
                elif metric == "delivery_accuracy":
                    insights.append({
                        "type": "group",
                        "category": "delivery_accuracy",
                        "insight": f"Best delivery accuracy: {performer['group']} at {performer['value']}%"
                    })
                elif metric == "cost_per_delivery":
                    insights.append({
                        "type": "group",
                        "category": "cost",
                        "insight": f"Lowest cost per delivery: {performer['group']} at ${performer['value']}"
                    })
            
            # Add insights for worst performers
            for metric, performer in worst_performers.items():
                if metric == "on_time_delivery" and performer["value"] < 80:
                    insights.append({
                        "type": "group",
                        "category": "on_time_delivery",
                        "insight": f"Worst on-time delivery: {performer['group']} at {performer['value']}%"
                    })
                elif metric == "delivery_accuracy" and performer["value"] < 90:
                    insights.append({
                        "type": "group",
                        "category": "delivery_accuracy",
                        "insight": f"Worst delivery accuracy: {performer['group']} at {performer['value']}%"
                    })
                elif metric == "cost_per_delivery":
                    # Only add if the cost is significantly higher than average
                    if "cost_per_delivery" in summary_metrics:
                        avg_cost = summary_metrics["cost_per_delivery"]["value"]
                        if performer["value"] > avg_cost * 1.2:  # 20% higher than average
                            insights.append({
                                "type": "group",
                                "category": "cost",
                                "insight": f"Highest cost per delivery: {performer['group']} at ${performer['value']}"
                            })
        
        return insights
    
    def _generate_recommendations(
        self,
        summary_metrics: Dict[str, Any],
        grouped_metrics: Dict[str, Dict[str, Any]],
        trends: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on delivery metrics and insights.
        
        Args:
            summary_metrics: Summary metrics
            grouped_metrics: Grouped metrics
            trends: Trend data
            insights: Generated insights
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate recommendations based on overall performance
        if "on_time_delivery" in summary_metrics:
            on_time_value = summary_metrics["on_time_delivery"]["value"]
            performance_level = summary_metrics["on_time_delivery"]["performance_level"]
            
            if performance_level in ("below_average", "poor"):
                recommendations.append({
                    "priority": "high",
                    "category": "on_time_delivery",
                    "recommendation": "Implement daily carrier performance reviews to address on-time delivery issues",
                    "expected_impact": "Potential improvement of 5-10% in on-time delivery rate"
                })
        
        if "delivery_accuracy" in summary_metrics:
            accuracy_value = summary_metrics["delivery_accuracy"]["value"]
            performance_level = summary_metrics["delivery_accuracy"]["performance_level"]
            
            if performance_level in ("below_average", "poor"):
                recommendations.append({
                    "priority": "high",
                    "category": "delivery_accuracy",
                    "recommendation": "Implement barcode scanning at all handoff points to improve delivery accuracy",
                    "expected_impact": "Potential improvement of 3-8% in delivery accuracy"
                })
        
        if "order_cycle_time" in summary_metrics:
            cycle_time = summary_metrics["order_cycle_time"]["value"]
            target = summary_metrics["order_cycle_time"]["target"]
            ratio = summary_metrics["order_cycle_time"]["ratio"]
            performance_level = summary_metrics["order_cycle_time"]["performance_level"]
            
            if performance_level in ("below_average", "poor"):
                recommendations.append({
                    "priority": "medium",
                    "category": "order_cycle_time",
                    "recommendation": "Review and optimize order processing workflows to reduce cycle time",
                    "expected_impact": "Potential reduction of 10-15% in order cycle time"
                })
        
        # Generate recommendations based on trends
        for metric, trend_data in trends.items():
            direction = trend_data["direction"]
            magnitude = trend_data["magnitude"]
            
            if direction == "declining" and magnitude >= 5:
                if metric == "on_time_delivery":
                    recommendations.append({
                        "priority": "high",
                        "category": "on_time_delivery",
                        "recommendation": "Investigate root causes of declining on-time delivery performance",
                        "expected_impact": "Identify and address factors causing the negative trend"
                    })
                elif metric == "delivery_accuracy":
                    recommendations.append({
                        "priority": "high",
                        "category": "delivery_accuracy",
                        "recommendation": "Audit delivery processes to identify causes of declining accuracy",
                        "expected_impact": "Identify and address factors causing the negative trend"
                    })
                elif metric == "order_cycle_time" and direction == "declining":  # Note: for cycle time, declining means increasing time
                    recommendations.append({
                        "priority": "medium",
                        "category": "order_cycle_time",
                        "recommendation": "Analyze bottlenecks in order fulfillment process",
                        "expected_impact": "Identify and address delays in the process"
                    })
        
        # Generate recommendations based on grouped insights
        negative_group_insights = [
            insight for insight in insights
            if insight["type"] == "group" and "Worst" in insight["insight"]
        ]
        
        for insight in negative_group_insights:
            category = insight["category"]
            # Extract group name from insight text
            group_name = insight["insight"].split(": ")[1].split(" at ")[0]
            
            if category == "on_time_delivery":
                recommendations.append({
                    "priority": "medium",
                    "category": "on_time_delivery",
                    "recommendation": f"Develop improvement plan for {group_name} to address on-time delivery performance",
                    "expected_impact": "Target 5-10% improvement in on-time delivery for this group"
                })
            elif category == "delivery_accuracy":
                recommendations.append({
                    "priority": "medium",
                    "category": "delivery_accuracy",
                    "recommendation": f"Provide additional training and resources for {group_name} to improve delivery accuracy",
                    "expected_impact": "Target 5-8% improvement in delivery accuracy for this group"
                })
            elif category == "cost":
                recommendations.append({
                    "priority": "medium",
                    "category": "cost",
                    "recommendation": f"Review delivery costs for {group_name} to identify cost reduction opportunities",
                    "expected_impact": "Potential 10-15% cost reduction for this group"
                })
        
        # Add general recommendations
        recommendations.append({
            "priority": "low",
            "category": "general",
            "recommendation": "Implement regular delivery performance reviews with all carriers",
            "expected_impact": "Improved visibility and accountability across delivery operations"
        })
        
        recommendations.append({
            "priority": "low",
            "category": "general",
            "recommendation": "Enhance delivery tracking and visibility for customers",
            "expected_impact": "Improved customer experience and reduced WISMO (Where Is My Order) inquiries"
        })
        
        return recommendations
    
    def _calculate_exception_metrics(
        self,
        exception_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate exception metrics from delivery exception data.
        
        Args:
            exception_data: Delivery exception data
            
        Returns:
            Dictionary with exception metrics
        """
        exception_metrics = {}
        
        # Extract key metrics
        if "summary" in exception_data:
            summary = exception_data["summary"]
            
            # Overall exception rate
            if "total_deliveries" in summary and "total_exceptions" in summary:
                total_deliveries = summary["total_deliveries"]
                total_exceptions = summary["total_exceptions"]
                
                exception_rate = (total_exceptions / total_deliveries * 100) if total_deliveries > 0 else 0
                
                exception_metrics["exception_rate"] = {
                    "value": round(exception_rate, 1),
                    "total_deliveries": total_deliveries,
                    "total_exceptions": total_exceptions,
                    "unit": "%"
                }
            
            # Average resolution time
            if "avg_resolution_time" in summary:
                avg_resolution_time = summary["avg_resolution_time"]
                
                exception_metrics["avg_resolution_time"] = {
                    "value": round(avg_resolution_time, 1),
                    "unit": "hours"
                }
            
            # Unresolved exceptions
            if "unresolved_exceptions" in summary and "total_exceptions" in summary:
                unresolved_exceptions = summary["unresolved_exceptions"]
                total_exceptions = summary["total_exceptions"]
                
                unresolved_rate = (unresolved_exceptions / total_exceptions * 100) if total_exceptions > 0 else 0
                
                exception_metrics["unresolved_rate"] = {
                    "value": round(unresolved_rate, 1),
                    "unresolved_exceptions": unresolved_exceptions,
                    "total_exceptions": total_exceptions,
                    "unit": "%"
                }
            
            # Customer impact metrics
            if "customer_impact" in summary:
                customer_impact = summary["customer_impact"]
                
                exception_metrics["customer_impact"] = customer_impact
        
        return exception_metrics
    
    def _group_exceptions_by_type(
        self,
        exception_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group exceptions by type.
        
        Args:
            exception_data: Exception data
            
        Returns:
            Dictionary with exceptions grouped by type
        """
        grouped_exceptions = {}
        
        # Extract exception types
        if "by_type" in exception_data:
            by_type = exception_data["by_type"]
            
            for exception_type, type_data in by_type.items():
                # Extract key metrics for this type
                type_metrics = {}
                
                if "count" in type_data and "total_exceptions" in type_data:
                    count = type_data["count"]
                    total_exceptions = type_data["total_exceptions"]
                    
                    percentage = (count / total_exceptions * 100) if total_exceptions > 0 else 0
                    
                    type_metrics["count"] = count
                    type_metrics["percentage"] = round(percentage, 1)
                    type_metrics["unit"] = "%"
                
                if "avg_resolution_time" in type_data:
                    type_metrics["avg_resolution_time"] = round(type_data["avg_resolution_time"], 1)
                
                if "description" in type_data:
                    type_metrics["description"] = type_data["description"]
                
                grouped_exceptions[exception_type] = type_metrics
        
        return grouped_exceptions
    
    def _group_exceptions_by_cause(
        self,
        exception_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group exceptions by root cause.
        
        Args:
            exception_data: Exception data
            
        Returns:
            Dictionary with exceptions grouped by cause
        """
        grouped_by_cause = {}
        
        # Extract exception causes
        if "by_cause" in exception_data:
            by_cause = exception_data["by_cause"]
            
            for cause, cause_data in by_cause.items():
                # Extract key metrics for this cause
                cause_metrics = {}
                
                if "count" in cause_data and "total_exceptions" in cause_data:
                    count = cause_data["count"]
                    total_exceptions = cause_data["total_exceptions"]
                    
                    percentage = (count / total_exceptions * 100) if total_exceptions > 0 else 0
                    
                    cause_metrics["count"] = count
                    cause_metrics["percentage"] = round(percentage, 1)
                    cause_metrics["unit"] = "%"
                
                if "description" in cause_data:
                    cause_metrics["description"] = cause_data["description"]
                
                if "preventable" in cause_data:
                    cause_metrics["preventable"] = cause_data["preventable"]
                
                grouped_by_cause[cause] = cause_metrics
        
        return grouped_by_cause
    
    def _calculate_exception_trends(
        self,
        exception_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate exception trends.
        
        Args:
            exception_data: Exception data
            
        Returns:
            Dictionary with exception trends
        """
        exception_trends = {}
        
        # Extract trend data
        if "trends" in exception_data:
            trend_data = exception_data["trends"]
            
            # Exception rate trend
            if "exception_rate" in trend_data:
                rate_trend = trend_data["exception_rate"]
                
                if "values" in rate_trend and "periods" in rate_trend:
                    values = rate_trend["values"]
                    periods = rate_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for exception rate, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    exception_trends["exception_rate"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # Resolution time trend
            if "resolution_time" in trend_data:
                time_trend = trend_data["resolution_time"]
                
                if "values" in time_trend and "periods" in time_trend:
                    values = time_trend["values"]
                    periods = time_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for resolution time, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    exception_trends["resolution_time"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
        
        return exception_trends
    
    def _generate_exception_insights(
        self,
        exception_metrics: Dict[str, Any],
        exceptions_by_type: Dict[str, Dict[str, Any]],
        exceptions_by_cause: Dict[str, Dict[str, Any]],
        exception_trends: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from exception data.
        
        Args:
            exception_metrics: Exception metrics
            exceptions_by_type: Exceptions grouped by type
            exceptions_by_cause: Exceptions grouped by cause
            exception_trends: Exception trends
            
        Returns:
            List of insights
        """
        insights = []
        
        # Generate insights based on overall metrics
        if "exception_rate" in exception_metrics:
            exception_rate = exception_metrics["exception_rate"]["value"]
            
            if exception_rate > 10:
                insights.append({
                    "type": "negative",
                    "category": "exception_rate",
                    "insight": f"High overall exception rate of {exception_rate}%"
                })
            elif exception_rate < 3:
                insights.append({
                    "type": "positive",
                    "category": "exception_rate",
                    "insight": f"Low overall exception rate of {exception_rate}%"
                })
        
        if "avg_resolution_time" in exception_metrics:
            resolution_time = exception_metrics["avg_resolution_time"]["value"]
            
            if resolution_time > 24:
                insights.append({
                    "type": "negative",
                    "category": "resolution_time",
                    "insight": f"Long average exception resolution time of {resolution_time} hours"
                })
            elif resolution_time < 4:
                insights.append({
                    "type": "positive",
                    "category": "resolution_time",
                    "insight": f"Fast average exception resolution time of {resolution_time} hours"
                })
        
        if "unresolved_rate" in exception_metrics:
            unresolved_rate = exception_metrics["unresolved_rate"]["value"]
            
            if unresolved_rate > 15:
                insights.append({
                    "type": "negative",
                    "category": "unresolved_rate",
                    "insight": f"High rate of unresolved exceptions at {unresolved_rate}%"
                })
            elif unresolved_rate < 5:
                insights.append({
                    "type": "positive",
                    "category": "unresolved_rate",
                    "insight": f"Low rate of unresolved exceptions at {unresolved_rate}%"
                })
        
        # Generate insights about top exception types
        if exceptions_by_type:
            # Sort exception types by count
            sorted_types = sorted(
                exceptions_by_type.items(),
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )
            
            # Get top exception types
            top_types = sorted_types[:3]
            
            for exception_type, type_data in top_types:
                percentage = type_data.get("percentage", 0)
                
                if percentage >= 15:
                    insights.append({
                        "type": "type",
                        "category": "exception_type",
                        "insight": f"'{exception_type}' is a major exception type, representing {percentage}% of all exceptions"
                    })
        
        # Generate insights about top exception causes
        if exceptions_by_cause:
            # Sort exception causes by count
            sorted_causes = sorted(
                exceptions_by_cause.items(),
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )
            
            # Get top preventable causes
            top_preventable_causes = [
                (cause, cause_data) for cause, cause_data in sorted_causes
                if cause_data.get("preventable", False)
            ][:3]
            
            for cause, cause_data in top_preventable_causes:
                percentage = cause_data.get("percentage", 0)
                
                if percentage >= 10:
                    insights.append({
                        "type": "cause",
                        "category": "preventable_cause",
                        "insight": f"'{cause}' is a major preventable cause, accounting for {percentage}% of exceptions"
                    })
        
        # Generate trend insights
        if "exception_rate" in exception_trends:
            direction = exception_trends["exception_rate"]["direction"]
            magnitude = exception_trends["exception_rate"]["magnitude"]
            
            if direction == "improving" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "exception_rate",
                    "insight": f"Exception rate improving (decreasing) by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "exception_rate",
                    "insight": f"Exception rate declining (increasing) by {magnitude:.1f}%"
                })
        
        if "resolution_time" in exception_trends:
            direction = exception_trends["resolution_time"]["direction"]
            magnitude = exception_trends["resolution_time"]["magnitude"]
            
            if direction == "improving" and magnitude >= 10:
                insights.append({
                    "type": "trend",
                    "category": "resolution_time",
                    "insight": f"Resolution time improving (decreasing) by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 10:
                insights.append({
                    "type": "trend",
                    "category": "resolution_time",
                    "insight": f"Resolution time declining (increasing) by {magnitude:.1f}%"
                })
        
        return insights
    
    def _generate_exception_recommendations(
        self,
        exception_metrics: Dict[str, Any],
        exceptions_by_type: Dict[str, Dict[str, Any]],
        exceptions_by_cause: Dict[str, Dict[str, Any]],
        exception_trends: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on exception data.
        
        Args:
            exception_metrics: Exception metrics
            exceptions_by_type: Exceptions grouped by type
            exceptions_by_cause: Exceptions grouped by cause
            exception_trends: Exception trends
            insights: Generated insights
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate recommendations based on overall metrics
        if "exception_rate" in exception_metrics:
            exception_rate = exception_metrics["exception_rate"]["value"]
            
            if exception_rate > 8:
                recommendations.append({
                    "priority": "high",
                    "category": "exception_rate",
                    "recommendation": "Implement daily exception review process to address high exception rate",
                    "expected_impact": "Potential reduction of 20-30% in exception rate"
                })
        
        if "avg_resolution_time" in exception_metrics:
            resolution_time = exception_metrics["avg_resolution_time"]["value"]
            
            if resolution_time > 12:
                recommendations.append({
                    "priority": "medium",
                    "category": "resolution_time",
                    "recommendation": "Establish Service Level Agreements (SLAs) for exception resolution",
                    "expected_impact": "Reduce average resolution time by 30-40%"
                })
        
        if "unresolved_rate" in exception_metrics:
            unresolved_rate = exception_metrics["unresolved_rate"]["value"]
            
            if unresolved_rate > 10:
                recommendations.append({
                    "priority": "high",
                    "category": "unresolved_rate",
                    "recommendation": "Implement escalation process for exceptions unresolved after 24 hours",
                    "expected_impact": "Reduce unresolved exception rate by 50-60%"
                })
        
        # Generate recommendations based on top exception types
        if exceptions_by_type:
            # Sort exception types by count
            sorted_types = sorted(
                exceptions_by_type.items(),
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )
            
            # Get top exception type
            if sorted_types:
                top_type, top_type_data = sorted_types[0]
                percentage = top_type_data.get("percentage", 0)
                
                if percentage >= 20:
                    recommendations.append({
                        "priority": "high",
                        "category": "exception_type",
                        "recommendation": f"Develop specific action plan to address '{top_type}' exceptions",
                        "expected_impact": f"Potential reduction of 30-40% in '{top_type}' exceptions"
                    })
        
        # Generate recommendations based on preventable causes
        if exceptions_by_cause:
            # Find preventable causes
            preventable_causes = [
                (cause, cause_data) for cause, cause_data in exceptions_by_cause.items()
                if cause_data.get("preventable", False)
            ]
            
            # Sort by count
            sorted_preventable = sorted(
                preventable_causes,
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )
            
            # Recommend addressing top preventable causes
            for cause, cause_data in sorted_preventable[:2]:
                percentage = cause_data.get("percentage", 0)
                
                if percentage >= 5:
                    recommendations.append({
                        "priority": "medium",
                        "category": "preventable_cause",
                        "recommendation": f"Implement process improvements to address '{cause}' exceptions",
                        "expected_impact": f"Potential elimination of up to 80% of '{cause}' exceptions"
                    })
        
        # Generate recommendations based on trends
        if "exception_rate" in exception_trends:
            direction = exception_trends["exception_rate"]["direction"]
            
            if direction == "declining":
                recommendations.append({
                    "priority": "high",
                    "category": "exception_rate",
                    "recommendation": "Investigate root causes of increasing exception rate",
                    "expected_impact": "Identify and address factors causing the negative trend"
                })
        
        # Add general recommendations
        recommendations.append({
            "priority": "low",
            "category": "general",
            "recommendation": "Implement exception prevention training for warehouse and delivery staff",
            "expected_impact": "Improve awareness and reduce preventable exceptions"
        })
        
        recommendations.append({
            "priority": "low",
            "category": "general",
            "recommendation": "Develop exception resolution playbooks for common exception types",
            "expected_impact": "Standardize resolution process and reduce resolution time"
        })
        
        return recommendations
    
    def _calculate_cost_metrics(
        self,
        cost_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate cost metrics from delivery cost data.
        
        Args:
            cost_data: Delivery cost data
            
        Returns:
            Dictionary with cost metrics
        """
        cost_metrics = {}
        
        # Extract key metrics
        if "summary" in cost_data:
            summary = cost_data["summary"]
            
            # Cost per delivery
            if "total_cost" in summary and "total_deliveries" in summary:
                total_cost = summary["total_cost"]
                total_deliveries = summary["total_deliveries"]
                
                cost_per_delivery = total_cost / total_deliveries if total_deliveries > 0 else 0
                
                cost_metrics["cost_per_delivery"] = {
                    "value": round(cost_per_delivery, 2),
                    "total_cost": round(total_cost, 2),
                    "total_deliveries": total_deliveries,
                    "unit": "$"
                }
            
            # Cost per mile
            if "total_cost" in summary and "total_miles" in summary:
                total_cost = summary["total_cost"]
                total_miles = summary["total_miles"]
                
                cost_per_mile = total_cost / total_miles if total_miles > 0 else 0
                
                cost_metrics["cost_per_mile"] = {
                    "value": round(cost_per_mile, 2),
                    "total_cost": round(total_cost, 2),
                    "total_miles": total_miles,
                    "unit": "$"
                }
            
            # Cost breakdown
            if "cost_breakdown" in summary:
                cost_metrics["cost_breakdown"] = summary["cost_breakdown"]
        
        return cost_metrics
    
    def _group_cost_data(
        self,
        cost_data: Dict[str, Any],
        group_by: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group cost data by specified dimension.
        
        Args:
            cost_data: Cost data
            group_by: Dimension to group by
            
        Returns:
            Dictionary with grouped cost data
        """
        grouped_costs = {}
        
        # Extract group data
        if group_by in cost_data and "by_group" in cost_data:
            group_data = cost_data["by_group"].get(group_by, {})
            
            for group_name, group_values in group_data.items():
                # Calculate metrics for this group
                group_metrics = {}
                
                # Cost per delivery
                if "total_cost" in group_values and "total_deliveries" in group_values:
                    total_cost = group_values["total_cost"]
                    total_deliveries = group_values["total_deliveries"]
                    
                    cost_per_delivery = total_cost / total_deliveries if total_deliveries > 0 else 0
                    
                    group_metrics["cost_per_delivery"] = {
                        "value": round(cost_per_delivery, 2),
                        "total_cost": round(total_cost, 2),
                        "total_deliveries": total_deliveries,
                        "unit": "$"
                    }
                
                # Cost per mile
                if "total_cost" in group_values and "total_miles" in group_values:
                    total_cost = group_values["total_cost"]
                    total_miles = group_values["total_miles"]
                    
                    cost_per_mile = total_cost / total_miles if total_miles > 0 else 0
                    
                    group_metrics["cost_per_mile"] = {
                        "value": round(cost_per_mile, 2),
                        "total_cost": round(total_cost, 2),
                        "total_miles": total_miles,
                        "unit": "$"
                    }
                
                # Cost breakdown
                if "cost_breakdown" in group_values:
                    group_metrics["cost_breakdown"] = group_values["cost_breakdown"]
                
                grouped_costs[group_name] = group_metrics
        
        return grouped_costs
    
    def _calculate_cost_trends(
        self,
        cost_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate cost trends from delivery cost data.
        
        Args:
            cost_data: Delivery cost data
            
        Returns:
            Dictionary with cost trends
        """
        cost_trends = {}
        
        # Extract trend data
        if "trends" in cost_data:
            trend_data = cost_data["trends"]
            
            # Cost per delivery trend
            if "cost_per_delivery" in trend_data:
                delivery_trend = trend_data["cost_per_delivery"]
                
                if "values" in delivery_trend and "periods" in delivery_trend:
                    values = delivery_trend["values"]
                    periods = delivery_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for cost, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    cost_trends["cost_per_delivery"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # Cost per mile trend
            if "cost_per_mile" in trend_data:
                mile_trend = trend_data["cost_per_mile"]
                
                if "values" in mile_trend and "periods" in mile_trend:
                    values = mile_trend["values"]
                    periods = mile_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for cost, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    cost_trends["cost_per_mile"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
        
        return cost_trends
    
    def _generate_cost_insights(
        self,
        cost_metrics: Dict[str, Any],
        grouped_costs: Dict[str, Dict[str, Any]],
        cost_trends: Dict[str, Any],
        group_by: str
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from delivery cost data.
        
        Args:
            cost_metrics: Cost metrics
            grouped_costs: Grouped cost data
            cost_trends: Cost trends
            group_by: Group by dimension
            
        Returns:
            List of insights
        """
        insights = []
        
        # Generate insights based on overall costs
        if "cost_per_delivery" in cost_metrics:
            cost_per_delivery = cost_metrics["cost_per_delivery"]["value"]
            
            insights.append({
                "type": "metric",
                "category": "cost_per_delivery",
                "insight": f"Average cost per delivery is ${cost_per_delivery}"
            })
        
        if "cost_per_mile" in cost_metrics:
            cost_per_mile = cost_metrics["cost_per_mile"]["value"]
            
            insights.append({
                "type": "metric",
                "category": "cost_per_mile",
                "insight": f"Average cost per mile is ${cost_per_mile}"
            })
        
        # Generate insights based on cost breakdown
        if "cost_breakdown" in cost_metrics:
            breakdown = cost_metrics["cost_breakdown"]
            
            # Find highest cost category
            highest_category = max(breakdown.items(), key=lambda x: x[1]["value"])
            category_name, category_data = highest_category
            percentage = category_data.get("percentage", 0)
            
            if percentage > 40:
                insights.append({
                    "type": "breakdown",
                    "category": "cost_category",
                    "insight": f"'{category_name}' represents {percentage}% of total delivery costs"
                })
        
        # Generate insights based on cost trends
        if "cost_per_delivery" in cost_trends:
            direction = cost_trends["cost_per_delivery"]["direction"]
            magnitude = cost_trends["cost_per_delivery"]["magnitude"]
            
            if direction == "improving" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "cost_per_delivery",
                    "insight": f"Cost per delivery decreasing by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "cost_per_delivery",
                    "insight": f"Cost per delivery increasing by {magnitude:.1f}%"
                })
        
        # Generate insights based on grouped costs
        if grouped_costs and group_by != "none":
            # Find highest and lowest cost groups
            if any("cost_per_delivery" in group_metrics for group_metrics in grouped_costs.values()):
                # Create mapping of group to cost per delivery
                group_costs = {
                    group_name: group_metrics["cost_per_delivery"]["value"]
                    for group_name, group_metrics in grouped_costs.items()
                    if "cost_per_delivery" in group_metrics
                }
                
                if group_costs:
                    # Find highest cost group
                    highest_group = max(group_costs.items(), key=lambda x: x[1])
                    group_name, group_cost = highest_group
                    
                    # Find lowest cost group
                    lowest_group = min(group_costs.items(), key=lambda x: x[1])
                    low_group_name, low_group_cost = lowest_group
                    
                    # Calculate cost difference percentage
                    if low_group_cost > 0:
                        diff_percent = ((group_cost - low_group_cost) / low_group_cost * 100)
                    else:
                        diff_percent = 0
                    
                    if diff_percent >= 30:
                        insights.append({
                            "type": "group",
                            "category": "cost_variation",
                            "insight": f"'{group_name}' has {diff_percent:.1f}% higher cost per delivery than '{low_group_name}'"
                        })
        
        return insights
    
    def _generate_cost_recommendations(
        self,
        cost_metrics: Dict[str, Any],
        grouped_costs: Dict[str, Dict[str, Any]],
        cost_trends: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on delivery cost data.
        
        Args:
            cost_metrics: Cost metrics
            grouped_costs: Grouped cost data
            cost_trends: Cost trends
            insights: Generated insights
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate recommendations based on cost trends
        if "cost_per_delivery" in cost_trends:
            direction = cost_trends["cost_per_delivery"]["direction"]
            magnitude = cost_trends["cost_per_delivery"]["magnitude"]
            
            if direction == "declining" and magnitude >= 8:
                recommendations.append({
                    "priority": "high",
                    "category": "cost_trend",
                    "recommendation": "Investigate drivers of increasing delivery costs",
                    "expected_impact": "Identify cost drivers and implement targeted cost reduction measures"
                })
        
        # Generate recommendations based on cost breakdown
        if "cost_breakdown" in cost_metrics:
            breakdown = cost_metrics["cost_breakdown"]
            
            # Find highest cost category
            highest_category = max(breakdown.items(), key=lambda x: x[1]["value"])
            category_name, category_data = highest_category
            percentage = category_data.get("percentage", 0)
            
            if percentage > 40:
                recommendations.append({
                    "priority": "high",
                    "category": "cost_category",
                    "recommendation": f"Develop cost reduction strategy for '{category_name}' costs",
                    "expected_impact": f"Target 10-15% reduction in '{category_name}' costs"
                })
        
        # Generate recommendations based on grouped costs
        high_cost_groups = []
        if grouped_costs and any("cost_per_delivery" in group_metrics for group_metrics in grouped_costs.values()):
            # Calculate average cost per delivery
            avg_cost = 0
            if "cost_per_delivery" in cost_metrics:
                avg_cost = cost_metrics["cost_per_delivery"]["value"]
            else:
                group_costs = [
                    group_metrics["cost_per_delivery"]["value"]
                    for group_metrics in grouped_costs.values()
                    if "cost_per_delivery" in group_metrics
                ]
                if group_costs:
                    avg_cost = sum(group_costs) / len(group_costs)
            
            # Find groups with significantly higher cost than average
            for group_name, group_metrics in grouped_costs.items():
                if "cost_per_delivery" in group_metrics:
                    group_cost = group_metrics["cost_per_delivery"]["value"]
                    
                    # If cost is 20% or more above average, add to high cost groups
                    if avg_cost > 0 and group_cost > avg_cost * 1.2:
                        high_cost_groups.append((group_name, group_cost, (group_cost - avg_cost) / avg_cost * 100))
            
            # Sort high cost groups by percentage above average
            high_cost_groups.sort(key=lambda x: x[2], reverse=True)
            
            # Generate recommendations for high cost groups
            for group_name, group_cost, percent_above in high_cost_groups[:3]:
                recommendations.append({
                    "priority": "medium",
                    "category": "high_cost_group",
                    "recommendation": f"Analyze and optimize delivery operations for '{group_name}'",
                    "expected_impact": f"Potential to reduce '{group_name}' delivery costs by 15-20%"
                })
        
        # Add general cost-saving recommendations
        recommendations.append({
            "priority": "medium",
            "category": "cost_saving",
            "recommendation": "Implement route optimization to reduce total miles traveled",
            "expected_impact": "Potential 10-15% reduction in transportation costs"
        })
        
        recommendations.append({
            "priority": "medium",
            "category": "cost_saving",
            "recommendation": "Review carrier contracts and negotiate volume-based discounts",
            "expected_impact": "Potential 5-10% reduction in carrier fees"
        })
        
        recommendations.append({
            "priority": "low",
            "category": "cost_saving",
            "recommendation": "Implement regular cost variance analysis",
            "expected_impact": "Better visibility into cost drivers and opportunities for savings"
        })
        
        return recommendations
        
    def _calculate_last_mile_metrics(
        self,
        last_mile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate last mile delivery metrics.
        
        Args:
            last_mile_data: Last mile delivery data
            
        Returns:
            Dictionary with last mile metrics
        """
        last_mile_metrics = {}
        
        # Extract key metrics
        if "summary" in last_mile_data:
            summary = last_mile_data["summary"]
            
            # On-time delivery rate
            if "total_deliveries" in summary and "on_time_deliveries" in summary:
                total_deliveries = summary["total_deliveries"]
                on_time_deliveries = summary["on_time_deliveries"]
                
                on_time_rate = (on_time_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
                
                last_mile_metrics["on_time_delivery"] = {
                    "value": round(on_time_rate, 1),
                    "total_deliveries": total_deliveries,
                    "on_time_deliveries": on_time_deliveries,
                    "unit": "%",
                    "performance_level": self._get_performance_level(
                        "on_time_delivery", on_time_rate
                    )
                }
            
            # First attempt delivery rate
            if "total_deliveries" in summary and "first_attempt_success" in summary:
                total_deliveries = summary["total_deliveries"]
                first_attempt_success = summary["first_attempt_success"]
                
                first_attempt_rate = (first_attempt_success / total_deliveries * 100) if total_deliveries > 0 else 0
                
                last_mile_metrics["first_attempt_rate"] = {
                    "value": round(first_attempt_rate, 1),
                    "total_deliveries": total_deliveries,
                    "first_attempt_success": first_attempt_success,
                    "unit": "%"
                }
            
            # Average time per delivery
            if "avg_delivery_time" in summary:
                avg_delivery_time = summary["avg_delivery_time"]
                
                last_mile_metrics["avg_delivery_time"] = {
                    "value": round(avg_delivery_time, 1),
                    "unit": "minutes"
                }
            
            # Average stops per route
            if "avg_stops_per_route" in summary:
                avg_stops_per_route = summary["avg_stops_per_route"]
                
                last_mile_metrics["avg_stops_per_route"] = {
                    "value": round(avg_stops_per_route, 1),
                    "unit": "stops"
                }
            
            # Customer satisfaction
            if "customer_satisfaction" in summary:
                customer_satisfaction = summary["customer_satisfaction"]
                
                last_mile_metrics["customer_satisfaction"] = {
                    "value": round(customer_satisfaction, 1),
                    "unit": "score (0-10)"
                }
        
        return last_mile_metrics
    
    def _group_last_mile_by_region(
        self,
        last_mile_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group last mile performance by region.
        
        Args:
            last_mile_data: Last mile delivery data
            
        Returns:
            Dictionary with performance by region
        """
        performance_by_region = {}
        
        # Extract region data
        if "by_region" in last_mile_data:
            region_data = last_mile_data["by_region"]
            
            for region_name, region_values in region_data.items():
                # Calculate metrics for this region
                region_metrics = {}
                
                # On-time delivery rate
                if "total_deliveries" in region_values and "on_time_deliveries" in region_values:
                    total_deliveries = region_values["total_deliveries"]
                    on_time_deliveries = region_values["on_time_deliveries"]
                    
                    on_time_rate = (on_time_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
                    
                    region_metrics["on_time_delivery"] = {
                        "value": round(on_time_rate, 1),
                        "total_deliveries": total_deliveries,
                        "on_time_deliveries": on_time_deliveries,
                        "unit": "%",
                        "performance_level": self._get_performance_level(
                            "on_time_delivery", on_time_rate
                        )
                    }
                
                # First attempt delivery rate
                if "total_deliveries" in region_values and "first_attempt_success" in region_values:
                    total_deliveries = region_values["total_deliveries"]
                    first_attempt_success = region_values["first_attempt_success"]
                    
                    first_attempt_rate = (first_attempt_success / total_deliveries * 100) if total_deliveries > 0 else 0
                    
                    region_metrics["first_attempt_rate"] = {
                        "value": round(first_attempt_rate, 1),
                        "total_deliveries": total_deliveries,
                        "first_attempt_success": first_attempt_success,
                        "unit": "%"
                    }
                
                # Average delivery time
                if "avg_delivery_time" in region_values:
                    region_metrics["avg_delivery_time"] = {
                        "value": round(region_values["avg_delivery_time"], 1),
                        "unit": "minutes"
                    }
                
                # Add metrics for this region
                performance_by_region[region_name] = region_metrics
        
        return performance_by_region
    
    def _calculate_last_mile_trends(
        self,
        last_mile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate last mile delivery trends.
        
        Args:
            last_mile_data: Last mile delivery data
            
        Returns:
            Dictionary with last mile trends
        """
        last_mile_trends = {}
        
        # Extract trend data
        if "trends" in last_mile_data:
            trend_data = last_mile_data["trends"]
            
            # On-time delivery trend
            if "on_time_delivery" in trend_data:
                on_time_trend = trend_data["on_time_delivery"]
                
                if "values" in on_time_trend and "periods" in on_time_trend:
                    values = on_time_trend["values"]
                    periods = on_time_trend["periods"]
                    
                    # Calculate trend direction and magnitude
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values)
                    
                    last_mile_trends["on_time_delivery"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # First attempt delivery trend
            if "first_attempt_rate" in trend_data:
                first_attempt_trend = trend_data["first_attempt_rate"]
                
                if "values" in first_attempt_trend and "periods" in first_attempt_trend:
                    values = first_attempt_trend["values"]
                    periods = first_attempt_trend["periods"]
                    
                    # Calculate trend direction and magnitude
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values)
                    
                    last_mile_trends["first_attempt_rate"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
            
            # Average delivery time trend
            if "avg_delivery_time" in trend_data:
                time_trend = trend_data["avg_delivery_time"]
                
                if "values" in time_trend and "periods" in time_trend:
                    values = time_trend["values"]
                    periods = time_trend["periods"]
                    
                    # Calculate trend direction and magnitude (note: for time, decreasing is positive)
                    trend_direction, trend_magnitude = self._calculate_trend_direction(values, decreasing_is_positive=True)
                    
                    last_mile_trends["avg_delivery_time"] = {
                        "values": values,
                        "periods": periods,
                        "direction": trend_direction,
                        "magnitude": round(trend_magnitude, 2)
                    }
        
        return last_mile_trends
    
    def _generate_last_mile_insights(
        self,
        last_mile_metrics: Dict[str, Any],
        performance_by_region: Dict[str, Dict[str, Any]],
        last_mile_trends: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from last mile delivery data.
        
        Args:
            last_mile_metrics: Last mile metrics
            performance_by_region: Performance by region
            last_mile_trends: Last mile trends
            
        Returns:
            List of insights
        """
        insights = []
        
        # Generate insights based on overall metrics
        if "on_time_delivery" in last_mile_metrics:
            on_time_value = last_mile_metrics["on_time_delivery"]["value"]
            performance_level = last_mile_metrics["on_time_delivery"]["performance_level"]
            
            if performance_level in ("excellent", "good"):
                insights.append({
                    "type": "positive",
                    "category": "on_time_delivery",
                    "insight": f"Strong last mile on-time delivery at {on_time_value}%"
                })
            elif performance_level in ("below_average", "poor"):
                insights.append({
                    "type": "negative",
                    "category": "on_time_delivery",
                    "insight": f"Poor last mile on-time delivery at {on_time_value}%"
                })
        
        if "first_attempt_rate" in last_mile_metrics:
            first_attempt_rate = last_mile_metrics["first_attempt_rate"]["value"]
            
            if first_attempt_rate >= 90:
                insights.append({
                    "type": "positive",
                    "category": "first_attempt_rate",
                    "insight": f"High first attempt delivery success rate at {first_attempt_rate}%"
                })
            elif first_attempt_rate < 80:
                insights.append({
                    "type": "negative",
                    "category": "first_attempt_rate",
                    "insight": f"Low first attempt delivery success rate at {first_attempt_rate}%"
                })
        
        if "customer_satisfaction" in last_mile_metrics:
            satisfaction = last_mile_metrics["customer_satisfaction"]["value"]
            
            if satisfaction >= 8.5:
                insights.append({
                    "type": "positive",
                    "category": "customer_satisfaction",
                    "insight": f"High customer satisfaction with last mile delivery at {satisfaction}/10"
                })
            elif satisfaction < 7:
                insights.append({
                    "type": "negative",
                    "category": "customer_satisfaction",
                    "insight": f"Low customer satisfaction with last mile delivery at {satisfaction}/10"
                })
        
        # Generate insights based on trends
        if "on_time_delivery" in last_mile_trends:
            direction = last_mile_trends["on_time_delivery"]["direction"]
            magnitude = last_mile_trends["on_time_delivery"]["magnitude"]
            
            if direction == "improving" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "on_time_delivery",
                    "insight": f"Last mile on-time delivery improving by {magnitude:.1f}%"
                })
            elif direction == "declining" and magnitude >= 5:
                insights.append({
                    "type": "trend",
                    "category": "on_time_delivery",
                    "insight": f"Last mile on-time delivery declining by {magnitude:.1f}%"
                })
        
        # Generate insights by region
        if performance_by_region:
            # Find best and worst performing regions
            on_time_by_region = {
                region: metrics["on_time_delivery"]["value"]
                for region, metrics in performance_by_region.items()
                if "on_time_delivery" in metrics
            }
            
            if on_time_by_region:
                best_region = max(on_time_by_region.items(), key=lambda x: x[1])
                worst_region = min(on_time_by_region.items(), key=lambda x: x[1])
                
                region_gap = best_region[1] - worst_region[1]
                
                if region_gap >= 10:
                    insights.append({
                        "type": "region",
                        "category": "regional_variation",
                        "insight": f"Large on-time delivery gap between regions: {best_region[0]} ({best_region[1]}%) vs {worst_region[0]} ({worst_region[1]}%)"
                    })
            
            # Find regions with particularly low first attempt rates
            first_attempt_by_region = {
                region: metrics["first_attempt_rate"]["value"]
                for region, metrics in performance_by_region.items()
                if "first_attempt_rate" in metrics
            }
            
            if first_attempt_by_region:
                low_regions = [
                    (region, rate) for region, rate in first_attempt_by_region.items()
                    if rate < 75
                ]
                
                for region, rate in low_regions[:2]:  # Limit to top 2
                    insights.append({
                        "type": "region",
                        "category": "first_attempt_rate",
                        "insight": f"Low first attempt delivery rate in {region} at {rate}%"
                    })
        
        return insights
    
    def _generate_last_mile_recommendations(
        self,
        last_mile_metrics: Dict[str, Any],
        performance_by_region: Dict[str, Dict[str, Any]],
        last_mile_trends: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on last mile delivery data.
        
        Args:
            last_mile_metrics: Last mile metrics
            performance_by_region: Performance by region
            last_mile_trends: Last mile trends
            insights: Generated insights
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate recommendations based on overall metrics
        if "on_time_delivery" in last_mile_metrics:
            on_time_value = last_mile_metrics["on_time_delivery"]["value"]
            performance_level = last_mile_metrics["on_time_delivery"]["performance_level"]
            
            if performance_level in ("below_average", "poor"):
                recommendations.append({
                    "priority": "high",
                    "category": "on_time_delivery",
                    "recommendation": "Implement real-time tracking and dynamic routing for last mile delivery",
                    "expected_impact": "Potential improvement of 10-15% in on-time delivery rate"
                })
        
        if "first_attempt_rate" in last_mile_metrics:
            first_attempt_rate = last_mile_metrics["first_attempt_rate"]["value"]
            
            if first_attempt_rate < 85:
                recommendations.append({
                    "priority": "high",
                    "category": "first_attempt_rate",
                    "recommendation": "Implement delivery scheduling and notification system for customers",
                    "expected_impact": "Potential increase of 10-20% in first attempt delivery success"
                })
        
        if "avg_delivery_time" in last_mile_metrics:
            avg_delivery_time = last_mile_metrics["avg_delivery_time"]["value"]
            
            if avg_delivery_time > 10:
                recommendations.append({
                    "priority": "medium",
                    "category": "delivery_time",
                    "recommendation": "Optimize package sorting and loading processes to reduce delivery time",
                    "expected_impact": "Potential reduction of 15-25% in average delivery time"
                })
        
        # Generate recommendations based on trends
        for metric, trend_data in last_mile_trends.items():
            direction = trend_data["direction"]
            magnitude = trend_data["magnitude"]
            
            if direction == "declining" and magnitude >= 5:
                if metric == "on_time_delivery":
                    recommendations.append({
                        "priority": "high",
                        "category": "on_time_delivery",
                        "recommendation": "Investigate causes of declining on-time delivery performance",
                        "expected_impact": "Identify and address factors causing the negative trend"
                    })
                elif metric == "first_attempt_rate":
                    recommendations.append({
                        "priority": "high",
                        "category": "first_attempt_rate",
                        "recommendation": "Analyze patterns in failed delivery attempts",
                        "expected_impact": "Identify key factors reducing first attempt success rate"
                    })
        
        # Generate region-specific recommendations
        if performance_by_region:
            # Identify poorly performing regions
            poor_regions = []
            for region, metrics in performance_by_region.items():
                if "on_time_delivery" in metrics:
                    on_time = metrics["on_time_delivery"]["value"]
                    performance_level = metrics["on_time_delivery"]["performance_level"]
                    
                    if performance_level in ("below_average", "poor"):
                        poor_regions.append((region, on_time))
            
            # Sort by on-time delivery rate (worst first)
            poor_regions.sort(key=lambda x: x[1])
            
            # Make recommendations for worst performing regions
            for region, on_time in poor_regions[:2]:  # Limit to top 2
                recommendations.append({
                    "priority": "medium",
                    "category": "regional",
                    "recommendation": f"Develop region-specific improvement plan for {region}",
                    "expected_impact": f"Address unique challenges in {region} to improve performance"
                })
        
        # Add general recommendations
        recommendations.append({
            "priority": "medium",
            "category": "technology",
            "recommendation": "Implement mobile proof of delivery app with geolocation verification",
            "expected_impact": "Improve delivery accuracy and provide better visibility"
        })
        
        recommendations.append({
            "priority": "low",
            "category": "training",
            "recommendation": "Enhance training program for last mile delivery drivers",
            "expected_impact": "Improve delivery quality and customer interaction"
        })
        
        return recommendations
    
    def _generate_mock_last_mile_data(
        self,
        time_period: str = "last_30_days",
        filters: Optional[Dict[str, Any]] = None,
        error: bool = False
    ) -> Dict[str, Any]:
        """
        Generate mock last mile delivery data for testing.
        
        Args:
            time_period: Time period
            filters: Optional filters
            error: Whether to simulate an error
            
        Returns:
            Dictionary with mock last mile data
        """
        if error:
            return {
                "error": "Failed to retrieve last mile data",
                "time_period": time_period,
                "is_mock_data": True
            }
        
        # Generate a deterministic random seed
        seed = hash(time_period) % 10000
        np.random.seed(seed)
        
        # Generate mock summary data
        total_deliveries = np.random.randint(5000, 20000)
        on_time_rate = np.random.uniform(0.8, 0.93)
        on_time_deliveries = int(total_deliveries * on_time_rate)
        
        first_attempt_rate = np.random.uniform(0.82, 0.95)
        first_attempt_success = int(total_deliveries * first_attempt_rate)
        
        avg_delivery_time = np.random.uniform(5, 12)  # minutes
        avg_stops_per_route = np.random.uniform(25, 40)
        customer_satisfaction = np.random.uniform(7.5, 9.0)  # 0-10 scale
        
        # Generate data by region
        regions = [
            "Northeast",
            "Southeast",
            "Midwest",
            "Southwest",
            "West",
            "Northwest"
        ]
        
        region_data = {}
        for region in regions:
            region_deliveries = np.random.randint(total_deliveries // (len(regions) * 2), total_deliveries // len(regions))
            
            region_on_time_rate = on_time_rate * np.random.uniform(0.9, 1.1)
            region_on_time_deliveries = int(region_deliveries * region_on_time_rate)
            
            region_first_attempt_rate = first_attempt_rate * np.random.uniform(0.9, 1.1)
            region_first_attempt_success = int(region_deliveries * region_first_attempt_rate)
            
            region_delivery_time = avg_delivery_time * np.random.uniform(0.85, 1.15)
            
            region_data[region] = {
                "total_deliveries": region_deliveries,
                "on_time_deliveries": region_on_time_deliveries,
                "first_attempt_success": region_first_attempt_success,
                "avg_delivery_time": region_delivery_time
            }
        
        # Generate trend data
        periods = 12
        trend_periods = []
        
        # Create period labels
        if time_period == "last_30_days":
            # Daily periods
            for i in range(periods):
                days_ago = periods - i - 1
                trend_periods.append(f"Day {days_ago + 1}")
        elif time_period == "last_12_months":
            # Monthly periods
            current_month = datetime.now().month
            current_year = datetime.now().year
            
            for i in range(periods):
                month = current_month - i
                year = current_year
                if month <= 0:
                    month += 12
                    year -= 1
                trend_periods.append(f"{year}-{month:02d}")
        else:
            # Generic periods
            for i in range(periods):
                trend_periods.append(f"Period {i+1}")
        
        # Generate trend values with some randomness and trend
        def generate_trend_values(base: float, trend_factor: float, noise: float) -> List[float]:
            return [
                max(0, base * (1 + trend_factor * (i / periods)) + np.random.normal(0, noise * base))
                for i in range(periods)
            ]
        
        on_time_trend = generate_trend_values(on_time_rate * 100, np.random.uniform(-0.05, 0.05), 0.03)
        first_attempt_trend = generate_trend_values(first_attempt_rate * 100, np.random.uniform(-0.05, 0.05), 0.03)
        delivery_time_trend = generate_trend_values(avg_delivery_time, np.random.uniform(-0.1, 0.1), 0.05)
        
        # Compile last mile data
        last_mile_data = {
            "time_period": time_period,
            "summary": {
                "total_deliveries": total_deliveries,
                "on_time_deliveries": on_time_deliveries,
                "first_attempt_success": first_attempt_success,
                "avg_delivery_time": avg_delivery_time,
                "avg_stops_per_route": avg_stops_per_route,
                "customer_satisfaction": customer_satisfaction
            },
            "by_region": region_data,
            "trends": {
                "on_time_delivery": {
                    "values": on_time_trend,
                    "periods": trend_periods
                },
                "first_attempt_rate": {
                    "values": first_attempt_trend,
                    "periods": trend_periods
                },
                "avg_delivery_time": {
                    "values": delivery_time_trend,
                    "periods": trend_periods
                }
            },
            "is_mock_data": True
        }
        
        # Apply filters if provided
        if filters:
            # This is a placeholder for filter logic
            last_mile_data["applied_filters"] = filters
        
        return last_mile_data