"""
Carrier Performance Analytics Module

This module provides functionality for analyzing and scoring
the performance of carriers and transportation providers.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class CarrierPerformanceAnalyzer:
    """
    Analyzes and scores carrier performance based on various metrics.
    """
    
    # Default weights for performance metrics
    DEFAULT_METRIC_WEIGHTS = {
        "on_time_delivery": 0.30,
        "transit_time": 0.15,
        "damage_rate": 0.20,
        "cost_performance": 0.15,
        "documentation_accuracy": 0.10,
        "responsiveness": 0.10
    }
    
    # Performance score ranges
    SCORE_RANGES = {
        "excellent": (90, 100),
        "good": (80, 90),
        "average": (70, 80),
        "below_average": (60, 70),
        "poor": (0, 60)
    }
    
    def __init__(
        self,
        metric_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the carrier performance analyzer.
        
        Args:
            metric_weights: Optional custom weights for performance metrics
        """
        # Use custom weights if provided, otherwise use defaults
        self.metric_weights = metric_weights or self.DEFAULT_METRIC_WEIGHTS
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(self.metric_weights.values())
        if total_weight != 1.0:
            self.metric_weights = {k: v / total_weight for k, v in self.metric_weights.items()}
    
    async def analyze_carrier_performance(
        self,
        carrier_id: str,
        time_period: str = "last_6_months",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        benchmark_comparison: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze performance for a specific carrier.
        
        Args:
            carrier_id: Carrier ID
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            benchmark_comparison: Whether to include benchmark comparison
            
        Returns:
            Dictionary with carrier performance analysis
        """
        try:
            # Get carrier data
            carrier_data = await self._get_carrier_data(
                carrier_id=carrier_id,
                time_period=time_period,
                client_id=client_id,
                connection_id=connection_id
            )
            
            # Get benchmark data if requested
            benchmark_data = None
            if benchmark_comparison:
                benchmark_data = await self._get_benchmark_data(
                    carrier_data=carrier_data,
                    time_period=time_period,
                    client_id=client_id,
                    connection_id=connection_id
                )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                carrier_data=carrier_data,
                benchmark_data=benchmark_data
            )
            
            # Calculate overall performance score
            overall_score = self._calculate_overall_score(performance_metrics)
            
            # Generate performance insights
            insights = self._generate_performance_insights(
                performance_metrics=performance_metrics,
                overall_score=overall_score,
                carrier_data=carrier_data,
                benchmark_data=benchmark_data
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                performance_metrics=performance_metrics,
                overall_score=overall_score,
                carrier_data=carrier_data,
                benchmark_data=benchmark_data
            )
            
            # Prepare result
            result = {
                "carrier_id": carrier_id,
                "carrier_name": carrier_data.get("carrier_name", "Unknown Carrier"),
                "time_period": time_period,
                "overall_score": overall_score,
                "performance_category": self._get_performance_category(overall_score),
                "metrics": performance_metrics,
                "insights": insights,
                "recommendations": recommendations,
                "historical_trend": carrier_data.get("historical_trend"),
                "benchmark_comparison": benchmark_data if benchmark_comparison else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing carrier performance: {str(e)}")
            return {
                "error": str(e),
                "carrier_id": carrier_id,
                "overall_score": 0,
                "metrics": {}
            }
    
    async def analyze_all_carriers(
        self,
        time_period: str = "last_6_months",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze performance for all carriers.
        
        Args:
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            limit: Maximum number of carriers to analyze
            
        Returns:
            Dictionary with analysis for all carriers
        """
        try:
            # Get list of carriers
            carriers = await self._get_carrier_list(
                client_id=client_id,
                connection_id=connection_id,
                limit=limit
            )
            
            # Analyze each carrier
            carrier_analyses = []
            for carrier in carriers:
                analysis = await self.analyze_carrier_performance(
                    carrier_id=carrier["carrier_id"],
                    time_period=time_period,
                    client_id=client_id,
                    connection_id=connection_id,
                    benchmark_comparison=False  # Skip benchmark comparison for individual carriers
                )
                carrier_analyses.append(analysis)
            
            # Calculate benchmark data
            benchmark_data = self._calculate_benchmark_from_analyses(carrier_analyses)
            
            # Calculate rankings
            rankings = self._calculate_carrier_rankings(carrier_analyses)
            
            # Generate overall insights
            overall_insights = self._generate_overall_insights(
                carrier_analyses=carrier_analyses,
                benchmark_data=benchmark_data
            )
            
            # Prepare result
            result = {
                "time_period": time_period,
                "carrier_count": len(carrier_analyses),
                "carriers": carrier_analyses,
                "rankings": rankings,
                "benchmark_data": benchmark_data,
                "overall_insights": overall_insights
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing all carriers: {str(e)}")
            return {
                "error": str(e),
                "carrier_count": 0,
                "carriers": []
            }
    
    async def _get_carrier_data(
        self,
        carrier_id: str,
        time_period: str = "last_6_months",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get carrier data for analysis.
        
        Args:
            carrier_id: Carrier ID
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Dictionary with carrier data
        """
        try:
            # Get data from database
            if client_id and connection_id:
                # Import interfaces
                from app.db.interfaces.carrier_interface import CarrierInterface
                
                # Create interface
                carrier_interface = CarrierInterface(client_id=client_id, connection_id=connection_id)
                
                # Get carrier data
                carrier_data = await carrier_interface.get_carrier_data(
                    carrier_id=carrier_id,
                    time_period=time_period
                )
                
                return carrier_data
            
            # Generate mock data for demonstration or testing
            carrier_data = self._generate_mock_carrier_data(
                carrier_id=carrier_id,
                time_period=time_period
            )
            
            return carrier_data
            
        except Exception as e:
            logger.error(f"Error getting carrier data: {str(e)}")
            
            # Generate mock data if an error occurs
            return self._generate_mock_carrier_data(
                carrier_id=carrier_id,
                time_period=time_period,
                error=True
            )
    
    async def _get_benchmark_data(
        self,
        carrier_data: Dict[str, Any],
        time_period: str = "last_6_months",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get benchmark data for carrier comparison.
        
        Args:
            carrier_data: Carrier data
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Dictionary with benchmark data
        """
        try:
            # Get data from database
            if client_id and connection_id:
                # Import interfaces
                from app.db.interfaces.carrier_interface import CarrierInterface
                
                # Create interface
                carrier_interface = CarrierInterface(client_id=client_id, connection_id=connection_id)
                
                # Get benchmark data
                benchmark_data = await carrier_interface.get_benchmark_data(
                    carrier_type=carrier_data.get("carrier_type", "all"),
                    time_period=time_period
                )
                
                return benchmark_data
            
            # Generate mock benchmark data
            benchmark_data = self._generate_mock_benchmark_data(
                carrier_data=carrier_data,
                time_period=time_period
            )
            
            return benchmark_data
            
        except Exception as e:
            logger.error(f"Error getting benchmark data: {str(e)}")
            
            # Generate mock data if an error occurs
            return self._generate_mock_benchmark_data(
                carrier_data=carrier_data,
                time_period=time_period,
                error=True
            )
    
    async def _get_carrier_list(
        self,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get list of carriers for analysis.
        
        Args:
            client_id: Optional client ID
            connection_id: Optional connection ID
            limit: Maximum number of carriers to return
            
        Returns:
            List of carriers
        """
        try:
            # Get data from database
            if client_id and connection_id:
                # Import interfaces
                from app.db.interfaces.carrier_interface import CarrierInterface
                
                # Create interface
                carrier_interface = CarrierInterface(client_id=client_id, connection_id=connection_id)
                
                # Get carrier list
                carriers = await carrier_interface.get_carrier_list(limit=limit)
                
                return carriers
            
            # Generate mock carrier list
            carriers = []
            for i in range(min(10, limit)):
                carriers.append({
                    "carrier_id": f"CARR-{i+1:04d}",
                    "carrier_name": f"Carrier {i+1}",
                    "carrier_type": "LTL" if i % 3 == 0 else ("FTL" if i % 3 == 1 else "Parcel"),
                    "active": True,
                    "is_mock_data": True
                })
            
            return carriers
            
        except Exception as e:
            logger.error(f"Error getting carrier list: {str(e)}")
            
            # Return minimal mock data if an error occurs
            return [
                {
                    "carrier_id": "CARR-0001",
                    "carrier_name": "Mock Carrier 1",
                    "carrier_type": "LTL",
                    "active": True,
                    "is_mock_data": True
                },
                {
                    "carrier_id": "CARR-0002",
                    "carrier_name": "Mock Carrier 2",
                    "carrier_type": "FTL",
                    "active": True,
                    "is_mock_data": True
                }
            ]
    
    def _calculate_performance_metrics(
        self,
        carrier_data: Dict[str, Any],
        benchmark_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics for a carrier.
        
        Args:
            carrier_data: Carrier data
            benchmark_data: Optional benchmark data
            
        Returns:
            Dictionary with performance metrics
        """
        # Extract metrics from carrier data
        metrics = {}
        
        # On-time delivery rate
        if "deliveries" in carrier_data:
            deliveries = carrier_data["deliveries"]
            
            total_deliveries = deliveries.get("total", 0)
            on_time_deliveries = deliveries.get("on_time", 0)
            
            on_time_rate = (on_time_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
            metrics["on_time_delivery"] = {
                "value": on_time_rate,
                "raw_value": on_time_rate,
                "total_deliveries": total_deliveries,
                "on_time_deliveries": on_time_deliveries,
                "unit": "%"
            }
        
        # Transit time performance
        if "transit_times" in carrier_data:
            transit_times = carrier_data["transit_times"]
            
            actual_avg = transit_times.get("actual_avg", 0)
            expected_avg = transit_times.get("expected_avg", 1)
            
            # Lower is better - ratio of actual to expected (less than 1 means faster than expected)
            transit_ratio = actual_avg / expected_avg if expected_avg > 0 else 1
            
            # Convert to score where 100 = perfect (25% faster than expected or better)
            # and 0 = terrible (100% slower than expected or worse)
            if transit_ratio <= 0.75:
                transit_score = 100
            elif transit_ratio >= 2:
                transit_score = 0
            else:
                transit_score = max(0, 100 - ((transit_ratio - 0.75) / 1.25 * 100))
            
            metrics["transit_time"] = {
                "value": transit_score,
                "raw_value": transit_ratio,
                "actual_avg": actual_avg,
                "expected_avg": expected_avg,
                "unit": "score"
            }
        
        # Damage rate
        if "damages" in carrier_data:
            damages = carrier_data["damages"]
            
            total_shipments = damages.get("total_shipments", 0)
            damaged_shipments = damages.get("damaged_shipments", 0)
            
            damage_rate = (damaged_shipments / total_shipments * 100) if total_shipments > 0 else 0
            
            # Convert to score where 100 = perfect (0% damage) and 0 = terrible (10% damage or worse)
            damage_score = max(0, 100 - (damage_rate * 10))
            
            metrics["damage_rate"] = {
                "value": damage_score,
                "raw_value": damage_rate,
                "total_shipments": total_shipments,
                "damaged_shipments": damaged_shipments,
                "unit": "score"
            }
        
        # Cost performance
        if "costs" in carrier_data:
            costs = carrier_data["costs"]
            
            actual_cost = costs.get("actual_avg", 0)
            benchmark_cost = costs.get("benchmark_avg", 0) or (benchmark_data["costs"]["benchmark_avg"] if benchmark_data and "costs" in benchmark_data else 0)
            
            # Lower is better - ratio of actual to benchmark (less than 1 means cheaper than benchmark)
            cost_ratio = actual_cost / benchmark_cost if benchmark_cost > 0 else 1
            
            # Convert to score where 100 = perfect (25% cheaper than benchmark or better)
            # and 0 = terrible (100% more expensive than benchmark or worse)
            if cost_ratio <= 0.75:
                cost_score = 100
            elif cost_ratio >= 2:
                cost_score = 0
            else:
                cost_score = max(0, 100 - ((cost_ratio - 0.75) / 1.25 * 100))
            
            metrics["cost_performance"] = {
                "value": cost_score,
                "raw_value": cost_ratio,
                "actual_avg": actual_cost,
                "benchmark_avg": benchmark_cost,
                "unit": "score"
            }
        
        # Documentation accuracy
        if "documentation" in carrier_data:
            documentation = carrier_data["documentation"]
            
            total_documents = documentation.get("total", 0)
            accurate_documents = documentation.get("accurate", 0)
            
            accuracy_rate = (accurate_documents / total_documents * 100) if total_documents > 0 else 0
            
            metrics["documentation_accuracy"] = {
                "value": accuracy_rate,
                "raw_value": accuracy_rate,
                "total_documents": total_documents,
                "accurate_documents": accurate_documents,
                "unit": "%"
            }
        
        # Responsiveness
        if "responsiveness" in carrier_data:
            responsiveness = carrier_data["responsiveness"]
            
            avg_response_time = responsiveness.get("avg_response_time", 0)
            target_response_time = responsiveness.get("target_response_time", 1)
            
            # Lower is better - ratio of actual to target (less than 1 means faster than target)
            response_ratio = avg_response_time / target_response_time if target_response_time > 0 else 1
            
            # Convert to score where 100 = perfect (50% faster than target or better)
            # and 0 = terrible (3x slower than target or worse)
            if response_ratio <= 0.5:
                response_score = 100
            elif response_ratio >= 3:
                response_score = 0
            else:
                response_score = max(0, 100 - ((response_ratio - 0.5) / 2.5 * 100))
            
            metrics["responsiveness"] = {
                "value": response_score,
                "raw_value": response_ratio,
                "avg_response_time": avg_response_time,
                "target_response_time": target_response_time,
                "unit": "score"
            }
        
        # Add benchmark comparison if available
        if benchmark_data:
            for metric_name, metric_data in metrics.items():
                if metric_name in benchmark_data:
                    benchmark_value = benchmark_data[metric_name].get("value", 0)
                    difference = metric_data["value"] - benchmark_value
                    percent_difference = (difference / benchmark_value * 100) if benchmark_value > 0 else 0
                    
                    metrics[metric_name]["benchmark"] = {
                        "value": benchmark_value,
                        "difference": difference,
                        "percent_difference": percent_difference
                    }
        
        return metrics
    
    def _calculate_overall_score(
        self,
        performance_metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate overall performance score.
        
        Args:
            performance_metrics: Performance metrics
            
        Returns:
            Overall performance score (0-100)
        """
        overall_score = 0
        total_weight = 0
        
        # Calculate weighted sum of metrics
        for metric_name, weight in self.metric_weights.items():
            if metric_name in performance_metrics:
                metric_value = performance_metrics[metric_name]["value"]
                overall_score += metric_value * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            overall_score /= total_weight
        
        return round(overall_score, 1)
    
    def _get_performance_category(
        self,
        score: float
    ) -> str:
        """
        Get performance category based on score.
        
        Args:
            score: Performance score
            
        Returns:
            Performance category
        """
        for category, (min_score, max_score) in self.SCORE_RANGES.items():
            if min_score <= score < max_score:
                return category
        
        return "unknown"
    
    def _generate_performance_insights(
        self,
        performance_metrics: Dict[str, Any],
        overall_score: float,
        carrier_data: Dict[str, Any],
        benchmark_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """
        Generate insights based on performance metrics.
        
        Args:
            performance_metrics: Performance metrics
            overall_score: Overall performance score
            carrier_data: Carrier data
            benchmark_data: Optional benchmark data
            
        Returns:
            Dictionary with insights by category
        """
        insights = {
            "strengths": [],
            "weaknesses": [],
            "trends": [],
            "opportunities": []
        }
        
        # Identify strengths and weaknesses
        for metric_name, metric_data in performance_metrics.items():
            metric_value = metric_data["value"]
            
            # Strengths are metrics with scores >= 85
            if metric_value >= 85:
                if metric_name == "on_time_delivery":
                    insights["strengths"].append(
                        f"Strong on-time delivery performance at {metric_value:.1f}%"
                    )
                elif metric_name == "transit_time":
                    insights["strengths"].append(
                        f"Excellent transit time performance, with deliveries averaging " +
                        f"{metric_data['raw_value']:.2f}x of expected transit times"
                    )
                elif metric_name == "damage_rate":
                    insights["strengths"].append(
                        f"Very low damage rate of {metric_data['raw_value']:.2f}%"
                    )
                elif metric_name == "cost_performance":
                    insights["strengths"].append(
                        f"Competitive pricing at {metric_data['raw_value']:.2f}x of benchmark costs"
                    )
                elif metric_name == "documentation_accuracy":
                    insights["strengths"].append(
                        f"High documentation accuracy at {metric_value:.1f}%"
                    )
                elif metric_name == "responsiveness":
                    insights["strengths"].append(
                        f"Excellent responsiveness, with response times averaging " +
                        f"{metric_data['raw_value']:.2f}x of target times"
                    )
            
            # Weaknesses are metrics with scores < 70
            if metric_value < 70:
                if metric_name == "on_time_delivery":
                    insights["weaknesses"].append(
                        f"Poor on-time delivery performance at {metric_value:.1f}%"
                    )
                elif metric_name == "transit_time":
                    insights["weaknesses"].append(
                        f"Slow transit times, averaging {metric_data['raw_value']:.2f}x of expected times"
                    )
                elif metric_name == "damage_rate":
                    insights["weaknesses"].append(
                        f"High damage rate of {metric_data['raw_value']:.2f}%"
                    )
                elif metric_name == "cost_performance":
                    insights["weaknesses"].append(
                        f"High costs at {metric_data['raw_value']:.2f}x of benchmark costs"
                    )
                elif metric_name == "documentation_accuracy":
                    insights["weaknesses"].append(
                        f"Low documentation accuracy at {metric_value:.1f}%"
                    )
                elif metric_name == "responsiveness":
                    insights["weaknesses"].append(
                        f"Poor responsiveness, with response times averaging " +
                        f"{metric_data['raw_value']:.2f}x of target times"
                    )
            
            # Benchmark comparison insights
            if benchmark_data and "benchmark" in metric_data:
                benchmark = metric_data["benchmark"]
                percent_diff = benchmark["percent_difference"]
                
                if percent_diff >= 10:
                    insights["strengths"].append(
                        f"{metric_name.replace('_', ' ').title()} is {percent_diff:.1f}% better than benchmark"
                    )
                elif percent_diff <= -10:
                    insights["weaknesses"].append(
                        f"{metric_name.replace('_', ' ').title()} is {-percent_diff:.1f}% worse than benchmark"
                    )
        
        # Generate trend insights
        if "historical_trend" in carrier_data:
            trend_data = carrier_data["historical_trend"]
            
            for metric_name, trend in trend_data.items():
                if trend.get("direction") == "improving" and trend.get("magnitude", 0) >= 5:
                    insights["trends"].append(
                        f"{metric_name.replace('_', ' ').title()} has improved by {trend['magnitude']:.1f}% " +
                        f"over the past {trend.get('period', 'period')}"
                    )
                elif trend.get("direction") == "declining" and trend.get("magnitude", 0) >= 5:
                    insights["trends"].append(
                        f"{metric_name.replace('_', ' ').title()} has declined by {trend['magnitude']:.1f}% " +
                        f"over the past {trend.get('period', 'period')}"
                    )
        
        # Generate opportunity insights
        for metric_name, metric_data in performance_metrics.items():
            metric_value = metric_data["value"]
            
            # Opportunities are metrics between 70 and 85 that could be improved
            if 70 <= metric_value < 85:
                if metric_name == "on_time_delivery":
                    insights["opportunities"].append(
                        f"Improve on-time delivery from {metric_value:.1f}% to meet target of 95%"
                    )
                elif metric_name == "transit_time":
                    insights["opportunities"].append(
                        f"Reduce transit times which are currently {metric_data['raw_value']:.2f}x of expected times"
                    )
                elif metric_name == "damage_rate":
                    insights["opportunities"].append(
                        f"Reduce damage rate from {metric_data['raw_value']:.2f}% toward industry best practice of <1%"
                    )
                elif metric_name == "cost_performance":
                    insights["opportunities"].append(
                        f"Negotiate better rates to improve cost performance currently at {metric_data['raw_value']:.2f}x of benchmark"
                    )
                elif metric_name == "documentation_accuracy":
                    insights["opportunities"].append(
                        f"Improve documentation accuracy from {metric_value:.1f}% to meet target of 98%"
                    )
                elif metric_name == "responsiveness":
                    insights["opportunities"].append(
                        f"Improve responsiveness times currently at {metric_data['raw_value']:.2f}x of target times"
                    )
        
        return insights
    
    def _generate_recommendations(
        self,
        performance_metrics: Dict[str, Any],
        overall_score: float,
        carrier_data: Dict[str, Any],
        benchmark_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on performance metrics.
        
        Args:
            performance_metrics: Performance metrics
            overall_score: Overall performance score
            carrier_data: Carrier data
            benchmark_data: Optional benchmark data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Make recommendations based on performance category
        category = self._get_performance_category(overall_score)
        
        if category in ("excellent", "good"):
            recommendations.append({
                "type": "strategic",
                "priority": "medium",
                "recommendation": "Consider expanding business with this carrier given their strong performance",
                "expected_impact": "May reduce overall transportation costs and improve service levels"
            })
        
        elif category in ("below_average", "poor"):
            recommendations.append({
                "type": "strategic",
                "priority": "high",
                "recommendation": "Review relationship with this carrier and consider alternatives",
                "expected_impact": "Potentially significant improvement in service levels and cost performance"
            })
        
        # Make metric-specific recommendations
        for metric_name, metric_data in performance_metrics.items():
            metric_value = metric_data["value"]
            
            if metric_name == "on_time_delivery" and metric_value < 80:
                recommendations.append({
                    "type": "operational",
                    "priority": "high",
                    "recommendation": "Implement regular performance review meetings focused on on-time delivery",
                    "expected_impact": "Improved delivery performance and better visibility into issues"
                })
            
            if metric_name == "damage_rate" and metric_data["raw_value"] > 2:
                recommendations.append({
                    "type": "operational",
                    "priority": "high",
                    "recommendation": "Review packaging and handling procedures with carrier",
                    "expected_impact": "Reduction in damage rates and associated costs"
                })
            
            if metric_name == "cost_performance" and metric_data["raw_value"] > 1.1:
                recommendations.append({
                    "type": "financial",
                    "priority": "medium",
                    "recommendation": "Renegotiate contract terms or seek competitive bids",
                    "expected_impact": "Potential cost savings of 5-15%"
                })
            
            if metric_name == "documentation_accuracy" and metric_value < 90:
                recommendations.append({
                    "type": "operational",
                    "priority": "medium",
                    "recommendation": "Provide clear documentation templates and expectations",
                    "expected_impact": "Improved documentation accuracy and reduced processing time"
                })
            
            if metric_name == "responsiveness" and metric_data["raw_value"] > 1.5:
                recommendations.append({
                    "type": "operational",
                    "priority": "medium",
                    "recommendation": "Establish clear communication protocols and escalation procedures",
                    "expected_impact": "Faster issue resolution and improved communication"
                })
        
        # Add trend-based recommendations
        if "historical_trend" in carrier_data:
            trend_data = carrier_data["historical_trend"]
            
            for metric_name, trend in trend_data.items():
                if trend.get("direction") == "declining" and trend.get("magnitude", 0) >= 10:
                    recommendations.append({
                        "type": "strategic",
                        "priority": "high",
                        "recommendation": f"Investigate root causes of declining {metric_name.replace('_', ' ')} performance",
                        "expected_impact": "Address issues before they further impact operations"
                    })
        
        return recommendations
    
    def _calculate_benchmark_from_analyses(
        self,
        carrier_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate benchmark data from multiple carrier analyses.
        
        Args:
            carrier_analyses: List of carrier analyses
            
        Returns:
            Dictionary with benchmark data
        """
        benchmark_data = {}
        
        # Collect metric values across all carriers
        metric_values = {}
        for analysis in carrier_analyses:
            metrics = analysis.get("metrics", {})
            
            for metric_name, metric_data in metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                
                metric_values[metric_name].append(metric_data["value"])
        
        # Calculate benchmark statistics for each metric
        for metric_name, values in metric_values.items():
            if values:
                values_array = np.array(values)
                
                benchmark_data[metric_name] = {
                    "value": float(np.median(values_array)),
                    "average": float(np.mean(values_array)),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                    "p25": float(np.percentile(values_array, 25)),
                    "p75": float(np.percentile(values_array, 75)),
                    "count": len(values),
                    "unit": "%"
                }
        
        return benchmark_data
    
    def _calculate_carrier_rankings(
        self,
        carrier_analyses: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate carrier rankings based on performance metrics.
        
        Args:
            carrier_analyses: List of carrier analyses
            
        Returns:
            Dictionary with carrier rankings by metric
        """
        rankings = {
            "overall": [],
            "by_metric": {}
        }
        
        # Sort carriers by overall score for overall ranking
        overall_ranking = sorted(
            [
                {
                    "carrier_id": analysis["carrier_id"],
                    "carrier_name": analysis["carrier_name"],
                    "score": analysis["overall_score"]
                }
                for analysis in carrier_analyses
            ],
            key=lambda x: x["score"],
            reverse=True
        )
        
        rankings["overall"] = overall_ranking
        
        # Generate rankings by metric
        metrics = set()
        for analysis in carrier_analyses:
            metrics.update(analysis.get("metrics", {}).keys())
        
        for metric_name in metrics:
            metric_ranking = []
            
            for analysis in carrier_analyses:
                metrics_data = analysis.get("metrics", {})
                
                if metric_name in metrics_data:
                    metric_ranking.append({
                        "carrier_id": analysis["carrier_id"],
                        "carrier_name": analysis["carrier_name"],
                        "score": metrics_data[metric_name]["value"]
                    })
            
            # Sort by score
            metric_ranking.sort(key=lambda x: x["score"], reverse=True)
            
            rankings["by_metric"][metric_name] = metric_ranking
        
        return rankings
    
    def _generate_overall_insights(
        self,
        carrier_analyses: List[Dict[str, Any]],
        benchmark_data: Dict[str, Any]
    ) -> List[str]:
        """
        Generate insights for overall carrier performance.
        
        Args:
            carrier_analyses: List of carrier analyses
            benchmark_data: Benchmark data
            
        Returns:
            List of insights
        """
        insights = []
        
        # Calculate performance distribution
        performance_categories = {category: 0 for category in self.SCORE_RANGES.keys()}
        for analysis in carrier_analyses:
            category = analysis.get("performance_category", "unknown")
            performance_categories[category] = performance_categories.get(category, 0) + 1
        
        # Calculate percentage in each category
        total_carriers = len(carrier_analyses)
        category_percentages = {
            category: count / total_carriers * 100 if total_carriers > 0 else 0
            for category, count in performance_categories.items()
        }
        
        # Generate insights based on distribution
        if category_percentages["excellent"] + category_percentages["good"] >= 70:
            insights.append(
                f"Strong carrier performance overall with {category_percentages['excellent'] + category_percentages['good']:.1f}% " +
                f"of carriers rated good or excellent"
            )
        elif category_percentages["below_average"] + category_percentages["poor"] >= 50:
            insights.append(
                f"Concerning carrier performance with {category_percentages['below_average'] + category_percentages['poor']:.1f}% " +
                f"of carriers rated below average or poor"
            )
        
        # Identify best and worst performing metrics
        metric_averages = {}
        for metric_name, metric_data in benchmark_data.items():
            metric_averages[metric_name] = metric_data["average"]
        
        if metric_averages:
            best_metric = max(metric_averages.items(), key=lambda x: x[1])
            worst_metric = min(metric_averages.items(), key=lambda x: x[1])
            
            insights.append(
                f"Strongest overall performance in {best_metric[0].replace('_', ' ')} " +
                f"with an average score of {best_metric[1]:.1f}"
            )
            
            insights.append(
                f"Weakest overall performance in {worst_metric[0].replace('_', ' ')} " +
                f"with an average score of {worst_metric[1]:.1f}"
            )
        
        # Calculate performance spread
        if total_carriers >= 3:
            overall_scores = [analysis["overall_score"] for analysis in carrier_analyses]
            score_range = max(overall_scores) - min(overall_scores)
            
            if score_range >= 30:
                insights.append(
                    f"Wide performance variation among carriers with a {score_range:.1f} point " +
                    f"difference between highest and lowest performing carriers"
                )
            elif score_range <= 10:
                insights.append(
                    f"Consistent performance across carriers with only a {score_range:.1f} point " +
                    f"difference between highest and lowest performing carriers"
                )
        
        return insights
    
    def _generate_mock_carrier_data(
        self,
        carrier_id: str,
        time_period: str = "last_6_months",
        error: bool = False
    ) -> Dict[str, Any]:
        """
        Generate mock carrier data for testing.
        
        Args:
            carrier_id: Carrier ID
            time_period: Time period
            error: Whether to simulate an error
            
        Returns:
            Dictionary with mock carrier data
        """
        if error:
            return {
                "carrier_id": carrier_id,
                "carrier_name": f"Carrier {carrier_id}",
                "error": "Failed to retrieve carrier data",
                "is_mock_data": True
            }
        
        # Generate a deterministic random seed based on carrier_id
        seed = sum(ord(c) for c in carrier_id)
        np.random.seed(seed)
        
        # Generate mock deliveries data
        total_deliveries = np.random.randint(100, 1000)
        on_time_rate = np.random.uniform(0.75, 0.98)
        on_time_deliveries = int(total_deliveries * on_time_rate)
        
        # Generate mock transit times data
        expected_transit = np.random.uniform(2, 5)  # days
        transit_ratio = np.random.uniform(0.8, 1.3)  # ratio of actual to expected
        actual_transit = expected_transit * transit_ratio
        
        # Generate mock damages data
        total_shipments = total_deliveries
        damage_rate = np.random.uniform(0.005, 0.04)  # 0.5% to 4%
        damaged_shipments = int(total_shipments * damage_rate)
        
        # Generate mock costs data
        benchmark_cost = np.random.uniform(20, 50)  # $ per unit
        cost_ratio = np.random.uniform(0.85, 1.2)  # ratio of actual to benchmark
        actual_cost = benchmark_cost * cost_ratio
        
        # Generate mock documentation data
        total_documents = total_deliveries * 3  # multiple documents per delivery
        accuracy_rate = np.random.uniform(0.85, 0.99)
        accurate_documents = int(total_documents * accuracy_rate)
        
        # Generate mock responsiveness data
        target_response = np.random.uniform(2, 8)  # hours
        response_ratio = np.random.uniform(0.7, 1.5)  # ratio of actual to target
        actual_response = target_response * response_ratio
        
        # Generate mock historical trend data
        historical_trend = {}
        for metric in ["on_time_delivery", "transit_time", "damage_rate", "cost_performance", 
                      "documentation_accuracy", "responsiveness"]:
            # Randomly select improving or declining trend
            direction = np.random.choice(["improving", "declining", "stable"])
            magnitude = np.random.uniform(2, 15) if direction != "stable" else 0
            
            historical_trend[metric] = {
                "direction": direction,
                "magnitude": magnitude,
                "period": time_period
            }
        
        # Compile carrier data
        carrier_data = {
            "carrier_id": carrier_id,
            "carrier_name": f"Carrier {carrier_id}",
            "carrier_type": np.random.choice(["LTL", "FTL", "Parcel"]),
            "time_period": time_period,
            "deliveries": {
                "total": total_deliveries,
                "on_time": on_time_deliveries
            },
            "transit_times": {
                "expected_avg": expected_transit,
                "actual_avg": actual_transit
            },
            "damages": {
                "total_shipments": total_shipments,
                "damaged_shipments": damaged_shipments
            },
            "costs": {
                "benchmark_avg": benchmark_cost,
                "actual_avg": actual_cost
            },
            "documentation": {
                "total": total_documents,
                "accurate": accurate_documents
            },
            "responsiveness": {
                "target_response_time": target_response,
                "avg_response_time": actual_response
            },
            "historical_trend": historical_trend,
            "is_mock_data": True
        }
        
        return carrier_data
    
    def _generate_mock_benchmark_data(
        self,
        carrier_data: Dict[str, Any],
        time_period: str = "last_6_months",
        error: bool = False
    ) -> Dict[str, Any]:
        """
        Generate mock benchmark data for testing.
        
        Args:
            carrier_data: Carrier data
            time_period: Time period
            error: Whether to simulate an error
            
        Returns:
            Dictionary with mock benchmark data
        """
        if error:
            return {
                "error": "Failed to retrieve benchmark data",
                "is_mock_data": True
            }
        
        # Use a fixed seed for benchmarks
        np.random.seed(42)
        
        benchmark_data = {
            "time_period": time_period,
            "carrier_type": carrier_data.get("carrier_type", "all"),
            "is_mock_data": True
        }
        
        # Generate benchmark for on-time delivery
        if "deliveries" in carrier_data:
            benchmark_data["on_time_delivery"] = {
                "value": np.random.uniform(85, 92),
                "unit": "%"
            }
        
        # Generate benchmark for transit time
        if "transit_times" in carrier_data:
            benchmark_data["transit_time"] = {
                "value": np.random.uniform(75, 85),
                "raw_value": np.random.uniform(0.9, 1.1),
                "unit": "score"
            }
        
        # Generate benchmark for damage rate
        if "damages" in carrier_data:
            benchmark_data["damage_rate"] = {
                "value": np.random.uniform(80, 90),
                "raw_value": np.random.uniform(0.01, 0.025),
                "unit": "score"
            }
        
        # Generate benchmark for cost performance
        if "costs" in carrier_data:
            benchmark_data["cost_performance"] = {
                "value": 85,  # By definition, benchmark is at 85 points
                "raw_value": 1.0,  # By definition, benchmark ratio is 1.0
                "unit": "score"
            }
        
        # Generate benchmark for documentation accuracy
        if "documentation" in carrier_data:
            benchmark_data["documentation_accuracy"] = {
                "value": np.random.uniform(88, 95),
                "unit": "%"
            }
        
        # Generate benchmark for responsiveness
        if "responsiveness" in carrier_data:
            benchmark_data["responsiveness"] = {
                "value": np.random.uniform(78, 88),
                "raw_value": np.random.uniform(0.9, 1.1),
                "unit": "score"
            }
        
        # Add standard deviations
        for metric in benchmark_data:
            if metric not in ["time_period", "carrier_type", "is_mock_data"] and isinstance(benchmark_data[metric], dict):
                benchmark_data[metric]["std_dev"] = np.random.uniform(5, 15)
        
        return benchmark_data