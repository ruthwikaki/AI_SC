"""
Supplier Scorecard Module

This module provides functionality for generating and analyzing supplier scorecards,
with performance metrics, trends, and improvement recommendations.
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

class SupplierScorecard:
    """
    Generates and analyzes supplier scorecards with performance metrics.
    """
    
    # Default metric weights
    DEFAULT_METRIC_WEIGHTS = {
        "on_time_delivery": 0.25,
        "quality": 0.25,
        "price_competitiveness": 0.20,
        "responsiveness": 0.15,
        "innovation": 0.10,
        "sustainability": 0.05
    }
    
    # Performance rating thresholds
    PERFORMANCE_RATINGS = {
        "excellent": 90,
        "good": 80,
        "satisfactory": 70,
        "needs_improvement": 60,
        "poor": 0
    }
    
    def __init__(
        self,
        metric_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the supplier scorecard.
        
        Args:
            metric_weights: Optional custom weights for performance metrics
        """
        # Use custom weights if provided, otherwise use defaults
        self.metric_weights = metric_weights or self.DEFAULT_METRIC_WEIGHTS
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(self.metric_weights.values())
        if total_weight != 1.0:
            self.metric_weights = {k: v / total_weight for k, v in self.metric_weights.items()}
    
    async def generate_scorecard(
        self,
        supplier_id: str,
        time_period: str = "last_quarter",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        include_trends: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a scorecard for a supplier.
        
        Args:
            supplier_id: Supplier ID
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            include_trends: Whether to include performance trends
            include_recommendations: Whether to include recommendations
            
        Returns:
            Dictionary with supplier scorecard
        """
        try:
            # Get supplier data
            supplier_data = await self._get_supplier_data(
                supplier_id=supplier_id,
                time_period=time_period,
                client_id=client_id,
                connection_id=connection_id
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(supplier_data)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(performance_metrics)
            
            # Determine performance rating
            performance_rating = self._get_performance_rating(overall_score)
            
            # Get performance trends if requested
            trends = None
            if include_trends:
                trends = self._calculate_performance_trends(supplier_data)
            
            # Generate recommendations if requested
            recommendations = None
            if include_recommendations:
                recommendations = self._generate_recommendations(
                    performance_metrics=performance_metrics,
                    overall_score=overall_score,
                    performance_rating=performance_rating,
                    trends=trends
                )
            
            # Compile scorecard
            scorecard = {
                "supplier_id": supplier_id,
                "supplier_name": supplier_data.get("supplier_name", "Unknown Supplier"),
                "time_period": time_period,
                "generation_date": datetime.now().isoformat(),
                "overall_score": overall_score,
                "performance_rating": performance_rating,
                "performance_metrics": performance_metrics,
                "trends": trends,
                "recommendations": recommendations
            }
            
            return scorecard
            
        except Exception as e:
            logger.error(f"Error generating supplier scorecard: {str(e)}")
            return {
                "error": str(e),
                "supplier_id": supplier_id,
                "time_period": time_period,
                "overall_score": 0,
                "performance_rating": "unknown"
            }
    
    async def generate_comparative_analysis(
        self,
        supplier_ids: List[str],
        time_period: str = "last_quarter",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comparative analysis of multiple suppliers.
        
        Args:
            supplier_ids: List of supplier IDs
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            category: Optional supplier category for benchmarking
            
        Returns:
            Dictionary with comparative analysis
        """
        try:
            # Generate scorecard for each supplier
            scorecards = []
            for supplier_id in supplier_ids:
                scorecard = await self.generate_scorecard(
                    supplier_id=supplier_id,
                    time_period=time_period,
                    client_id=client_id,
                    connection_id=connection_id,
                    include_trends=False,
                    include_recommendations=False
                )
                scorecards.append(scorecard)
            
            # Calculate category benchmarks if requested
            category_benchmarks = None
            if category:
                category_benchmarks = await self._get_category_benchmarks(
                    category=category,
                    time_period=time_period,
                    client_id=client_id,
                    connection_id=connection_id
                )
            
            # Calculate comparative metrics
            comparative_metrics = self._calculate_comparative_metrics(scorecards)
            
            # Generate insights
            insights = self._generate_comparative_insights(
                scorecards=scorecards,
                comparative_metrics=comparative_metrics,
                category_benchmarks=category_benchmarks
            )
            
            # Compile comparative analysis
            analysis = {
                "time_period": time_period,
                "supplier_count": len(scorecards),
                "supplier_category": category,
                "generation_date": datetime.now().isoformat(),
                "scorecards": scorecards,
                "comparative_metrics": comparative_metrics,
                "category_benchmarks": category_benchmarks,
                "insights": insights
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating comparative analysis: {str(e)}")
            return {
                "error": str(e),
                "time_period": time_period,
                "supplier_count": len(supplier_ids),
                "supplier_category": category
            }
    
    async def _get_supplier_data(
        self,
        supplier_id: str,
        time_period: str = "last_quarter",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get supplier performance data.
        
        Args:
            supplier_id: Supplier ID
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Dictionary with supplier data
        """
        try:
            # Get data from database if client_id is provided
            if client_id:
                # Import supplier interface
                from app.db.interfaces.supplier_interface import SupplierInterface
                
                # Create interface
                supplier_interface = SupplierInterface(client_id=client_id, connection_id=connection_id)
                
                # Get category benchmark data
                benchmark_data = await supplier_interface.get_category_benchmarks(
                    category=category,
                    time_period=time_period
                )
                
                return benchmark_data
            
            # Generate mock benchmark data for demonstration
            benchmark_data = self._generate_mock_category_benchmarks(
                category=category,
                time_period=time_period
            )
            
            return benchmark_data
            
        except Exception as e:
            logger.error(f"Error getting category benchmarks: {str(e)}")
            
            # Generate mock data on error
            return self._generate_mock_category_benchmarks(
                category=category,
                time_period=time_period,
                error=True
            )
    
    def _calculate_performance_metrics(
        self,
        supplier_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance metrics from supplier data.
        
        Args:
            supplier_data: Supplier data
            
        Returns:
            Dictionary with calculated performance metrics
        """
        performance_metrics = {}
        
        # Extract metrics from supplier data
        metrics_data = supplier_data.get("metrics", {})
        
        # On-time delivery metric
        if "on_time_delivery" in metrics_data:
            otd_data = metrics_data["on_time_delivery"]
            
            performance_metrics["on_time_delivery"] = {
                "score": otd_data.get("score", 0),
                "raw_value": otd_data.get("value", 0),
                "target": otd_data.get("target", 95),
                "unit": otd_data.get("unit", "%"),
                "weight": self.metric_weights.get("on_time_delivery", 0.25)
            }
        
        # Quality metric
        if "quality" in metrics_data:
            quality_data = metrics_data["quality"]
            
            performance_metrics["quality"] = {
                "score": quality_data.get("score", 0),
                "raw_value": quality_data.get("value", 0),
                "target": quality_data.get("target", 99),
                "unit": quality_data.get("unit", "%"),
                "weight": self.metric_weights.get("quality", 0.25)
            }
        
        # Price competitiveness metric
        if "price_competitiveness" in metrics_data:
            price_data = metrics_data["price_competitiveness"]
            
            performance_metrics["price_competitiveness"] = {
                "score": price_data.get("score", 0),
                "raw_value": price_data.get("value", 0),
                "target": price_data.get("target", 100),
                "unit": price_data.get("unit", "index"),
                "weight": self.metric_weights.get("price_competitiveness", 0.20)
            }
        
        # Responsiveness metric
        if "responsiveness" in metrics_data:
            resp_data = metrics_data["responsiveness"]
            
            performance_metrics["responsiveness"] = {
                "score": resp_data.get("score", 0),
                "raw_value": resp_data.get("value", 0),
                "target": resp_data.get("target", 24),
                "unit": resp_data.get("unit", "hours"),
                "weight": self.metric_weights.get("responsiveness", 0.15)
            }
        
        # Innovation metric
        if "innovation" in metrics_data:
            innov_data = metrics_data["innovation"]
            
            performance_metrics["innovation"] = {
                "score": innov_data.get("score", 0),
                "raw_value": innov_data.get("value", 0),
                "target": innov_data.get("target", 5),
                "unit": innov_data.get("unit", "score"),
                "weight": self.metric_weights.get("innovation", 0.10)
            }
        
        # Sustainability metric
        if "sustainability" in metrics_data:
            sustain_data = metrics_data["sustainability"]
            
            performance_metrics["sustainability"] = {
                "score": sustain_data.get("score", 0),
                "raw_value": sustain_data.get("value", 0),
                "target": sustain_data.get("target", 80),
                "unit": sustain_data.get("unit", "score"),
                "weight": self.metric_weights.get("sustainability", 0.05)
            }
        
        return performance_metrics
    
    def _calculate_overall_score(
        self,
        performance_metrics: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Calculate overall score from performance metrics.
        
        Args:
            performance_metrics: Performance metrics
            
        Returns:
            Overall score (0-100)
        """
        if not performance_metrics:
            return 0
        
        overall_score = 0
        total_weight = 0
        
        # Sum weighted scores
        for metric_name, metric_data in performance_metrics.items():
            score = metric_data.get("score", 0)
            weight = metric_data.get("weight", 0)
            
            overall_score += score * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            overall_score /= total_weight
        
        return round(overall_score, 1)
    
    def _get_performance_rating(
        self,
        overall_score: float
    ) -> str:
        """
        Get performance rating based on overall score.
        
        Args:
            overall_score: Overall score
            
        Returns:
            Performance rating
        """
        for rating, threshold in self.PERFORMANCE_RATINGS.items():
            if overall_score >= threshold:
                return rating
        
        return "unknown"
    
    def _calculate_performance_trends(
        self,
        supplier_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance trends from supplier data.
        
        Args:
            supplier_data: Supplier data
            
        Returns:
            Dictionary with performance trends
        """
        trends = {}
        
        # Extract trend data
        trend_data = supplier_data.get("trends", {})
        
        # Process each metric trend
        for metric_name, metric_trend in trend_data.items():
            # Extract trend values and periods
            values = metric_trend.get("values", [])
            periods = metric_trend.get("periods", [])
            
            if not values or not periods or len(values) != len(periods):
                continue
            
            # Calculate trend direction and magnitude
            # For some metrics, lower is better
            decreasing_is_positive = metric_name in ["responsiveness"]
            direction, magnitude = self._calculate_trend_direction(values, decreasing_is_positive)
            
            trends[metric_name] = {
                "values": values,
                "periods": periods,
                "direction": direction,
                "magnitude": round(magnitude, 2)
            }
        
        # Calculate overall score trend if available
        if "overall_score" in trend_data:
            overall_trend = trend_data["overall_score"]
            values = overall_trend.get("values", [])
            periods = overall_trend.get("periods", [])
            
            if values and periods and len(values) == len(periods):
                direction, magnitude = self._calculate_trend_direction(values)
                
                trends["overall_score"] = {
                    "values": values,
                    "periods": periods,
                    "direction": direction,
                    "magnitude": round(magnitude, 2)
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
            values: List of values
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
    
    def _generate_recommendations(
        self,
        performance_metrics: Dict[str, Dict[str, Any]],
        overall_score: float,
        performance_rating: str,
        trends: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on performance metrics.
        
        Args:
            performance_metrics: Performance metrics
            overall_score: Overall score
            performance_rating: Performance rating
            trends: Optional performance trends
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate recommendations based on performance rating
        if performance_rating in ["poor", "needs_improvement"]:
            recommendations.append({
                "priority": "high",
                "category": "general",
                "recommendation": "Schedule quarterly performance review meetings with supplier",
                "expected_impact": "Improved communication and focus on key issues"
            })
            
            recommendations.append({
                "priority": "high",
                "category": "general",
                "recommendation": "Develop formal performance improvement plan",
                "expected_impact": "Clear roadmap for addressing critical issues"
            })
        
        # Generate metric-specific recommendations
        for metric_name, metric_data in performance_metrics.items():
            score = metric_data.get("score", 0)
            target = metric_data.get("target", 0)
            raw_value = metric_data.get("raw_value", 0)
            
            # On-time delivery recommendations
            if metric_name == "on_time_delivery" and score < 70:
                recommendations.append({
                    "priority": "high",
                    "category": "on_time_delivery",
                    "recommendation": "Implement weekly order status reporting requirement",
                    "expected_impact": "Early identification of potential delays"
                })
                
                recommendations.append({
                    "priority": "medium",
                    "category": "on_time_delivery",
                    "recommendation": "Review supplier's production capacity and planning process",
                    "expected_impact": "Address root causes of delivery issues"
                })
            
            # Quality recommendations
            if metric_name == "quality" and score < 70:
                recommendations.append({
                    "priority": "high",
                    "category": "quality",
                    "recommendation": "Implement enhanced incoming inspection protocol",
                    "expected_impact": "Prevent quality issues from reaching production"
                })
                
                recommendations.append({
                    "priority": "medium",
                    "category": "quality",
                    "recommendation": "Conduct joint quality improvement workshop",
                    "expected_impact": "Collaborate on addressing root causes of quality issues"
                })
            
            # Price competitiveness recommendations
            if metric_name == "price_competitiveness" and score < 70:
                recommendations.append({
                    "priority": "medium",
                    "category": "price_competitiveness",
                    "recommendation": "Initiate cost breakdown analysis with supplier",
                    "expected_impact": "Identify cost reduction opportunities"
                })
                
                recommendations.append({
                    "priority": "medium",
                    "category": "price_competitiveness",
                    "recommendation": "Evaluate alternative sourcing options",
                    "expected_impact": "Create competitive pressure and benchmark pricing"
                })
            
            # Responsiveness recommendations
            if metric_name == "responsiveness" and score < 70:
                recommendations.append({
                    "priority": "medium",
                    "category": "responsiveness",
                    "recommendation": "Establish clear communication protocols and escalation paths",
                    "expected_impact": "Faster resolution of issues and inquiries"
                })
            
            # Innovation recommendations
            if metric_name == "innovation" and score < 70:
                recommendations.append({
                    "priority": "low",
                    "category": "innovation",
                    "recommendation": "Schedule quarterly innovation sharing sessions",
                    "expected_impact": "Greater exposure to supplier's innovation capabilities"
                })
            
            # Sustainability recommendations
            if metric_name == "sustainability" and score < 70:
                recommendations.append({
                    "priority": "low",
                    "category": "sustainability",
                    "recommendation": "Request supplier sustainability roadmap",
                    "expected_impact": "Clear understanding of supplier's sustainability initiatives"
                })
        
        # Generate recommendations based on trends
        if trends:
            for metric_name, trend_data in trends.items():
                direction = trend_data.get("direction", "stable")
                magnitude = trend_data.get("magnitude", 0)
                
                if direction == "declining" and magnitude >= 5:
                    if metric_name == "overall_score":
                        recommendations.append({
                            "priority": "high",
                            "category": "general",
                            "recommendation": "Conduct comprehensive supplier performance review",
                            "expected_impact": "Identify and address causes of declining performance"
                        })
                    elif metric_name == "on_time_delivery":
                        recommendations.append({
                            "priority": "high",
                            "category": "on_time_delivery",
                            "recommendation": "Investigate causes of declining delivery performance",
                            "expected_impact": "Reverse negative trend in on-time delivery"
                        })
                    elif metric_name == "quality":
                        recommendations.append({
                            "priority": "high",
                            "category": "quality",
                            "recommendation": "Analyze root causes of increasing quality issues",
                            "expected_impact": "Reverse negative trend in quality performance"
                        })
        
        # For high-performing suppliers
        if performance_rating in ["excellent", "good"]:
            recommendations.append({
                "priority": "medium",
                "category": "general",
                "recommendation": "Explore strategic partnership opportunities",
                "expected_impact": "Leverage strong performance for mutual benefit"
            })
            
            recommendations.append({
                "priority": "low",
                "category": "general",
                "recommendation": "Consider for preferred supplier program",
                "expected_impact": "Recognize and incentivize continued strong performance"
            })
        
        return recommendations
    
    def _calculate_comparative_metrics(
        self,
        scorecards: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate comparative metrics from multiple scorecards.
        
        Args:
            scorecards: List of supplier scorecards
            
        Returns:
            Dictionary with comparative metrics
        """
        if not scorecards:
            return {}
        
        comparative_metrics = {
            "overall_scores": {},
            "metric_averages": {},
            "performance_distribution": {}
        }
        
        # Extract overall scores
        overall_scores = {}
        for scorecard in scorecards:
            supplier_id = scorecard.get("supplier_id", "unknown")
            supplier_name = scorecard.get("supplier_name", f"Supplier {supplier_id}")
            overall_score = scorecard.get("overall_score", 0)
            
            overall_scores[supplier_name] = overall_score
        
        # Sort suppliers by overall score
        sorted_suppliers = sorted(
            overall_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        comparative_metrics["overall_scores"] = {
            name: score for name, score in sorted_suppliers
        }
        
        # Calculate metric averages
        metric_values = {}
        
        for scorecard in scorecards:
            metrics = scorecard.get("performance_metrics", {})
            
            for metric_name, metric_data in metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                
                metric_values[metric_name].append(metric_data.get("score", 0))
        
        # Calculate averages
        for metric_name, values in metric_values.items():
            if values:
                comparative_metrics["metric_averages"][metric_name] = round(sum(values) / len(values), 1)
        
        # Calculate performance distribution
        ratings = [
            scorecard.get("performance_rating", "unknown")
            for scorecard in scorecards
        ]
        
        rating_counts = {}
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        total_suppliers = len(scorecards)
        for rating, count in rating_counts.items():
            rating_counts[rating] = {
                "count": count,
                "percentage": round(count / total_suppliers * 100, 1) if total_suppliers > 0 else 0
            }
        
        comparative_metrics["performance_distribution"] = rating_counts
        
        return comparative_metrics
    
    def _generate_comparative_insights(
        self,
        scorecards: List[Dict[str, Any]],
        comparative_metrics: Dict[str, Any],
        category_benchmarks: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from comparative analysis.
        
        Args:
            scorecards: List of supplier scorecards
            comparative_metrics: Comparative metrics
            category_benchmarks: Optional category benchmarks
            
        Returns:
            List of insights
        """
        insights = []
        
        # Generate insights on performance distribution
        distribution = comparative_metrics.get("performance_distribution", {})
        total_suppliers = len(scorecards)
        
        # Calculate percentage of suppliers in each performance category
        excellent_good_pct = 0
        poor_needs_improvement_pct = 0
        
        for rating, data in distribution.items():
            if rating in ["excellent", "good"]:
                excellent_good_pct += data.get("percentage", 0)
            elif rating in ["poor", "needs_improvement"]:
                poor_needs_improvement_pct += data.get("percentage", 0)
        
        # Add insights based on distribution
        if excellent_good_pct >= 70:
            insights.append({
                "type": "positive",
                "category": "performance_distribution",
                "insight": f"{excellent_good_pct}% of suppliers are performing at 'good' or 'excellent' levels"
            })
        elif poor_needs_improvement_pct >= 30:
            insights.append({
                "type": "negative",
                "category": "performance_distribution",
                "insight": f"{poor_needs_improvement_pct}% of suppliers need performance improvement"
            })
        
        # Generate insights on metric averages
        metric_averages = comparative_metrics.get("metric_averages", {})
        
        # Identify strongest and weakest metrics
        if metric_averages:
            sorted_metrics = sorted(
                metric_averages.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            strongest_metric = sorted_metrics[0]
            weakest_metric = sorted_metrics[-1]
            
            insights.append({
                "type": "metric",
                "category": "strongest_metric",
                "insight": f"Highest average performance in '{strongest_metric[0]}' at {strongest_metric[1]} points"
            })
            
            insights.append({
                "type": "metric",
                "category": "weakest_metric",
                "insight": f"Lowest average performance in '{weakest_metric[0]}' at {weakest_metric[1]} points"
            })
        
        # Generate insights based on benchmark comparison
        if category_benchmarks:
            benchmark_metrics = category_benchmarks.get("metric_averages", {})
            
            for metric_name, avg_score in metric_averages.items():
                if metric_name in benchmark_metrics:
                    benchmark_score = benchmark_metrics[metric_name]
                    difference = avg_score - benchmark_score
                    
                    if abs(difference) >=
                    5:
                        if difference > 0:
                            insights.append({
                                "type": "benchmark",
                                "category": metric_name,
                                "insight": f"'{metric_name}' performance is {difference:.1f} points above category benchmark"
                            })
                        else:
                            insights.append({
                                "type": "benchmark",
                                "category": metric_name,
                                "insight": f"'{metric_name}' performance is {-difference:.1f} points below category benchmark"
                            })
        
        # Generate insights on performance spread
        overall_scores = comparative_metrics.get("overall_scores", {})
        
        if len(overall_scores) >= 2:
            scores = list(overall_scores.values())
            max_score = max(scores)
            min_score = min(scores)
            spread = max_score - min_score
            
            if spread >= 25:
                insights.append({
                    "type": "spread",
                    "category": "performance_spread",
                    "insight": f"Wide performance gap of {spread:.1f} points between highest and lowest performing suppliers"
                })
            elif spread <= 10 and len(scores) >= 3:
                insights.append({
                    "type": "spread",
                    "category": "performance_spread",
                    "insight": f"Consistent supplier performance with only {spread:.1f} points difference across suppliers"
                })
        
        return insights
    
    def _generate_mock_supplier_data(
        self,
        supplier_id: str,
        time_period: str = "last_quarter",
        error: bool = False
    ) -> Dict[str, Any]:
        """
        Generate mock supplier data for testing.
        
        Args:
            supplier_id: Supplier ID
            time_period: Time period for analysis
            error: Whether to simulate an error
            
        Returns:
            Dictionary with mock supplier data
        """
        if error:
            return {
                "error": "Failed to retrieve supplier data",
                "supplier_id": supplier_id,
                "time_period": time_period,
                "is_mock_data": True
            }
        
        # Generate a deterministic random seed based on supplier_id
        seed = sum(ord(c) for c in supplier_id)
        np.random.seed(seed)
        
        # Generate supplier name
        supplier_name = f"Supplier {supplier_id}"
        
        # Generate mock metrics data
        metrics = {}
        
        # On-time delivery
        otd_value = np.random.uniform(80, 98)
        otd_target = 95
        otd_score = min(100, max(0, 100 - (otd_target - otd_value) * 5)) if otd_value < otd_target else 100
        
        metrics["on_time_delivery"] = {
            "value": round(otd_value, 1),
            "target": otd_target,
            "score": round(otd_score, 1),
            "unit": "%"
        }
        
        # Quality
        quality_value = np.random.uniform(95, 99.8)
        quality_target = 99
        quality_score = min(100, max(0, 100 - (quality_target - quality_value) * 20)) if quality_value < quality_target else 100
        
        metrics["quality"] = {
            "value": round(quality_value, 2),
            "target": quality_target,
            "score": round(quality_score, 1),
            "unit": "%"
        }
        
        # Price competitiveness (index where 100 is benchmark)
        price_value = np.random.uniform(85, 115)
        price_target = 100
        price_score = min(100, max(0, 100 - (price_value - price_target) * 2)) if price_value > price_target else 100
        
        metrics["price_competitiveness"] = {
            "value": round(price_value, 1),
            "target": price_target,
            "score": round(price_score, 1),
            "unit": "index"
        }
        
        # Responsiveness (hours, lower is better)
        resp_value = np.random.uniform(4, 48)
        resp_target = 24
        resp_score = min(100, max(0, 100 - (resp_value - resp_target) * 3)) if resp_value > resp_target else 100
        
        metrics["responsiveness"] = {
            "value": round(resp_value, 1),
            "target": resp_target,
            "score": round(resp_score, 1),
            "unit": "hours"
        }
        
        # Innovation (score out of 5)
        innov_value = np.random.uniform(2, 5)
        innov_target = 4
        innov_score = min(100, max(0, 100 * innov_value / innov_target)) if innov_target > 0 else 0
        
        metrics["innovation"] = {
            "value": round(innov_value, 1),
            "target": innov_target,
            "score": round(innov_score, 1),
            "unit": "score"
        }
        
        # Sustainability (score out of 100)
        sustain_value = np.random.uniform(50, 90)
        sustain_target = 80
        sustain_score = min(100, max(0, sustain_value))
        
        metrics["sustainability"] = {
            "value": round(sustain_value, 1),
            "target": sustain_target,
            "score": round(sustain_score, 1),
            "unit": "score"
        }
        
        # Generate trend data
        trends = {}
        periods = 4  # Quarters
        
        # Define period labels
        trend_periods = []
        current_quarter = (datetime.now().month - 1) // 3 + 1
        current_year = datetime.now().year
        
        for i in range(periods):
            quarter = current_quarter - i
            year = current_year
            
            while quarter <= 0:
                quarter += 4
                year -= 1
            
            trend_periods.insert(0, f"Q{quarter} {year}")
        
        # Generate trend values for each metric
        for metric_name in metrics.keys():
            base_value = metrics[metric_name]["score"]
            trend_factor = np.random.uniform(-0.1, 0.1)  # -10% to +10% trend
            noise = 0.05  # 5% random noise
            
            # Generate values with trend and noise
            values = [
                max(0, min(100, base_value * (1 + trend_factor * (i / periods)) + np.random.normal(0, noise * base_value)))
                for i in range(periods)
            ]
            
            trends[metric_name] = {
                "values": [round(v, 1) for v in values],
                "periods": trend_periods
            }
        
        # Generate overall score trend
        # Use weighted average of metric scores
        weights = self.metric_weights
        overall_values = []
        
        for i in range(periods):
            period_score = 0
            total_weight = 0
            
            for metric_name, metric_trend in trends.items():
                weight = weights.get(metric_name, 0)
                value = metric_trend["values"][i]
                
                period_score += value * weight
                total_weight += weight
            
            if total_weight > 0:
                period_score /= total_weight
            
            overall_values.append(round(period_score, 1))
        
        trends["overall_score"] = {
            "values": overall_values,
            "periods": trend_periods
        }
        
        # Compile supplier data
        supplier_data = {
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "time_period": time_period,
            "metrics": metrics,
            "trends": trends,
            "is_mock_data": True
        }
        
        return supplier_data
    
    def _generate_mock_category_benchmarks(
        self,
        category: str,
        time_period: str = "last_quarter",
        error: bool = False
    ) -> Dict[str, Any]:
        """
        Generate mock category benchmark data for testing.
        
        Args:
            category: Supplier category
            time_period: Time period for analysis
            error: Whether to simulate an error
            
        Returns:
            Dictionary with mock category benchmark data
        """
        if error:
            return {
                "error": "Failed to retrieve category benchmarks",
                "category": category,
                "time_period": time_period,
                "is_mock_data": True
            }
        
        # Generate a deterministic random seed based on category
        seed = sum(ord(c) for c in category)
        np.random.seed(seed)
        
        # Generate mock metric averages
        metric_averages = {
            "on_time_delivery": round(np.random.uniform(85, 92), 1),
            "quality": round(np.random.uniform(90, 95), 1),
            "price_competitiveness": round(np.random.uniform(80, 90), 1),
            "responsiveness": round(np.random.uniform(75, 85), 1),
            "innovation": round(np.random.uniform(70, 80), 1),
            "sustainability": round(np.random.uniform(70, 80), 1)
        }
        
        # Generate mock performance distribution
        performance_distribution = {
            "excellent": {
                "count": np.random.randint(1, 5),
                "percentage": 0  # Will calculate below
            },
            "good": {
                "count": np.random.randint(5, 15),
                "percentage": 0
            },
            "satisfactory": {
                "count": np.random.randint(10, 20),
                "percentage": 0
            },
            "needs_improvement": {
                "count": np.random.randint(3, 8),
                "percentage": 0
            },
            "poor": {
                "count": np.random.randint(1, 4),
                "percentage": 0
            }
        }
        
        # Calculate percentages
        total_suppliers = sum(data["count"] for data in performance_distribution.values())
        
        for rating, data in performance_distribution.items():
            data["percentage"] = round(data["count"] / total_suppliers * 100, 1) if total_suppliers > 0 else 0
        
        # Compile benchmark data
        benchmark_data = {
            "category": category,
            "time_period": time_period,
            "supplier_count": total_suppliers,
            "overall_average": round(np.random.uniform(75, 85), 1),
            "metric_averages": metric_averages,
            "performance_distribution": performance_distribution,
            "is_mock_data": True
        }
        
        return benchmark_data
                
                # Get supplier data
                supplier_data = await supplier_interface.get_supplier_data(
                    supplier_id=supplier_id,
                    time_period=time_period
                )
                
                return supplier_data
            
            # Generate mock data for demonstration
            supplier_data = self._generate_mock_supplier_data(
                supplier_id=supplier_id,
                time_period=time_period
            )
            
            return supplier_data
            
        except Exception as e:
            logger.error(f"Error getting supplier data: {str(e)}")
            
            # Generate mock data on error
            return self._generate_mock_supplier_data(
                supplier_id=supplier_id,
                time_period=time_period,
                error=True
            )
    
    async def _get_category_benchmarks(
        self,
        category: str,
        time_period: str = "last_quarter",
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get benchmark data for a supplier category.
        
        Args:
            category: Supplier category
            time_period: Time period for analysis
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Dictionary with category benchmark data
        """
        try:
            # Get data from database if client_id is provided
            if client_id:
                # Import supplier interface
                from app.db.interfaces.supplier_interface import SupplierInterface
                
                # Create interface
                supplier_interface = SupplierInterface(client_id=client_id, connection_id=connection_id)