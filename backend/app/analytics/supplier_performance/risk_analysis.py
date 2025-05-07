"""
Supplier Risk Analysis Module

This module provides functionality for analyzing and monitoring supplier risk factors,
generating risk scores, and providing mitigation recommendations.
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

class SupplierRiskAnalysis:
    """
    Analyzes and monitors supplier risk factors.
    """
    
    # Risk categories and weights
    RISK_CATEGORIES = {
        "financial": {
            "weight": 0.25,
            "factors": ["credit_score", "financial_stability", "payment_history"]
        },
        "operational": {
            "weight": 0.20,
            "factors": ["delivery_performance", "quality_consistency", "capacity_utilization"]
        },
        "compliance": {
            "weight": 0.15,
            "factors": ["regulatory_compliance", "certification_status", "ethical_standards"]
        },
        "geopolitical": {
            "weight": 0.15,
            "factors": ["country_risk", "political_stability", "natural_disaster_exposure"]
        },
        "strategic": {
            "weight": 0.15,
            "factors": ["market_position", "innovation_capability", "technological_leadership"]
        },
        "environmental": {
            "weight": 0.10,
            "factors": ["environmental_compliance", "sustainability_practices", "carbon_footprint"]
        }
    }
    
    # Risk level thresholds
    RISK_LEVELS = {
        "low": (0, 30),
        "medium": (30, 60),
        "high": (60, 80),
        "critical": (80, 100)
    }
    
    def __init__(
        self,
        risk_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the supplier risk analysis.
        
        Args:
            risk_weights: Optional custom weights for risk categories
        """
        # Use custom weights if provided, otherwise use defaults
        if risk_weights:
            # Update weights while preserving the factors
            for category, category_data in self.RISK_CATEGORIES.items():
                if category in risk_weights:
                    category_data["weight"] = risk_weights[category]
            
            # Normalize weights to ensure they sum to 1.0
            total_weight = sum(category_data["weight"] for category_data in self.RISK_CATEGORIES.values())
            if total_weight != 1.0:
                for category_data in self.RISK_CATEGORIES.values():
                    category_data["weight"] /= total_weight
    
    async def analyze_supplier_risk(
        self,
        supplier_id: str,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        include_historical: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze risk factors for a supplier.
        
        Args:
            supplier_id: Supplier ID
            client_id: Optional client ID
            connection_id: Optional connection ID
            include_historical: Whether to include historical risk data
            include_recommendations: Whether to include mitigation recommendations
            
        Returns:
            Dictionary with supplier risk analysis
        """
        try:
            # Get supplier risk data
            risk_data = await self._get_supplier_risk_data(
                supplier_id=supplier_id,
                client_id=client_id,
                connection_id=connection_id,
                include_historical=include_historical
            )
            
            # Calculate risk scores
            risk_scores = self._calculate_risk_scores(risk_data)
            
            # Calculate overall risk score
            overall_risk = self._calculate_overall_risk(risk_scores)
            
            # Determine risk level
            risk_level = self._get_risk_level(overall_risk)
            
            # Identify top risk factors
            top_risk_factors = self._identify_top_risk_factors(risk_scores)
            
            # Generate mitigation recommendations if requested
            recommendations = None
            if include_recommendations:
                recommendations = self._generate_recommendations(
                    risk_scores=risk_scores,
                    overall_risk=overall_risk,
                    risk_level=risk_level,
                    top_risk_factors=top_risk_factors
                )
            
            # Compile risk analysis
            analysis = {
                "supplier_id": supplier_id,
                "supplier_name": risk_data.get("supplier_name", "Unknown Supplier"),
                "analysis_date": datetime.now().isoformat(),
                "overall_risk": overall_risk,
                "risk_level": risk_level,
                "risk_scores": risk_scores,
                "top_risk_factors": top_risk_factors,
                "recommendations": recommendations
            }
            
            # Include historical data if available
            if include_historical and "historical" in risk_data:
                analysis["historical"] = risk_data["historical"]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing supplier risk: {str(e)}")
            return {
                "error": str(e),
                "supplier_id": supplier_id,
                "overall_risk": 0,
                "risk_level": "unknown"
            }
    
    async def generate_risk_heatmap(
        self,
        supplier_ids: List[str],
        risk_categories: Optional[List[str]] = None,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a risk heatmap for multiple suppliers.
        
        Args:
            supplier_ids: List of supplier IDs
            risk_categories: Optional list of risk categories to include
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Dictionary with risk heatmap data
        """
        try:
            # Use all categories if none specified
            if not risk_categories:
                risk_categories = list(self.RISK_CATEGORIES.keys())
            
            # Analyze risk for each supplier
            supplier_analyses = []
            for supplier_id in supplier_ids:
                analysis = await self.analyze_supplier_risk(
                    supplier_id=supplier_id,
                    client_id=client_id,
                    connection_id=connection_id,
                    include_historical=False,
                    include_recommendations=False
                )
                supplier_analyses.append(analysis)
            
            # Generate heatmap data
            heatmap_data = self._generate_heatmap_data(
                supplier_analyses=supplier_analyses,
                risk_categories=risk_categories
            )
            
            # Calculate risk statistics
            risk_statistics = self._calculate_risk_statistics(supplier_analyses)
            
            # Generate insights
            insights = self._generate_heatmap_insights(
                heatmap_data=heatmap_data,
                risk_statistics=risk_statistics
            )
            
            # Compile heatmap result
            result = {
                "analysis_date": datetime.now().isoformat(),
                "supplier_count": len(supplier_analyses),
                "risk_categories": risk_categories,
                "heatmap_data": heatmap_data,
                "risk_statistics": risk_statistics,
                "insights": insights
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating risk heatmap: {str(e)}")
            return {
                "error": str(e),
                "supplier_count": len(supplier_ids),
                "risk_categories": risk_categories
            }
    
    async def simulate_disruption_impact(
        self,
        disruption_scenario: str,
        supplier_ids: Optional[List[str]] = None,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simulate the impact of a potential disruption.
        
        Args:
            disruption_scenario: Type of disruption scenario
            supplier_ids: Optional list of supplier IDs to include
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Dictionary with disruption impact analysis
        """
        try:
            # Get suppliers to analyze
            if not supplier_ids and client_id:
                # Get all suppliers for this client if none specified
                supplier_ids = await self._get_all_supplier_ids(
                    client_id=client_id,
                    connection_id=connection_id
                )
            
            if not supplier_ids:
                return {
                    "error": "No suppliers specified for analysis",
                    "disruption_scenario": disruption_scenario
                }
            
            # Get disruption parameters
            disruption_params = self._get_disruption_parameters(disruption_scenario)
            
            # Analyze each supplier's risk
            supplier_analyses = []
            for supplier_id in supplier_ids:
                analysis = await self.analyze_supplier_risk(
                    supplier_id=supplier_id,
                    client_id=client_id,
                    connection_id=connection_id,
                    include_historical=False,
                    include_recommendations=False
                )
                supplier_analyses.append(analysis)
            
            # Simulate disruption impact
            impact_data = self._simulate_disruption(
                supplier_analyses=supplier_analyses,
                disruption_params=disruption_params
            )
            
            # Generate recommendations
            recommendations = self._generate_disruption_recommendations(
                impact_data=impact_data,
                disruption_scenario=disruption_scenario
            )
            
            # Compile simulation result
            result = {
                "analysis_date": datetime.now().isoformat(),
                "disruption_scenario": disruption_scenario,
                "supplier_count": len(supplier_analyses),
                "impact_data": impact_data,
                "recommendations": recommendations
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error simulating disruption impact: {str(e)}")
            return {
                "error": str(e),
                "disruption_scenario": disruption_scenario
            }
    
    async def _get_supplier_risk_data(
        self,
        supplier_id: str,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        include_historical: bool = True
    ) -> Dict[str, Any]:
        """
        Get supplier risk data.
        
        Args:
            supplier_id: Supplier ID
            client_id: Optional client ID
            connection_id: Optional connection ID
            include_historical: Whether to include historical risk data
            
        Returns:
            Dictionary with supplier risk data
        """
        try:
            # Get data from database if client_id is provided
            if client_id:
                # Import supplier interface
                from app.db.interfaces.supplier_interface import SupplierInterface
                
                # Create interface
                supplier_interface = SupplierInterface(client_id=client_id, connection_id=connection_id)
                
                # Get risk data
                risk_data = await supplier_interface.get_supplier_risk_data(
                    supplier_id=supplier_id,
                    include_historical=include_historical
                )
                
                return risk_data
            
            # Generate mock data for demonstration
            risk_data = self._generate_mock_risk_data(
                supplier_id=supplier_id,
                include_historical=include_historical
            )
            
            return risk_data
            
        except Exception as e:
            logger.error(f"Error getting supplier risk data: {str(e)}")
            
            # Generate mock data on error
            return self._generate_mock_risk_data(
                supplier_id=supplier_id,
                include_historical=include_historical,
                error=True
            )
    
    async def _get_all_supplier_ids(
        self,
        client_id: str,
        connection_id: Optional[str] = None
    ) -> List[str]:
        """
        Get IDs of all suppliers for a client.
        
        Args:
            client_id: Client ID
            connection_id: Optional connection ID
            
        Returns:
            List of supplier IDs
        """
        try:
            # Import supplier interface
            from app.db.interfaces.supplier_interface import SupplierInterface
            
            # Create interface
            supplier_interface = SupplierInterface(client_id=client_id, connection_id=connection_id)
            
            # Get supplier IDs
            supplier_ids = await supplier_interface.get_all_supplier_ids()
            
            return supplier_ids
            
        except Exception as e:
            logger.error(f"Error getting all supplier IDs: {str(e)}")
            
            # Return mock supplier IDs
            return [f"SUPP-{i:04d}" for i in range(1, 11)]
    
    def _calculate_risk_scores(
        self,
        risk_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate risk scores from supplier risk data.
        
        Args:
            risk_data: Supplier risk data
            
        Returns:
            Dictionary with calculated risk scores
        """
        risk_scores = {}
        
        # Extract risk factors from data
        factors_data = risk_data.get("risk_factors", {})
        
        # Calculate score for each risk category
        for category, category_data in self.RISK_CATEGORIES.items():
            category_factors = category_data["factors"]
            
            # Initialize category score
            category_score = {
                "score": 0,
                "factors": {},
                "weight": category_data["weight"]
            }
            
            # Get scores for each factor in this category
            factor_scores = []
            for factor in category_factors:
                if factor in factors_data:
                    factor_data = factors_data[factor]
                    
                    factor_score = factor_data.get("score", 0)
                    factor_scores.append(factor_score)
                    
                    # Store factor details
                    category_score["factors"][factor] = {
                        "score": factor_score,
                        "raw_value": factor_data.get("value", 0),
                        "description": factor_data.get("description", "")
                    }
            
            # Calculate average score for this category
            if factor_scores:
                category_score["score"] = sum(factor_scores) / len(factor_scores)
            
            # Add category score
            risk_scores[category] = category_score
        
        return risk_scores
    
    def _calculate_overall_risk(
        self,
        risk_scores: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Calculate overall risk score.
        
        Args:
            risk_scores: Risk scores by category
            
        Returns:
            Overall risk score (0-100)
        """
        if not risk_scores:
            return 0
        
        overall_risk = 0
        total_weight = 0
        
        # Sum weighted category scores
        for category, category_data in risk_scores.items():
            score = category_data.get("score", 0)
            weight = category_data.get("weight", 0)
            
            overall_risk += score * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            overall_risk /= total_weight
        
        return round(overall_risk, 1)
    
    def _get_risk_level(
        self,
        risk_score: float
    ) -> str:
        """
        Get risk level based on risk score.
        
        Args:
            risk_score: Risk score
            
        Returns:
            Risk level
        """
        for level, (min_score, max_score) in self.RISK_LEVELS.items():
            if min_score <= risk_score < max_score:
                return level
        
        return "unknown"
    
    def _identify_top_risk_factors(
        self,
        risk_scores: Dict[str, Dict[str, Any]],
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify top risk factors.
        
        Args:
            risk_scores: Risk scores by category
            count: Number of top factors to identify
            
        Returns:
            List of top risk factors
        """
        # Collect all factors with scores
        all_factors = []
        
        for category, category_data in risk_scores.items():
            factors = category_data.get("factors", {})
            
            for factor_name, factor_data in factors.items():
                all_factors.append({
                    "category": category,
                    "factor": factor_name,
                    "score": factor_data.get("score", 0),
                    "description": factor_data.get("description", "")
                })
        
        # Sort factors by score (descending)
        all_factors.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top factors
        return all_factors[:count]
    
    def _generate_recommendations(
        self,
        risk_scores: Dict[str, Dict[str, Any]],
        overall_risk: float,
        risk_level: str,
        top_risk_factors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate risk mitigation recommendations.
        
        Args:
            risk_scores: Risk scores by category
            overall_risk: Overall risk score
            risk_level: Risk level
            top_risk_factors: Top risk factors
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate recommendations based on risk level
        if risk_level in ["high", "critical"]:
            recommendations.append({
                "priority": "high",
                "category": "general",
                "recommendation": "Develop comprehensive risk mitigation plan",
                "expected_impact": "Systematic approach to addressing multiple risk factors"
            })
            
            recommendations.append({
                "priority": "high",
                "category": "general",
                "recommendation": "Increase monitoring frequency for critical risk indicators",
                "expected_impact": "Early detection of risk escalation"
            })
            
            recommendations.append({
                "priority": "high",
                "category": "general",
                "recommendation": "Identify alternative suppliers for critical components/services",
                "expected_impact": "Reduced supplier concentration risk"
            })
        
        # Generate recommendations for top risk factors
        for factor in top_risk_factors[:3]:  # Focus on top 3
            category = factor["category"]
            factor_name = factor["factor"]
            
            if category == "financial":
                if factor_name == "financial_stability":
                    recommendations.append({
                        "priority": "high",
                        "category": "financial",
                        "recommendation": "Request updated financial statements and conduct detailed financial analysis",
                        "expected_impact": "Better understanding of financial health and early warning indicators"
                    })
                elif factor_name == "credit_score":
                    recommendations.append({
                        "priority": "medium",
                        "category": "financial",
                        "recommendation": "Adjust payment terms to reduce financial exposure",
                        "expected_impact": "Lower financial risk in case of supplier default"
                    })
            
            elif category == "operational":
                if factor_name == "delivery_performance":
                    recommendations.append({
                        "priority": "high",
                        "category": "operational",
                        "recommendation": "Implement weekly delivery performance tracking and escalation process",
                        "expected_impact": "Early identification and resolution of delivery issues"
                    })
                elif factor_name == "quality_consistency":
                    recommendations.append({
                        "priority": "high",
                        "category": "operational",
                        "recommendation": "Increase inspection frequency and implement statistical process control",
                        "expected_impact": "Early detection of quality issues and root cause identification"
                    })
                elif factor_name == "capacity_utilization":
                    recommendations.append({
                        "priority": "medium",
                        "category": "operational",
                        "recommendation": "Conduct site visit to assess capacity constraints and expansion plans",
                        "expected_impact": "Better understanding of capacity risks and mitigation options"
                    })
            
            elif category == "compliance":
                if factor_name in ["regulatory_compliance", "certification_status"]:
                    recommendations.append({
                        "priority": "high",
                        "category": "compliance",
                        "recommendation": "Conduct comprehensive compliance audit and certification verification",
                        "expected_impact": "Identification of compliance gaps and remediation requirements"
                    })
            
            elif category == "geopolitical":
                if factor_name in ["country_risk", "political_stability"]:
                    recommendations.append({
                        "priority": "medium",
                        "category": "geopolitical",
                        "recommendation": "Develop contingency plans for geopolitical disruptions",
                        "expected_impact": "Improved preparedness for geopolitical events"
                    })
            
            elif category == "strategic":
                if factor_name in ["market_position", "innovation_capability"]:
                    recommendations.append({
                        "priority": "medium",
                        "category": "strategic",
                        "recommendation": "Assess supplier's strategic roadmap and market position",
                        "expected_impact": "Better understanding of supplier's long-term viability"
                    })
            
            elif category == "environmental":
                if factor_name in ["environmental_compliance", "sustainability_practices"]:
                    recommendations.append({
                        "priority": "medium",
                        "category": "environmental",
                        "recommendation": "Request environmental compliance documentation and sustainability roadmap",
                        "expected_impact": "Verification of environmental compliance and sustainability commitments"
                    })
        
        # Add general recommendations
        if risk_level in ["medium", "high", "critical"]:
            recommendations.append({
                "priority": "medium",
                "category": "general",
                "recommendation": "Schedule quarterly risk review meetings with supplier",
                "expected_impact": "Regular monitoring and discussion of key risk factors"
            })
        
        return recommendations
    
    def _generate_heatmap_data(
        self,
        supplier_analyses: List[Dict[str, Any]],
        risk_categories: List[str]
    ) -> Dict[str, Any]:
        """
        Generate data for a risk heatmap.
        
        Args:
            supplier_analyses: List of supplier risk analyses
            risk_categories: List of risk categories to include
            
        Returns:
            Dictionary with heatmap data
        """
        heatmap_data = {
            "suppliers": [],
            "categories": risk_categories,
            "data": []
        }
        
        # Extract data for each supplier
        for analysis in supplier_analyses:
            supplier_id = analysis.get("supplier_id", "unknown")
            supplier_name = analysis.get("supplier_name", f"Supplier {supplier_id}")
            risk_scores = analysis.get("risk_scores", {})
            
            # Add supplier to list
            heatmap_data["suppliers"].append({
                "id": supplier_id,
                "name": supplier_name,
                "overall_risk": analysis.get("overall_risk", 0),
                "risk_level": analysis.get("risk_level", "unknown")
            })
            
            # Add category scores for this supplier
            for category in risk_categories:
                if category in risk_scores:
                    score = risk_scores[category].get("score", 0)
                else:
                    score = 0
                
                heatmap_data["data"].append({
                    "supplier_id": supplier_id,
                    "category": category,
                    "score": score,
                    "risk_level": self._get_risk_level(score)
                })
        
        return heatmap_data
    
    def _calculate_risk_statistics(
        self,
        supplier_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate risk statistics from supplier analyses.
        
        Args:
            supplier_analyses: List of supplier risk analyses
            
        Returns:
            Dictionary with risk statistics
        """
        statistics = {
            "overall": {},
            "by_category": {},
            "risk_distribution": {}
        }
        
        # Calculate overall risk statistics
        overall_risks = [
            analysis.get("overall_risk", 0)
            for analysis in supplier_analyses
        ]
        
        if overall_risks:
            statistics["overall"] = {
                "average": round(sum(overall_risks) / len(overall_risks), 1),
                "min": round(min(overall_risks), 1),
                "max": round(max(overall_risks), 1),
                "median": round(sorted(overall_risks)[len(overall_risks) // 2], 1)
            }
        
        # Calculate statistics by category
        category_scores = {}
        
        for analysis in supplier_analyses:
            risk_scores = analysis.get("risk_scores", {})
            
            for category, category_data in risk_scores.items():
                if category not in category_scores:
                    category_scores[category] = []
                
                category_scores[category].append(category_data.get("score", 0))
        
        # Calculate statistics for each category
        for category, scores in category_scores.items():
            if scores:
                statistics["by_category"][category] = {
                    "average": round(sum(scores) / len(scores), 1),
                    "min": round(min(scores), 1),
                    "max": round(max(scores), 1),
                    "median": round(sorted(scores)[len(scores) // 2], 1)
                }
        
        # Calculate risk level distribution
        risk_levels = [
            analysis.get("risk_level", "unknown")
            for analysis in supplier_analyses
        ]
        
        level_counts = {}
        for level in self.RISK_LEVELS.keys():
            level_counts[level] = risk_levels.count(level)
        
        total_suppliers = len(supplier_analyses)
        for level, count in level_counts.items():
            statistics["risk_distribution"][level] = {
                "count": count,
                "percentage": round(count / total_suppliers * 100, 1) if total_suppliers > 0 else 0
            }
        
        return statistics
    
    def _generate_heatmap_insights(
        self,
        heatmap_data: Dict[str, Any],
        risk_statistics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from risk heatmap data.
        
        Args:
            heatmap_data: Risk heatmap data
            risk_statistics: Risk statistics
            
        Returns:
            List of insights
        """
        insights = []
        
        # Generate insights on risk distribution
        distribution = risk_statistics.get("risk_distribution", {})
        high_critical_pct = 0
        
        for level in ["high", "critical"]:
            if level in distribution:
                high_critical_pct += distribution[level].get("percentage", 0)
        
        if high_critical_pct >= 30:
            insights.append({
                "type": "risk_distribution",
                "category": "overall",
                "insight": f"{high_critical_pct}% of suppliers have high or critical risk levels"
            })
        elif high_critical_pct <= 10 and high_critical_pct > 0:
            insights.append({
                "type": "risk_distribution",
                "category": "overall",
                "insight": f"Only {high_critical_pct}% of suppliers have high or critical risk levels"
            })
        
        # Generate insights on category risks
        category_stats = risk_statistics.get("by_category", {})
        
        # Identify highest risk category
        if category_stats:
            sorted_categories = sorted(
                category_stats.items(),
                key=lambda x: x[1]["average"],
                reverse=True
            )
            
            highest_category = sorted_categories[0]
            category_name, category_stats = highest_category
            
            if category_stats["average"] >= 60:
                insights.append({
                    "type": "category_risk",
                    "category": category_name,
                    "insight": f"'{category_name}' is the highest risk category with an average score of {category_stats['average']}"
                })
            
            # Identify lowest risk category
            lowest_category = sorted_categories[-1]
            category_name, category_stats = lowest_category
            
            insights.append({
                "type": "category_risk",
                "category": category_name,
                "insight": f"'{category_name}' is the lowest risk category with an average score of {category_stats['average']}"
            })
        
        # Generate insights on specific suppliers
        suppliers = heatmap_data.get("suppliers", [])
        
        # Identify suppliers with critical risk
        critical_suppliers = [
            supplier for supplier in suppliers
            if supplier.get("risk_level") == "critical"
        ]
        
        if critical_suppliers:
            supplier_names = [supplier["name"] for supplier in critical_suppliers[:3]]
            supplier_list = ", ".join(supplier_names)
            
            if len(critical_suppliers) > 3:
                supplier_list += f" and {len(critical_suppliers) - 3} others"
            
            insights.append({
                "type": "critical_suppliers",
                "category": "overall",
                "insight": f"Critical risk level identified for suppliers: {supplier_list}"
            })
        
        # Generate insights on risk spread
        overall_stats = risk_statistics.get("overall", {})
        if "max" in overall_stats and "min" in overall_stats:
            risk_spread = overall_stats["max"] - overall_stats["min"]
            
            if risk_spread >= 40:
                insights.append({
                    "type": "risk_spread",
                    "category": "overall",
                    "insight": f"Wide range of risk levels across suppliers ({risk_spread} points between highest and lowest)"
                })
            elif risk_spread <= 20:
                insights.append({
                    "type": "risk_spread",
                    "category": "overall",
                    "insight": f"Consistent risk levels across suppliers (only {risk_spread} points between highest and lowest)"
                })
        
        return insights
    
    def _get_disruption_parameters(
        self,
        disruption_scenario: str
    ) -> Dict[str, Any]:
        """
        Get parameters for a disruption scenario.
        
        Args:
            disruption_scenario: Type of disruption scenario
            
        Returns:
            Dictionary with disruption parameters
        """
        # Define parameters for common disruption scenarios
        parameters = {
            "pandemic": {
                "description": "Global pandemic affecting manufacturing and logistics",
                "category_impacts": {
                    "operational": 0.8,
                    "financial": 0.6,
                    "geopolitical": 0.5,
                    "strategic": 0.4,
                    "compliance": 0.3,
                    "environmental": 0.2
                },
                "duration_months": 12,
                "recovery_rate": 0.1  # 10% recovery per month
            },
            "natural_disaster": {
                "description": "Major natural disaster in key manufacturing region",
                "category_impacts": {
                    "operational": 0.9,
                    "financial": 0.5,
                    "geopolitical": 0.4,
                    "strategic": 0.3,
                    "compliance": 0.2,
                    "environmental": 0.6
                },
                "duration_months": 6,
                "recovery_rate": 0.15  # 15% recovery per month
            },
            "trade_war": {
                "description": "Trade war affecting tariffs and import/export restrictions",
                "category_impacts": {
                    "operational": 0.5,
                    "financial": 0.7,
                    "geopolitical": 0.9,
                    "strategic": 0.6,
                    "compliance": 0.4,
                    "environmental": 0.1
                },
                "duration_months": 24,
                "recovery_rate": 0.05  # 5% recovery per month
            },
            "cyber_attack": {
                "description": "Major cyber attack affecting supply chain systems",
                "category_impacts": {
                    "operational": 0.7,
                    "financial": 0.5,
                    "geopolitical": 0.2,
                    "strategic": 0.4,
                    "compliance": 0.6,
                    "environmental": 0.1
                },
                "duration_months": 3,
                "recovery_rate": 0.3  # 30% recovery per month
            },
            "financial_crisis": {
                "description": "Global financial crisis affecting credit and demand",
                "category_impacts": {
                    "operational": 0.5,
                    "financial": 0.9,
                    "geopolitical": 0.6,
                    "strategic": 0.7,
                    "compliance": 0.3,
                    "environmental": 0.2
                },
                "duration_months": 36,
                "recovery_rate": 0.03  # 3% recovery per month
            }
        }
        
        # Get parameters for the requested scenario
        if disruption_scenario in parameters:
            return parameters[disruption_scenario]
        else:
            # Default parameters for unknown scenarios
            return {
                "description": f"Generic disruption scenario: {disruption_scenario}",
                "category_impacts": {
                    "operational": 0.5,
                    "financial": 0.5,
                    "geopolitical": 0.5,
                    "strategic": 0.5,
                    "compliance": 0.3,
                    "environmental": 0.3
                },
                "duration_months": 12,
                "recovery_rate": 0.1
            }
    
    def _simulate_disruption(
        self,
        supplier_analyses: List[Dict[str, Any]],
        disruption_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate disruption impact on suppliers.
        
        Args:
            supplier_analyses: List of supplier risk analyses
            disruption_params: Disruption parameters
            
        Returns:
            Dictionary with disruption impact data
        """
        impact_data = {
            "disruption_description": disruption_params.get("description", "Unknown disruption"),
            "duration_months": disruption_params.get("duration_months", 12),
            "recovery_rate": disruption_params.get("recovery_rate", 0.1),
            "supplier_impacts": [],
            "overall_impact": {
                "high_impact_count": 0,
                "medium_impact_count": 0,
                "low_impact_count": 0
            }
        }
        
        # Get category impacts
        category_impacts = disruption_params.get("category_impacts", {})
        
        # Calculate impact for each supplier
        for analysis in supplier_analyses:
            supplier_id = analysis.get("supplier_id", "unknown")
            supplier_name = analysis.get("supplier_name", f"Supplier {supplier_id}")
            risk_scores = analysis.get("risk_scores", {})
            
            # Calculate impact for each category
            category_impact_scores = {}
            total_impact = 0
            weighted_impact = 0
            total_weight = 0
            
            for category, impact_factor in category_impacts.items():
                if category in risk_scores:
                    risk_score = risk_scores[category].get("score", 0)
                    category_weight = risk_scores[category].get("weight", 0)
                    
                    # Calculate impact as risk_score * impact_factor
                    impact_score = risk_score * impact_factor
                    
                    category_impact_scores[category] = {
                        "risk_score": risk_score,
                        "impact_factor": impact_factor,
                        "impact_score": impact_score
                    }
                    
                    # Add to total impact
                    weighted_impact += impact_score * category_weight
                    total_weight += category_weight
            
            # Calculate overall impact score
            if total_weight > 0:
                total_impact = weighted_impact / total_weight
            
            # Determine impact level
            impact_level = "low"
            if total_impact >= 60:
                impact_level = "high"
                impact_data["overall_impact"]["high_impact_count"] += 1
            elif total_impact >= 30:
                impact_level = "medium"
                impact_data["overall_impact"]["medium_impact_count"] += 1
            else:
                impact_data["overall_impact"]["low_impact_count"] += 1
            
            # Calculate recovery timeline
            recovery_rate = disruption_params.get("recovery_rate", 0.1)
            recovery_months = math.ceil(total_impact / (100 * recovery_rate))
            
            # Add to supplier impacts
            impact_data["supplier_impacts"].append({
                "supplier_id": supplier_id,
                "supplier_name": supplier_name,
                "total_impact": round(total_impact, 1),
                "impact_level": impact_level,
                "category_impacts": category_impact_scores,
                "estimated_recovery_months": recovery_months
            })
        
        # Sort supplier impacts by total impact (descending)
        impact_data["supplier_impacts"].sort(key=lambda x: x["total_impact"], reverse=True)
        
        # Calculate percentage distributions
        total_suppliers = len(supplier_analyses)
        if total_suppliers > 0:
            impact_data["overall_impact"]["high_impact_percentage"] = round(
                impact_data["overall_impact"]["high_impact_count"] / total_suppliers * 100, 1
            )
            impact_data["overall_impact"]["medium_impact_percentage"] = round(
                impact_data["overall_impact"]["medium_impact_count"] / total_suppliers * 100, 1
            )
            impact_data["overall_impact"]["low_impact_percentage"] = round(
                impact_data["overall_impact"]["low_impact_count"] / total_suppliers * 100, 1
            )
        
        return impact_data
    
    def _generate_disruption_recommendations(
        self,
        impact_data: Dict[str, Any],
        disruption_scenario: str
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for disruption mitigation.
        
        Args:
            impact_data: Disruption impact data
            disruption_scenario: Type of disruption scenario
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # General recommendations based on disruption type
        if disruption_scenario == "pandemic":
            recommendations.append({
                "priority": "high",
                "category": "strategic",
                "recommendation": "Develop pandemic response plan with key suppliers",
                "expected_impact": "Improved preparedness and response coordination"
            })
            
            recommendations.append({
                "priority": "high",
                "category": "operational",
                "recommendation": "Establish remote monitoring capabilities for supplier operations",
                "expected_impact": "Maintain visibility during travel restrictions"
            })
            
            recommendations.append({
                "priority": "medium",
                "category": "financial",
                "recommendation": "Review payment terms for financially vulnerable suppliers",
                "expected_impact": "Prevent supplier financial failures"
            })
        
        elif disruption_scenario == "natural_disaster":
            recommendations.append({
                "priority": "high",
                "category": "operational",
                "recommendation": "Develop regional diversification strategy for critical components",
                "expected_impact": "Reduced geographic concentration risk"
            })
            
            recommendations.append({
                "priority": "high",
                "category": "strategic",
                "recommendation": "Establish backup production capabilities for critical items",
                "expected_impact": "Business continuity during regional disruptions"
            })
        
        elif disruption_scenario == "trade_war":
            recommendations.append({
                "priority": "high",
                "category": "strategic",
                "recommendation": "Develop multi-regional sourcing strategy",
                "expected_impact": "Reduced exposure to trade restrictions"
            })
            
            recommendations.append({
                "priority": "medium",
                "category": "financial",
                "recommendation": "Develop tariff impact scenarios and mitigation plans",
                "expected_impact": "Financial preparedness for tariff changes"
            })
        
        elif disruption_scenario == "cyber_attack":
            recommendations.append({
                "priority": "high",
                "category": "operational",
                "recommendation": "Establish secure communication channels with key suppliers",
                "expected_impact": "Maintain critical communications during system outages"
            })
            
            recommendations.append({
                "priority": "high",
                "category": "compliance",
                "recommendation": "Conduct cybersecurity assessments for key suppliers",
                "expected_impact": "Identify and address security vulnerabilities"
            })
        
        elif disruption_scenario == "financial_crisis":
            recommendations.append({
                "priority": "high",
                "category": "financial",
                "recommendation": "Conduct financial stress tests for critical suppliers",
                "expected_impact": "Identify financially vulnerable suppliers"
            })
            
            recommendations.append({
                "priority": "medium",
                "category": "strategic",
                "recommendation": "Develop alternative sourcing options for suppliers with high financial risk",
                "expected_impact": "Reduced exposure to supplier financial failures"
            })
        
        # High-impact supplier recommendations
        high_impact_suppliers = [
            supplier for supplier in impact_data.get("supplier_impacts", [])
            if supplier.get("impact_level") == "high"
        ]
        
        if high_impact_suppliers:
            # Extract the top 3 most impacted suppliers
            top_impacted = high_impact_suppliers[:3]
            
            suppliers_list = ", ".join(s["supplier_name"] for s in top_impacted)
            
            recommendations.append({
                "priority": "high",
                "category": "high_impact_suppliers",
                "recommendation": f"Develop dedicated risk mitigation plans for highest-impact suppliers: {suppliers_list}",
                "expected_impact": "Focused risk reduction for most vulnerable suppliers"
            })
            
            recommendations.append({
                "priority": "high",
                "category": "high_impact_suppliers",
                "recommendation": "Increase monitoring frequency for high-impact suppliers",
                "expected_impact": "Early detection of risk escalation"
            })
            
            recommendations.append({
                "priority": "high",
                "category": "high_impact_suppliers",
                "recommendation": "Identify alternative sources for critical items from high-impact suppliers",
                "expected_impact": "Supply continuity for critical items"
            })
        
        # General preparedness recommendations
        recommendations.append({
            "priority": "medium",
            "category": "general",
            "recommendation": "Develop and test business continuity plans with key suppliers",
            "expected_impact": "Improved response coordination and recovery time"
        })
        
        recommendations.append({
            "priority": "medium",
            "category": "general",
            "recommendation": "Establish crisis communication protocol with all suppliers",
            "expected_impact": "Effective communication during disruptions"
        })
        
        recommendations.append({
            "priority": "low",
            "category": "general",
            "recommendation": "Conduct regular disruption scenario planning exercises",
            "expected_impact": "Improved organizational preparedness for various disruptions"
        })
        
        return recommendations
    
    def _generate_mock_risk_data(
        self,
        supplier_id: str,
        include_historical: bool = True,
        error: bool = False
    ) -> Dict[str, Any]:
        """
        Generate mock supplier risk data for testing.
        
        Args:
            supplier_id: Supplier ID
            include_historical: Whether to include historical risk data
            error: Whether to simulate an error
            
        Returns:
            Dictionary with mock supplier risk data
        """
        if error:
            return {
                "error": "Failed to retrieve supplier risk data",
                "supplier_id": supplier_id,
                "is_mock_data": True
            }
        
        # Generate a deterministic random seed based on supplier_id
        seed = sum(ord(c) for c in supplier_id)
        np.random.seed(seed)
        
        # Generate supplier name
        supplier_name = f"Supplier {supplier_id}"
        
        # Generate risk factors for each category
        risk_factors = {}
        
        # Financial risk factors
        credit_score = np.random.uniform(50, 90)
        risk_factors["credit_score"] = {
            "value": round(credit_score, 1),
            "score": round(max(0, min(100, 100 - (90 - credit_score) * 2)), 1),
            "description": "Credit score and financial rating"
        }
        
        financial_stability = np.random.uniform(60, 95)
        risk_factors["financial_stability"] = {
            "value": round(financial_stability, 1),
            "score": round(max(0, min(100, 100 - (90 - financial_stability) * 2)), 1),
            "description": "Financial stability assessment"
        }
        
        payment_history = np.random.uniform(70, 98)
        risk_factors["payment_history"] = {
            "value": round(payment_history, 1),
            "score": round(max(0, min(100, 100 - (90 - payment_history) * 2)), 1),
            "description": "Payment history and reliability"
        }
        
        # Operational risk factors
        delivery_performance = np.random.uniform(75, 98)
        risk_factors["delivery_performance"] = {
            "value": round(delivery_performance, 1),
            "score": round(max(0, min(100, 100 - (95 - delivery_performance) * 5)), 1),
            "description": "On-time delivery performance"
        }
        
        quality_consistency = np.random.uniform(85, 99.5)
        risk_factors["quality_consistency"] = {
            "value": round(quality_consistency, 2),
            "score": round(max(0, min(100, 100 - (98 - quality_consistency) * 10)), 1),
            "description": "Product quality consistency"
        }
        
        capacity_utilization = np.random.uniform(60, 95)
        capacity_score = 0
        if capacity_utilization < 70:
            capacity_score = 70  # Underutilization risk
        elif capacity_utilization > 90:
            capacity_score = (95 - capacity_utilization) * 10  # Overutilization risk
        else:
            capacity_score = 100  # Optimal utilization
        
        risk_factors["capacity_utilization"] = {
            "value": round(capacity_utilization, 1),
            "score": round(capacity_score, 1),
            "description": "Production capacity utilization"
        }
        
        # Compliance risk factors
        regulatory_compliance = np.random.uniform(80, 100)
        risk_factors["regulatory_compliance"] = {
            "value": round(regulatory_compliance, 1),
            "score": round(max(0, min(100, 100 - (95 - regulatory_compliance) * 5)), 1),
            "description": "Regulatory compliance status"
        }
        
        certification_status = np.random.uniform(70, 100)
        risk_factors["certification_status"] = {
            "value": round(certification_status, 1),
            "score": round(max(0, min(100, 100 - (90 - certification_status) * 3)), 1),
            "description": "Status of required certifications"
        }
        
        ethical_standards = np.random.uniform(75, 100)
        risk_factors["ethical_standards"] = {
            "value": round(ethical_standards, 1),
            "score": round(max(0, min(100, 100 - (90 - ethical_standards) * 3)), 1),
            "description": "Adherence to ethical standards"
        }
        
        # Geopolitical risk factors
        country_risk = np.random.uniform(30, 90)
        risk_factors["country_risk"] = {
            "value": round(country_risk, 1),
            "score": round(max(0, min(100, 100 - (80 - country_risk) * 2)), 1) if country_risk < 80 else round(max(0, min(100, 100 - (country_risk - 80) * 5)), 1),
            "description": "Country risk rating"
        }
        
        political_stability = np.random.uniform(40, 90)
        risk_factors["political_stability"] = {
            "value": round(political_stability, 1),
            "score": round(max(0, min(100, 100 - (80 - political_stability) * 2)), 1) if political_stability < 80 else round(max(0, min(100, 100 - (political_stability - 80) * 5)), 1),
            "description": "Political stability assessment"
        }
        
        natural_disaster_exposure = np.random.uniform(20, 80)
        risk_factors["natural_disaster_exposure"] = {
            "value": round(natural_disaster_exposure, 1),
            "score": round(max(0, min(100, 100 - natural_disaster_exposure)), 1),
            "description": "Exposure to natural disasters"
        }
        
        # Strategic risk factors
        market_position = np.random.uniform(30, 90)
        risk_factors["market_position"] = {
            "value": round(market_position, 1),
            "score": round(max(0, min(100, market_position)), 1),
            "description": "Market position strength"
        }
        
        innovation_capability = np.random.uniform(30, 90)
        risk_factors["innovation_capability"] = {
            "value": round(innovation_capability, 1),
            "score": round(max(0, min(100, innovation_capability)), 1),
            "description": "Innovation and R&D capability"
        }
        
        technological_leadership = np.random.uniform(30, 90)
        risk_factors["technological_leadership"] = {
            "value": round(technological_leadership, 1),
            "score": round(max(0, min(100, technological_leadership)), 1),
            "description": "Technological leadership position"
        }
        
        # Environmental risk factors
        environmental_compliance = np.random.uniform(70, 100)
        risk_factors["environmental_compliance"] = {
            "value": round(environmental_compliance, 1),
            "score": round(max(0, min(100, 100 - (90 - environmental_compliance) * 3)), 1),
            "description": "Environmental compliance status"
        }
        
        sustainability_practices = np.random.uniform(50, 90)
        risk_factors["sustainability_practices"] = {
            "value": round(sustainability_practices, 1),
            "score": round(max(0, min(100, sustainability_practices)), 1),
            "description": "Sustainability practices assessment"
        }
        
        carbon_footprint = np.random.uniform(30, 80)
        risk_factors["carbon_footprint"] = {
            "value": round(carbon_footprint, 1),
            "score": round(max(0, min(100, 100 - carbon_footprint)), 1),
            "description": "Carbon footprint assessment"
        }
        
        # Generate historical data if requested
        historical_data = None
        if include_historical:
            historical_data = self._generate_mock_historical_data(supplier_id)
        
        # Compile risk data
        risk_data = {
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "risk_factors": risk_factors,
            "historical": historical_data,
            "is_mock_data": True
        }
        
        return risk_data
    
    def _generate_mock_historical_data(
        self,
        supplier_id: str
    ) -> Dict[str, Any]:
        """
        Generate mock historical risk data.
        
        Args:
            supplier_id: Supplier ID
            
        Returns:
            Dictionary with mock historical data
        """
        # Generate a deterministic random seed based on supplier_id
        seed = sum(ord(c) for c in supplier_id)
        np.random.seed(seed)
        
        # Generate historical data for the past 12 months
        months = 12
        history = {
            "periods": [],
            "overall_risk": [],
            "category_risks": {}
        }
        
        # Generate period labels (months)
        current_month = datetime.now().month
        current_year = datetime.now().year
        
        for i in range(months):
            month = current_month - i
            year = current_year
            
            while month <= 0:
                month += 12
                year -= 1
            
            history["periods"].insert(0, f"{year}-{month:02d}")
        
        # Generate risk trends for each category with some randomness and trend
        for category in self.RISK_CATEGORIES.keys():
            # Generate a base risk score for this category
            base_score = np.random.uniform(30, 70)
            
            # Generate a trend direction and magnitude
            trend_direction = np.random.choice([-1, 1])
            trend_magnitude = np.random.uniform(0, 20)
            
            # Generate values with trend and noise
            values = []
            for i in range(months):
                # Calculate trend component
                trend_component = trend_direction * trend_magnitude * (i / months)
                
                # Add random noise
                noise = np.random.normal(0, 5)
                
                # Calculate value
                value = base_score + trend_component + noise
                
                # Ensure value is within bounds
                value = max(0, min(100, value))
                
                values.append(round(value, 1))
            
            history["category_risks"][category] = values
        
        # Calculate overall risk for each period
        for i in range(months):
            period_risk = 0
            total_weight = 0
            
            for category, weights in self.RISK_CATEGORIES.items():
                weight = weights["weight"]
                value = history["category_risks"][category][i]
                
                period_risk += value * weight
                total_weight += weight
            
            if total_weight > 0:
                period_risk /= total_weight
            
            history["overall_risk"].append(round(period_risk, 1))
        
        return history