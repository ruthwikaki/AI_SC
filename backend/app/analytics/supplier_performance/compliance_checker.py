"""
Supplier Compliance Checker Module

This module provides functionality for checking and monitoring supplier compliance
with regulatory requirements, certifications, and company policies.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class SupplierComplianceChecker:
    """
    Checks and monitors supplier compliance with requirements.
    """
    
    # Compliance categories
    COMPLIANCE_CATEGORIES = {
        "regulatory": {
            "weight": 0.30,
            "requirements": [
                "export_compliance",
                "import_compliance",
                "product_safety",
                "environmental_regulations",
                "labor_laws"
            ]
        },
        "certifications": {
            "weight": 0.25,
            "requirements": [
                "quality_certifications",
                "industry_certifications",
                "environmental_certifications",
                "safety_certifications"
            ]
        },
        "policies": {
            "weight": 0.20,
            "requirements": [
                "code_of_conduct",
                "anti_corruption",
                "conflict_minerals",
                "data_protection"
            ]
        },
        "documentation": {
            "weight": 0.15,
            "requirements": [
                "technical_documentation",
                "compliance_declarations",
                "test_reports",
                "material_declarations"
            ]
        },
        "sustainability": {
            "weight": 0.10,
            "requirements": [
                "environmental_management",
                "carbon_emissions",
                "waste_management",
                "sustainable_sourcing"
            ]
        }
    }
    
    # Compliance status levels
    COMPLIANCE_LEVELS = {
        "compliant": "Meets all requirements",
        "minor_issues": "Minor non-compliance issues",
        "major_issues": "Major non-compliance issues",
        "non_compliant": "Does not meet critical requirements"
    }
    
    def __init__(
        self,
        category_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the supplier compliance checker.
        
        Args:
            category_weights: Optional custom weights for compliance categories
        """
        # Use custom weights if provided, otherwise use defaults
        if category_weights:
            # Update weights while preserving the requirements
            for category, category_data in self.COMPLIANCE_CATEGORIES.items():
                if category in category_weights:
                    category_data["weight"] = category_weights[category]
            
            # Normalize weights to ensure they sum to 1.0
            total_weight = sum(category_data["weight"] for category_data in self.COMPLIANCE_CATEGORIES.values())
            if total_weight != 1.0:
                for category_data in self.COMPLIANCE_CATEGORIES.values():
                    category_data["weight"] /= total_weight
    
    async def check_supplier_compliance(
        self,
        supplier_id: str,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        include_details: bool = True,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        Check compliance for a supplier.
        
        Args:
            supplier_id: Supplier ID
            client_id: Optional client ID
            connection_id: Optional connection ID
            include_details: Whether to include detailed compliance data
            include_history: Whether to include compliance history
            
        Returns:
            Dictionary with supplier compliance check
        """
        try:
            # Get supplier compliance data
            compliance_data = await self._get_supplier_compliance_data(
                supplier_id=supplier_id,
                client_id=client_id,
                connection_id=connection_id,
                include_history=include_history
            )
            
            # Calculate compliance scores
            compliance_scores = self._calculate_compliance_scores(compliance_data)
            
            # Calculate overall compliance score
            overall_score = self._calculate_overall_score(compliance_scores)
            
            # Determine compliance status
            compliance_status = self._determine_compliance_status(compliance_scores)
            
            # Identify non-compliance issues
            non_compliance_issues = self._identify_non_compliance_issues(compliance_data)
            
            # Generate actions required
            actions_required = self._generate_actions_required(non_compliance_issues)
            
            # Compile compliance check
            compliance_check = {
                "supplier_id": supplier_id,
                "supplier_name": compliance_data.get("supplier_name", "Unknown Supplier"),
                "check_date": datetime.now().isoformat(),
                "overall_score": overall_score,
                "compliance_status": compliance_status,
                "non_compliance_issues": non_compliance_issues,
                "actions_required": actions_required
            }
            
            # Include detailed scores if requested
            if include_details:
                compliance_check["compliance_scores"] = compliance_scores
            
            # Include history if available and requested
            if include_history and "history" in compliance_data:
                compliance_check["compliance_history"] = compliance_data["history"]
            
            return compliance_check
            
        except Exception as e:
            logger.error(f"Error checking supplier compliance: {str(e)}")
            return {
                "error": str(e),
                "supplier_id": supplier_id,
                "compliance_status": "unknown"
            }
    
    async def generate_compliance_dashboard(
        self,
        client_id: str,
        connection_id: Optional[str] = None,
        supplier_ids: Optional[List[str]] = None,
        compliance_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a compliance dashboard for multiple suppliers.
        
        Args:
            client_id: Client ID
            connection_id: Optional connection ID
            supplier_ids: Optional list of supplier IDs to include
            compliance_categories: Optional list of compliance categories to include
            
        Returns:
            Dictionary with compliance dashboard
        """
        try:
            # Get suppliers to include
            if not supplier_ids:
                # Get all suppliers for this client
                supplier_ids = await self._get_all_supplier_ids(
                    client_id=client_id,
                    connection_id=connection_id
                )
            
            # Use all categories if none specified
            if not compliance_categories:
                compliance_categories = list(self.COMPLIANCE_CATEGORIES.keys())
            
            # Check compliance for each supplier
            supplier_checks = []
            for supplier_id in supplier_ids:
                check = await self.check_supplier_compliance(
                    supplier_id=supplier_id,
                    client_id=client_id,
                    connection_id=connection_id,
                    include_details=True,
                    include_history=False
                )
                supplier_checks.append(check)
            
            # Calculate compliance statistics
            compliance_statistics = self._calculate_compliance_statistics(
                supplier_checks=supplier_checks,
                compliance_categories=compliance_categories
            )
            
            # Identify high-risk suppliers
            high_risk_suppliers = self._identify_high_risk_suppliers(supplier_checks)
            
            # Generate insights
            insights = self._generate_dashboard_insights(
                supplier_checks=supplier_checks,
                compliance_statistics=compliance_statistics,
                high_risk_suppliers=high_risk_suppliers
            )
            
            # Compile dashboard
            dashboard = {
                "generation_date": datetime.now().isoformat(),
                "supplier_count": len(supplier_checks),
                "compliance_categories": compliance_categories,
                "compliance_statistics": compliance_statistics,
                "high_risk_suppliers": high_risk_suppliers,
                "insights": insights,
                "supplier_checks": supplier_checks
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating compliance dashboard: {str(e)}")
            return {
                "error": str(e),
                "client_id": client_id,
                "supplier_count": len(supplier_ids) if supplier_ids else 0
            }
    
    async def check_regulatory_updates(
        self,
        regions: List[str],
        industries: List[str],
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check for regulatory updates that may affect supplier compliance.
        
        Args:
            regions: List of regions to check
            industries: List of industries to check
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Dictionary with regulatory updates
        """
        try:
            # Get regulatory updates
            updates = await self._get_regulatory_updates(
                regions=regions,
                industries=industries,
                client_id=client_id,
                connection_id=connection_id
            )
            
            # Identify affected suppliers
            affected_suppliers = await self._identify_affected_suppliers(
                updates=updates,
                client_id=client_id,
                connection_id=connection_id
            )
            
            # Generate recommended actions
            recommended_actions = self._generate_regulatory_actions(
                updates=updates,
                affected_suppliers=affected_suppliers
            )
            
            # Compile update check
            update_check = {
                "check_date": datetime.now().isoformat(),
                "regions": regions,
                "industries": industries,
                "updates": updates,
                "affected_suppliers": affected_suppliers,
                "recommended_actions": recommended_actions
            }
            
            return update_check
            
        except Exception as e:
            logger.error(f"Error checking regulatory updates: {str(e)}")
            return {
                "error": str(e),
                "regions": regions,
                "industries": industries
            }
    
    async def _get_supplier_compliance_data(
        self,
        supplier_id: str,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        Get supplier compliance data.
        
        Args:
            supplier_id: Supplier ID
            client_id: Optional client ID
            connection_id: Optional connection ID
            include_history: Whether to include compliance history
            
        Returns:
            Dictionary with supplier compliance data
        """
        try:
            # Get data from database if client_id is provided
            if client_id:
                # Import supplier interface
                from app.db.interfaces.supplier_interface import SupplierInterface
                
                # Create interface
                supplier_interface = SupplierInterface(client_id=client_id, connection_id=connection_id)
                
                # Get compliance data
                compliance_data = await supplier_interface.get_supplier_compliance_data(
                    supplier_id=supplier_id,
                    include_history=include_history
                )
                
                return compliance_data
            
            # Generate mock data for demonstration
            compliance_data = self._generate_mock_compliance_data(
                supplier_id=supplier_id,
                include_history=include_history
            )
            
            return compliance_data
            
        except Exception as e:
            logger.error(f"Error getting supplier compliance data: {str(e)}")
            
            # Generate mock data on error
            return self._generate_mock_compliance_data(
                supplier_id=supplier_id,
                include_history=include_history,
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
    
    async def _get_regulatory_updates(
        self,
        regions: List[str],
        industries: List[str],
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get regulatory updates for specified regions and industries.
        
        Args:
            regions: List of regions to check
            industries: List of industries to check
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            List of regulatory updates
        """
        try:
            # Import compliance interface
            if client_id:
                from app.db.interfaces.compliance_interface import ComplianceInterface
                
                # Create interface
                compliance_interface = ComplianceInterface(client_id=client_id, connection_id=connection_id)
                
                # Get regulatory updates
                updates = await compliance_interface.get_regulatory_updates(
                    regions=regions,
                    industries=industries
                )
                
                return updates
            
            # Generate mock updates for demonstration
            updates = self._generate_mock_regulatory_updates(regions, industries)
            
            return updates
            
        except Exception as e:
            logger.error(f"Error getting regulatory updates: {str(e)}")
            
            # Return empty list on error
            return []
    
    async def _identify_affected_suppliers(
        self,
        updates: List[Dict[str, Any]],
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify suppliers affected by regulatory updates.
        
        Args:
            updates: Regulatory updates
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Dictionary mapping supplier IDs to relevant updates
        """
        try:
            # Get all supplier IDs
            supplier_ids = await self._get_all_supplier_ids(
                client_id=client_id,
                connection_id=connection_id
            )
            
            # Get supplier data for each supplier
            affected_suppliers = {}
            
            for supplier_id in supplier_ids:
                # Get supplier data
                supplier_data = await self._get_supplier_compliance_data(
                    supplier_id=supplier_id,
                    client_id=client_id,
                    connection_id=connection_id,
                    include_history=False
                )
                
                # Get supplier regions and industries
                supplier_regions = supplier_data.get("regions", [])
                supplier_industries = supplier_data.get("industries", [])
                
                # Find relevant updates for this supplier
                relevant_updates = []
                
                for update in updates:
                    update_region = update.get("region", "")
                    update_industry = update.get("industry", "")
                    
                    if (update_region in supplier_regions or update_region == "Global") and \
                       (update_industry in supplier_industries or update_industry == "All"):
                        relevant_updates.append(update)
                
                # Add supplier to affected list if there are relevant updates
                if relevant_updates:
                    affected_suppliers[supplier_id] = {
                        "supplier_name": supplier_data.get("supplier_name", f"Supplier {supplier_id}"),
                        "regions": supplier_regions,
                        "industries": supplier_industries,
                        "relevant_updates": relevant_updates
                    }
            
            return affected_suppliers
            
        except Exception as e:
            logger.error(f"Error identifying affected suppliers: {str(e)}")
            
            # Return empty dictionary on error
            return {}
    
    def _calculate_compliance_scores(
        self,
        compliance_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate compliance scores from supplier compliance data.
        
        Args:
            compliance_data: Supplier compliance data
            
        Returns:
            Dictionary with calculated compliance scores
        """
        compliance_scores = {}
        
        # Extract compliance requirements from data
        requirements_data = compliance_data.get("requirements", {})
        
        # Calculate score for each compliance category
        for category, category_data in self.COMPLIANCE_CATEGORIES.items():
            category_requirements = category_data["requirements"]
            
            # Initialize category score
            category_score = {
                "score": 0,
                "requirements": {},
                "weight": category_data["weight"]
            }
            
            # Get scores for each requirement in this category
            requirement_scores = []
            for requirement in category_requirements:
                if requirement in requirements_data:
                    req_data = requirements_data[requirement]
                    
                    req_score = req_data.get("score", 0)
                    requirement_scores.append(req_score)
                    
                    # Store requirement details
                    category_score["requirements"][requirement] = {
                        "score": req_score,
                        "status": req_data.get("status", "unknown"),
                        "details": req_data.get("details", ""),
                        "expiration": req_data.get("expiration", None)
                    }
            
            # Calculate average score for this category
            if requirement_scores:
                category_score["score"] = sum(requirement_scores) / len(requirement_scores)
            
            # Add category score
            compliance_scores[category] = category_score
        
        return compliance_scores
    
    def _calculate_overall_score(
        self,
        compliance_scores: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Calculate overall compliance score.
        
        Args:
            compliance_scores: Compliance scores by category
            
        Returns:
            Overall compliance score (0-100)
        """
        if not compliance_scores:
            return 0
        
        overall_score = 0
        total_weight = 0
        
        # Sum weighted category scores
        for category, category_data in compliance_scores.items():
            score = category_data.get("score", 0)
            weight = category_data.get("weight", 0)
            
            overall_score += score * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            overall_score /= total_weight
        
        return round(overall_score, 1)
    
    def _determine_compliance_status(
        self,
        compliance_scores: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Determine overall compliance status.
        
        Args:
            compliance_scores: Compliance scores by category
            
        Returns:
            Compliance status
        """
        # Count number of non-compliant requirements
        non_compliant_count = 0
        major_issues_count = 0
        minor_issues_count = 0
        
        for category, category_data in compliance_scores.items():
            requirements = category_data.get("requirements", {})
            
            for req_name, req_data in requirements.items():
                status = req_data.get("status", "unknown")
                
                if status == "non_compliant":
                    non_compliant_count += 1
                elif status == "major_issues":
                    major_issues_count += 1
                elif status == "minor_issues":
                    minor_issues_count += 1
        
        # Determine overall status
        if non_compliant_count > 0:
            return "non_compliant"
        elif major_issues_count > 0:
            return "major_issues"
        elif minor_issues_count > 0:
            return "minor_issues"
        else:
            return "compliant"
    
    def _identify_non_compliance_issues(
        self,
        compliance_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify non-compliance issues.
        
        Args:
            compliance_data: Supplier compliance data
            
        Returns:
            List of non-compliance issues
        """
        issues = []
        
        # Extract requirements data
        requirements_data = compliance_data.get("requirements", {})
        
        # Collect all non-compliance issues
        for category, category_data in self.COMPLIANCE_CATEGORIES.items():
            category_requirements = category_data["requirements"]
            
            for requirement in category_requirements:
                if requirement in requirements_data:
                    req_data = requirements_data[requirement]
                    
                    status = req_data.get("status", "unknown")
                    
                    # Add issue if not compliant
                    if status in ["non_compliant", "major_issues", "minor_issues"]:
                        issues.append({
                            "category": category,
                            "requirement": requirement,
                            "status": status,
                            "details": req_data.get("details", ""),
                            "severity": "high" if status == "non_compliant" else "medium" if status == "major_issues" else "low"
                        })
        
        # Sort issues by severity (high to low)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        issues.sort(key=lambda x: severity_order.get(x["severity"], 3))
        
        return issues
    
    def _generate_actions_required(
        self,
        non_compliance_issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate actions required to address non-compliance issues.
        
        Args:
            non_compliance_issues: List of non-compliance issues
            
        Returns:
            List of required actions
        """
        actions = []
        
        # Generate action for each issue
        for issue in non_compliance_issues:
            category = issue["category"]
            requirement = issue["requirement"]
            severity = issue["severity"]
            
            # Generate action based on category and requirement
            action = {
                "priority": severity,
                "category": category,
                "requirement": requirement,
                "action": self._get_action_for_requirement(category, requirement, severity),
                "timeline": "immediate" if severity == "high" else "30 days" if severity == "medium" else "90 days"
            }
            
            actions.append(action)
        
        return actions
    
    def _get_action_for_requirement(
        self,
        category: str,
        requirement: str,
        severity: str
    ) -> str:
        """
        Get appropriate action for a specific requirement.
        
        Args:
            category: Compliance category
            requirement: Specific requirement
            severity: Issue severity
            
        Returns:
            Action description
        """
        # Define actions for different requirements
        actions = {
            "regulatory": {
                "export_compliance": "Review and update export compliance documentation",
                "import_compliance": "Ensure import documentation meets all requirements",
                "product_safety": "Conduct product safety assessment and testing",
                "environmental_regulations": "Verify compliance with environmental regulations",
                "labor_laws": "Audit labor practices and documentation"
            },
            "certifications": {
                "quality_certifications": "Obtain required quality certifications",
                "industry_certifications": "Update industry-specific certifications",
                "environmental_certifications": "Complete environmental certification process",
                "safety_certifications": "Renew safety certifications"
            },
            "policies": {
                "code_of_conduct": "Ensure adherence to code of conduct",
                "anti_corruption": "Implement anti-corruption compliance measures",
                "conflict_minerals": "Provide conflict minerals documentation",
                "data_protection": "Update data protection policies and practices"
            },
            "documentation": {
                "technical_documentation": "Complete and submit technical documentation",
                "compliance_declarations": "Provide missing compliance declarations",
                "test_reports": "Submit up-to-date test reports",
                "material_declarations": "Complete material declarations"
            },
            "sustainability": {
                "environmental_management": "Implement environmental management system",
                "carbon_emissions": "Provide carbon emissions data and reduction plan",
                "waste_management": "Document waste management practices",
                "sustainable_sourcing": "Implement sustainable sourcing program"
            }
        }
        
        # Get action for this requirement
        if category in actions and requirement in actions[category]:
            return actions[category][requirement]
        else:
            return f"Address non-compliance with {requirement}"
    
    def _calculate_compliance_statistics(
        self,
        supplier_checks: List[Dict[str, Any]],
        compliance_categories: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate compliance statistics from supplier checks.
        
        Args:
            supplier_checks: List of supplier compliance checks
            compliance_categories: List of compliance categories to include
            
        Returns:
            Dictionary with compliance statistics
        """
        statistics = {
            "overall": {},
            "by_category": {},
            "status_distribution": {}
        }
        
        # Calculate overall compliance statistics
        overall_scores = [
            check.get("overall_score", 0)
            for check in supplier_checks
        ]
        
        if overall_scores:
            statistics["overall"] = {
                "average": round(sum(overall_scores) / len(overall_scores), 1),
                "min": round(min(overall_scores), 1),
                "max": round(max(overall_scores), 1),
                "median": round(sorted(overall_scores)[len(overall_scores) // 2], 1)
            }
        
        # Calculate statistics by category
        category_scores = {}
        
        for check in supplier_checks:
            compliance_scores = check.get("compliance_scores", {})
            
            for category in compliance_categories:
                if category in compliance_scores:
                    if category not in category_scores:
                        category_scores[category] = []
                    
                    category_scores[category].append(compliance_scores[category].get("score", 0))
        
        # Calculate statistics for each category
        for category, scores in category_scores.items():
            if scores:
                statistics["by_category"][category] = {
                    "average": round(sum(scores) / len(scores), 1),
                    "min": round(min(scores), 1),
                    "max": round(max(scores), 1),
                    "median": round(sorted(scores)[len(scores) // 2], 1)
                }
        
        # Calculate compliance status distribution
        statuses = [
            check.get("compliance_status", "unknown")
            for check in supplier_checks
        ]
        
        status_counts = {}
        for status in self.COMPLIANCE_LEVELS.keys():
            status_counts[status] = statuses.count(status)
        
        total_suppliers = len(supplier_checks)
        for status, count in status_counts.items():
            statistics["status_distribution"][status] = {
                "count": count,
                "percentage": round(count / total_suppliers * 100, 1) if total_suppliers > 0 else 0
            }
        
        return statistics
    
    def _identify_high_risk_suppliers(
        self,
        supplier_checks: List[Dict[str, Any]],
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify high-risk suppliers based on compliance checks.
        
        Args:
            supplier_checks: List of supplier compliance checks
            count: Number of high-risk suppliers to identify
            
        Returns:
            List of high-risk suppliers
        """
        # Sort suppliers by compliance status (non_compliant first, then by score)
        status_order = {
            "non_compliant": 0,
            "major_issues": 1,
            "minor_issues": 2,
            "compliant": 3,
            "unknown": 4
        }
        
        sorted_suppliers = sorted(
            supplier_checks,
            key=lambda x: (status_order.get(x.get("compliance_status"), 4), x.get("overall_score", 100))
        )
        
        # Get top high-risk suppliers
        high_risk_suppliers = []
        
        for check in sorted_suppliers[:count]:
            high_risk_suppliers.append({
                "supplier_id": check.get("supplier_id", "unknown"),
                "supplier_name": check.get("supplier_name", f"Supplier {check.get('supplier_id', '')}"),
                "compliance_status": check.get("compliance_status", "unknown"),
                "overall_score": check.get("overall_score", 0),
                "non_compliance_count": len(check.get("non_compliance_issues", []))
            })
        
        return high_risk_suppliers
    
    def _generate_dashboard_insights(
        self,
        supplier_checks: List[Dict[str, Any]],
        compliance_statistics: Dict[str, Any],
        high_risk_suppliers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from compliance dashboard data.
        
        Args:
            supplier_checks: List of supplier compliance checks
            compliance_statistics: Compliance statistics
            high_risk_suppliers: List of high-risk suppliers
            
        Returns:
            List of insights
        """
        insights = []
        
        # Generate insights on compliance status distribution
        status_distribution = compliance_statistics.get("status_distribution", {})
        compliant_pct = status_distribution.get("compliant", {}).get("percentage", 0)
        non_compliant_pct = status_distribution.get("non_compliant", {}).get("percentage", 0)
        
        if compliant_pct >= 90:
            insights.append({
                "type": "positive",
                "category": "compliance_status",
                "insight": f"{compliant_pct}% of suppliers are fully compliant with requirements"
            })
        elif non_compliant_pct >= 10:
            insights.append({
                "type": "negative",
                "category": "compliance_status",
                "insight": f"{non_compliant_pct}% of suppliers are non-compliant with critical requirements"
            })
        
        # Generate insights on category performance
        category_stats = compliance_statistics.get("by_category", {})
        
        if category_stats:
            # Identify lowest performing category
            sorted_categories = sorted(
                category_stats.items(),
                key=lambda x: x[1]["average"]
            )
            
            lowest_category = sorted_categories[0]
            category_name, category_stats = lowest_category
            
            if category_stats["average"] < 70:
                insights.append({
                    "type": "category",
                    "category": category_name,
                    "insight": f"'{category_name}' has the lowest compliance score at {category_stats['average']}"
                })
        
        # Generate insights on high-risk suppliers
        if high_risk_suppliers:
            non_compliant_suppliers = [
                s for s in high_risk_suppliers
                if s["compliance_status"] == "non_compliant"
            ]
            
            if non_compliant_suppliers:
                supplier_names = [s["supplier_name"] for s in non_compliant_suppliers[:3]]
                supplier_list = ", ".join(supplier_names)
                
                if len(non_compliant_suppliers) > 3:
                    supplier_list += f" and {len(non_compliant_suppliers) - 3} others"
                
                insights.append({
                    "type": "high_risk",
                    "category": "non_compliant",
                    "insight": f"Non-compliant suppliers requiring immediate attention: {supplier_list}"
                })
        
        # Generate insights on common non-compliance issues
        issue_types = {}
        
        for check in supplier_checks:
            for issue in check.get("non_compliance_issues", []):
                requirement = issue.get("requirement", "unknown")
                severity = issue.get("severity", "low")
                
                if requirement not in issue_types:
                    issue_types[requirement] = {
                        "count": 0,
                        "high_severity": 0
                    }
                
                issue_types[requirement]["count"] += 1
                if severity == "high":
                    issue_types[requirement]["high_severity"] += 1
        
        # Sort issues by count
        sorted_issues = sorted(
            issue_types.items(),
            key=lambda x: (x[1]["high_severity"], x[1]["count"]),
            reverse=True
        )
        
        # Add insight for most common high-severity issue
        if sorted_issues and sorted_issues[0][1]["high_severity"] > 0:
            issue_name, issue_data = sorted_issues[0]
            insights.append({
                "type": "common_issue",
                "category": "high_severity",
                "insight": f"Most common high-severity issue: '{issue_name}' affecting {issue_data['high_severity']} suppliers"
            })
        
        return insights
    
    def _generate_regulatory_actions(
        self,
        updates: List[Dict[str, Any]],
        affected_suppliers: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommended actions for regulatory updates.
        
        Args:
            updates: Regulatory updates
            affected_suppliers: Affected suppliers
            
        Returns:
            List of recommended actions
        """
        actions = []
        
        # Generate action for each regulatory update
        for update in updates:
            update_id = update.get("id", "unknown")
            title = update.get("title", "Regulatory update")
            description = update.get("description", "")
            effective_date = update.get("effective_date", "")
            region = update.get("region", "")
            industry = update.get("industry", "")
            impact_level = update.get("impact_level", "medium")
            
            # Get count of affected suppliers for this update
            affected_count = 0
            for supplier_id, supplier_data in affected_suppliers.items():
                if any(u.get("id") == update_id for u in supplier_data.get("relevant_updates", [])):
                    affected_count += 1
            
            # Add action
            actions.append({
                "update_id": update_id,
                "title": title,
                "effective_date": effective_date,
                "region": region,
                "industry": industry,
                "affected_suppliers": affected_count,
                "priority": impact_level,
                "action": f"Notify and assess compliance with {title}",
                "timeline": "immediate" if impact_level == "high" else "30 days" if impact_level == "medium" else "60 days"
            })
        
        # Sort actions by priority (high to low)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        actions.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return actions
    
    def _generate_mock_compliance_data(
        self,
        supplier_id: str,
        include_history: bool = True,
        error: bool = False
    ) -> Dict[str, Any]:
        """
        Generate mock supplier compliance data for testing.
        
        Args:
            supplier_id: Supplier ID
            include_history: Whether to include compliance history
            error: Whether to simulate an error
            
        Returns:
            Dictionary with mock supplier compliance data
        """
        if error:
            return {
                "error": "Failed to retrieve supplier compliance data",
                "supplier_id": supplier_id,
                "is_mock_data": True
            }
        
        # Generate a deterministic random seed based on supplier_id
        seed = sum(ord(c) for c in supplier_id)
        np.random.seed(seed)
        
        # Generate supplier name
        supplier_name = f"Supplier {supplier_id}"
        
        # Generate mock regions and industries
        regions = np.random.choice(
            ["North America", "Europe", "Asia", "South America", "Africa"],
            size=np.random.randint(1, 3),
            replace=False
        ).tolist()
        
        industries = np.random.choice(
            ["Electronics", "Automotive", "Aerospace", "Chemical", "Medical", "Consumer Goods"],
            size=np.random.randint(1, 3),
            replace=False
        ).tolist()
        
        # Generate compliance requirements for each category
        requirements = {}
        
        # Regulatory requirements
        for req in self.COMPLIANCE_CATEGORIES["regulatory"]["requirements"]:
            # Determine compliance status
            status_choice = np.random.choice(
                ["compliant", "minor_issues", "major_issues", "non_compliant"],
                p=[0.7, 0.15, 0.1, 0.05]
            )
            
            # Generate score based on status
            if status_choice == "compliant":
                score = np.random.uniform(90, 100)
            elif status_choice == "minor_issues":
                score = np.random.uniform(70, 90)
            elif status_choice == "major_issues":
                score = np.random.uniform(50, 70)
            else:  # non_compliant
                score = np.random.uniform(0, 50)
            
            # Generate details
            details = f"{req} documentation "
            if status_choice == "compliant":
                details += "complete and up to date"
            elif status_choice == "minor_issues":
                details += "has minor issues requiring attention"
            elif status_choice == "major_issues":
                details += "has significant issues requiring immediate attention"
            else:  # non_compliant
                details += "missing or non-compliant"
            
            # Add requirement
            requirements[req] = {
                "status": status_choice,
                "score": round(score, 1),
                "details": details
            }
        
        # Certifications requirements
        for req in self.COMPLIANCE_CATEGORIES["certifications"]["requirements"]:
            # Determine compliance status
            status_choice = np.random.choice(
                ["compliant", "minor_issues", "major_issues", "non_compliant"],
                p=[0.8, 0.1, 0.05, 0.05]
            )
            
            # Generate score based on status
            if status_choice == "compliant":
                score = np.random.uniform(90, 100)
            elif status_choice == "minor_issues":
                score = np.random.uniform(70, 90)
            elif status_choice == "major_issues":
                score = np.random.uniform(50, 70)
            else:  # non_compliant
                score = np.random.uniform(0, 50)
            
            # Generate expiration date
expiration = None
                if status_choice == "compliant" or status_choice == "minor_issues":
                    # Generate future expiration date
                    days_to_expiration = np.random.randint(30, 365)
                    expiration = (datetime.now() + timedelta(days=days_to_expiration)).isoformat()
                elif status_choice == "major_issues":
                    # Could be expired or close to expiration
                    days_to_expiration = np.random.randint(-30, 30)
                    expiration = (datetime.now() + timedelta(days=days_to_expiration)).isoformat()
                elif status_choice == "non_compliant":
                    # Expired certification
                    days_to_expiration = np.random.randint(-365, -30)
                    expiration = (datetime.now() + timedelta(days=days_to_expiration)).isoformat()
            
                # Generate details
                details = f"{req} "
                if status_choice == "compliant":
                    details += "valid and current"
                elif status_choice == "minor_issues":
                    details += "valid but renewal required soon"
                elif status_choice == "major_issues":
                    details += "expired or missing required information"
                else:  # non_compliant
                    details += "missing or invalid"
                
                # Add requirement
                requirements[req] = {
                    "status": status_choice,
                    "score": round(score, 1),
                    "details": details,
                    "expiration": expiration
                }
        
        # Policies requirements
        for req in self.COMPLIANCE_CATEGORIES["policies"]["requirements"]:
            # Determine compliance status
            status_choice = np.random.choice(
                ["compliant", "minor_issues", "major_issues", "non_compliant"],
                p=[0.75, 0.15, 0.05, 0.05]
            )
            
            # Generate score based on status
            if status_choice == "compliant":
                score = np.random.uniform(90, 100)
            elif status_choice == "minor_issues":
                score = np.random.uniform(70, 90)
            elif status_choice == "major_issues":
                score = np.random.uniform(50, 70)
            else:  # non_compliant
                score = np.random.uniform(0, 50)
            
            # Generate details
            details = f"{req} policy "
            if status_choice == "compliant":
                details += "implemented and documented"
            elif status_choice == "minor_issues":
                details += "implemented but documentation incomplete"
            elif status_choice == "major_issues":
                details += "partially implemented with significant gaps"
            else:  # non_compliant
                details += "not implemented or documented"
            
            # Add requirement
            requirements[req] = {
                "status": status_choice,
                "score": round(score, 1),
                "details": details
            }
        
        # Documentation requirements
        for req in self.COMPLIANCE_CATEGORIES["documentation"]["requirements"]:
            # Determine compliance status
            status_choice = np.random.choice(
                ["compliant", "minor_issues", "major_issues", "non_compliant"],
                p=[0.7, 0.2, 0.05, 0.05]
            )
            
            # Generate score based on status
            if status_choice == "compliant":
                score = np.random.uniform(90, 100)
            elif status_choice == "minor_issues":
                score = np.random.uniform(70, 90)
            elif status_choice == "major_issues":
                score = np.random.uniform(50, 70)
            else:  # non_compliant
                score = np.random.uniform(0, 50)
            
            # Generate details
            details = f"{req} "
            if status_choice == "compliant":
                details += "complete and accurate"
            elif status_choice == "minor_issues":
                details += "mostly complete with minor inaccuracies"
            elif status_choice == "major_issues":
                details += "incomplete with major issues"
            else:  # non_compliant
                details += "missing or severely deficient"
            
            # Add requirement
            requirements[req] = {
                "status": status_choice,
                "score": round(score, 1),
                "details": details
            }
        
        # Sustainability requirements
        for req in self.COMPLIANCE_CATEGORIES["sustainability"]["requirements"]:
            # Determine compliance status
            status_choice = np.random.choice(
                ["compliant", "minor_issues", "major_issues", "non_compliant"],
                p=[0.6, 0.25, 0.1, 0.05]
            )
            
            # Generate score based on status
            if status_choice == "compliant":
                score = np.random.uniform(90, 100)
            elif status_choice == "minor_issues":
                score = np.random.uniform(70, 90)
            elif status_choice == "major_issues":
                score = np.random.uniform(50, 70)
            else:  # non_compliant
                score = np.random.uniform(0, 50)
            
            # Generate details
            details = f"{req} "
            if status_choice == "compliant":
                details += "meets all requirements"
            elif status_choice == "minor_issues":
                details += "meets most requirements with minor gaps"
            elif status_choice == "major_issues":
                details += "has significant gaps in implementation"
            else:  # non_compliant
                details += "not implemented or severely deficient"
            
            # Add requirement
            requirements[req] = {
                "status": status_choice,
                "score": round(score, 1),
                "details": details
            }
        
        # Generate compliance history if requested
        history = None
        if include_history:
            # Generate 3-6 historical checks
            num_checks = np.random.randint(3, 7)
            history = []
            
            for i in range(num_checks):
                # Generate date (going back in time)
                days_ago = (i + 1) * 90 + np.random.randint(-15, 15)
                check_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
                
                # Generate historical score (with some random variation)
                check_score = np.random.normal(70, 10)
                check_score = max(0, min(100, check_score))
                
                # Determine status based on score
                if check_score >= 90:
                    check_status = "compliant"
                elif check_score >= 70:
                    check_status = "minor_issues"
                elif check_score >= 50:
                    check_status = "major_issues"
                else:
                    check_status = "non_compliant"
                
                # Add historical check
                history.append({
                    "check_date": check_date,
                    "overall_score": round(check_score, 1),
                    "compliance_status": check_status
                })
            
            # Sort history by date (most recent first)
            history.sort(key=lambda x: x["check_date"], reverse=True)
        
        # Compile compliance data
        compliance_data = {
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "regions": regions,
            "industries": industries,
            "requirements": requirements,
            "is_mock_data": True
        }
        
        # Add history if included
        if include_history and history is not None:
            compliance_data["history"] = history
        
        return compliance_data
    
    def _generate_mock_regulatory_updates(
        self,
        regions: List[str],
        industries: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate mock regulatory updates for testing.
        
        Args:
            regions: List of regions to include
            industries: List of industries to include
            
        Returns:
            List of mock regulatory updates
        """
        # Define possible updates
        possible_updates = [
            {
                "id": "REG-2023-001",
                "title": "Enhanced Product Safety Standards",
                "description": "New safety standards for consumer products",
                "region": "Global",
                "industry": "Consumer Goods",
                "impact_level": "high",
                "effective_date": (datetime.now() + timedelta(days=90)).isoformat()
            },
            {
                "id": "REG-2023-002",
                "title": "Updated Environmental Reporting Requirements",
                "description": "New requirements for environmental impact reporting",
                "region": "Europe",
                "industry": "All",
                "impact_level": "medium",
                "effective_date": (datetime.now() + timedelta(days=120)).isoformat()
            },
            {
                "id": "REG-2023-003",
                "title": "Conflict Minerals Disclosure Rules",
                "description": "Enhanced disclosure requirements for conflict minerals",
                "region": "North America",
                "industry": "Electronics",
                "impact_level": "medium",
                "effective_date": (datetime.now() + timedelta(days=60)).isoformat()
            },
            {
                "id": "REG-2023-004",
                "title": "Chemical Substance Restrictions",
                "description": "New restrictions on use of specific chemical substances",
                "region": "Asia",
                "industry": "Chemical",
                "impact_level": "high",
                "effective_date": (datetime.now() + timedelta(days=30)).isoformat()
            },
            {
                "id": "REG-2023-005",
                "title": "Supply Chain Due Diligence Requirements",
                "description": "New requirements for supply chain due diligence",
                "region": "Europe",
                "industry": "All",
                "impact_level": "medium",
                "effective_date": (datetime.now() + timedelta(days=180)).isoformat()
            },
            {
                "id": "REG-2023-006",
                "title": "Medical Device Quality System Requirements",
                "description": "Updated quality system requirements for medical devices",
                "region": "Global",
                "industry": "Medical",
                "impact_level": "high",
                "effective_date": (datetime.now() + timedelta(days=45)).isoformat()
            },
            {
                "id": "REG-2023-007",
                "title": "Aerospace Component Testing Standards",
                "description": "New standards for testing aerospace components",
                "region": "North America",
                "industry": "Aerospace",
                "impact_level": "medium",
                "effective_date": (datetime.now() + timedelta(days=75)).isoformat()
            },
            {
                "id": "REG-2023-008",
                "title": "Automotive Emissions Standards",
                "description": "Updated emissions standards for automotive industry",
                "region": "Global",
                "industry": "Automotive",
                "impact_level": "high",
                "effective_date": (datetime.now() + timedelta(days=150)).isoformat()
            },
            {
                "id": "REG-2023-009",
                "title": "Data Protection Requirements",
                "description": "Enhanced data protection requirements for suppliers",
                "region": "Europe",
                "industry": "All",
                "impact_level": "medium",
                "effective_date": (datetime.now() + timedelta(days=100)).isoformat()
            },
            {
                "id": "REG-2023-010",
                "title": "Labor Practice Audit Requirements",
                "description": "New requirements for labor practice audits",
                "region": "Asia",
                "industry": "All",
                "impact_level": "medium",
                "effective_date": (datetime.now() + timedelta(days=90)).isoformat()
            }
        ]
        
        # Filter updates by region and industry
        filtered_updates = []
        
        for update in possible_updates:
            update_region = update["region"]
            update_industry = update["industry"]
            
            # Check if update should be included
            if (update_region == "Global" or update_region in regions) and \
               (update_industry == "All" or update_industry in industries):
                filtered_updates.append(update)
        
        return filtered_updates

# Example usage
async def example_usage():
    """
    Example usage of the SupplierComplianceChecker.
    """
    # Create checker
    checker = SupplierComplianceChecker()
    
    # Check compliance for a supplier
    compliance_check = await checker.check_supplier_compliance(
        supplier_id="SUPP-0001",
        include_details=True,
        include_history=True
    )
    
    # Print check results
    print(json.dumps(compliance_check, indent=2))
    
    # Generate compliance dashboard
    dashboard = await checker.generate_compliance_dashboard(
        client_id="CLIENT-001",
        compliance_categories=["regulatory", "certifications"]
    )
    
    # Print dashboard
    print(json.dumps(dashboard, indent=2))
    
    # Check regulatory updates
    updates = await checker.check_regulatory_updates(
        regions=["North America", "Europe"],
        industries=["Electronics", "Automotive"]
    )
    
    # Print updates
    print(json.dumps(updates, indent=2))

# Run example if script is executed directly
if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())