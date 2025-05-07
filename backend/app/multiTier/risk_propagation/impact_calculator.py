"""
Impact calculator module for multi-tier supply chain risk analysis.

This module provides functionality to calculate the business impact
of supply chain disruptions across multiple tiers of suppliers.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
import networkx as nx
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import math

from app.db.connectors.postgres import PostgresConnector
from app.db.schema.schema_discovery import discover_client_schema
from app.db.schema.schema_mapper import get_domain_mappings
from app.utils.logger import get_logger
from app.multiTier.supplier_mapping.network_builder import SupplierNetworkBuilder
from app.multiTier.risk_propagation.cascade_analyzer import RiskCascadeAnalyzer

# Initialize logger
logger = get_logger(__name__)

class DisruptionImpactCalculator:
    """
    Calculates the business impact of supply chain disruptions.
    
    This class translates disruption scenarios into tangible business
    impacts like financial costs, production delays, and service levels.
    """
    
    def __init__(
        self,
        client_id: str,
        connection_id: Optional[str] = None,
        cascade_analyzer: Optional[RiskCascadeAnalyzer] = None,
        network_builder: Optional[SupplierNetworkBuilder] = None
    ):
        """
        Initialize the disruption impact calculator.
        
        Args:
            client_id: Client identifier
            connection_id: Optional database connection identifier
            cascade_analyzer: Optional existing cascade analyzer instance
            network_builder: Optional existing network builder instance
        """
        self.client_id = client_id
        self.connection_id = connection_id
        self.cascade_analyzer = cascade_analyzer
        self.network_builder = network_builder
        self.db_connector = None
        self.schema = None
        self.domain_mappings = None
        self.graph = None
        self.product_data = {}
        self.material_data = {}
        
    async def initialize(self) -> None:
        """
        Initialize the impact calculator with client schema and connections.
        """
        try:
            # Get database connector
            self.db_connector = PostgresConnector(
                client_id=self.client_id,
                connection_id=self.connection_id
            )
            
            # Discover schema
            self.schema = await discover_client_schema(
                client_id=self.client_id,
                connection_id=self.connection_id
            )
            
            # Get domain mappings
            self.domain_mappings = await get_domain_mappings(self.client_id)
            
            # Initialize network builder if not provided
            if not self.network_builder:
                self.network_builder = SupplierNetworkBuilder(
                    client_id=self.client_id,
                    connection_id=self.connection_id
                )
                await self.network_builder.initialize()
            
            # Initialize cascade analyzer if not provided
            if not self.cascade_analyzer:
                self.cascade_analyzer = RiskCascadeAnalyzer(
                    client_id=self.client_id,
                    connection_id=self.connection_id,
                    network_builder=self.network_builder
                )
                await self.cascade_analyzer.initialize()
            
            # Load product and material data
            await self._load_product_data()
            await self._load_material_data()
            
            logger.info(f"Initialized disruption impact calculator for client: {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error initializing disruption impact calculator: {str(e)}")
            raise
    
    async def _load_product_data(self) -> None:
        """
        Load product data from the database.
        """
        try:
            # Reset product data
            self.product_data = {}
            
            # Find relevant tables
            product_table = self._find_table_by_domain("product")
            bom_table = self._find_table_by_domain("bill_of_materials")
            
            if not product_table:
                logger.warning("Could not find product table for impact analysis")
                return
            
            # Load product data
            query = f"""
            SELECT 
                p.*,
                COALESCE(sales_data.avg_daily_sales, 0) AS avg_daily_sales,
                COALESCE(sales_data.contribution_margin, 0) AS contribution_margin,
                COALESCE(inventory.stock_level, 0) AS current_stock,
                COALESCE(inventory.safety_stock, 0) AS safety_stock
            FROM 
                {product_table} p
            LEFT JOIN (
                SELECT 
                    product_id,
                    AVG(quantity) AS avg_daily_sales,
                    AVG(sale_price - cost) AS contribution_margin
                FROM 
                    sales
                WHERE 
                    sale_date >= CURRENT_DATE - INTERVAL '90 days'
                GROUP BY 
                    product_id
            ) sales_data ON p.id = sales_data.product_id
            LEFT JOIN (
                SELECT 
                    product_id,
                    SUM(quantity) AS stock_level,
                    safety_stock
                FROM 
                    inventory
                GROUP BY 
                    product_id, safety_stock
            ) inventory ON p.id = inventory.product_id
            """
            
            # Execute query
            result = await self.db_connector.execute_query(query)
            products = result.get("data", [])
            
            # Process product data
            for product in products:
                product_id = str(product.get("id"))
                self.product_data[product_id] = product
            
            # If BOM table exists, load product-material relationships
            if bom_table:
                query = f"""
                SELECT 
                    bom.product_id,
                    bom.material_id,
                    bom.quantity_required,
                    bom.supplier_id,
                    s.name AS supplier_name
                FROM 
                    {bom_table} bom
                LEFT JOIN
                    suppliers s ON bom.supplier_id = s.id
                """
                
                # Execute query
                result = await self.db_connector.execute_query(query)
                bom_data = result.get("data", [])
                
                # Process BOM data
                for bom_item in bom_data:
                    product_id = str(bom_item.get("product_id"))
                    material_id = str(bom_item.get("material_id"))
                    supplier_id = str(bom_item.get("supplier_id"))
                    
                    if product_id in self.product_data:
                        # Add materials to product data
                        if "materials" not in self.product_data[product_id]:
                            self.product_data[product_id]["materials"] = []
                        
                        self.product_data[product_id]["materials"].append({
                            "material_id": material_id,
                            "quantity_required": bom_item.get("quantity_required", 0),
                            "supplier_id": supplier_id,
                            "supplier_name": bom_item.get("supplier_name", "Unknown")
                        })
            
            logger.info(f"Loaded data for {len(self.product_data)} products")
            
        except Exception as e:
            logger.error(f"Error loading product data: {str(e)}")
            # Continue with partial data
    
    async def _load_material_data(self) -> None:
        """
        Load material data from the database.
        """
        try:
            # Reset material data
            self.material_data = {}
            
            # Find relevant tables
            material_table = self._find_table_by_domain("material")
            material_supplier_table = self._find_table_by_domain("material_supplier")
            
            if not material_table:
                logger.warning("Could not find material table for impact analysis")
                return
            
            # Load material data
            query = f"""
            SELECT 
                m.*,
                COALESCE(inventory.stock_level, 0) AS current_stock,
                COALESCE(inventory.safety_stock, 0) AS safety_stock,
                COALESCE(inventory.lead_time, 0) AS lead_time
            FROM 
                {material_table} m
            LEFT JOIN (
                SELECT 
                    material_id,
                    SUM(quantity) AS stock_level,
                    safety_stock,
                    AVG(lead_time) AS lead_time
                FROM 
                    material_inventory
                GROUP BY 
                    material_id, safety_stock
            ) inventory ON m.id = inventory.material_id
            """
            
            # Execute query
            result = await self.db_connector.execute_query(query)
            materials = result.get("data", [])
            
            # Process material data
            for material in materials:
                material_id = str(material.get("id"))
                self.material_data[material_id] = material
            
            # If material-supplier table exists, load material-supplier relationships
            if material_supplier_table:
                query = f"""
                SELECT 
                    ms.material_id,
                    ms.supplier_id,
                    s.name AS supplier_name,
                    ms.is_primary,
                    ms.lead_time,
                    ms.price_per_unit
                FROM 
                    {material_supplier_table} ms
                LEFT JOIN
                    suppliers s ON ms.supplier_id = s.id
                """
                
                # Execute query
                result = await self.db_connector.execute_query(query)
                supplier_data = result.get("data", [])
                
                # Process supplier data
                for supplier_item in supplier_data:
                    material_id = str(supplier_item.get("material_id"))
                    supplier_id = str(supplier_item.get("supplier_id"))
                    
                    if material_id in self.material_data:
                        # Add suppliers to material data
                        if "suppliers" not in self.material_data[material_id]:
                            self.material_data[material_id]["suppliers"] = []
                        
                        self.material_data[material_id]["suppliers"].append({
                            "supplier_id": supplier_id,
                            "supplier_name": supplier_item.get("supplier_name", "Unknown"),
                            "is_primary": supplier_item.get("is_primary", False),
                            "lead_time": supplier_item.get("lead_time", 0),
                            "price_per_unit": supplier_item.get("price_per_unit", 0)
                        })
            
            logger.info(f"Loaded data for {len(self.material_data)} materials")
            
        except Exception as e:
            logger.error(f"Error loading material data: {str(e)}")
            # Continue with partial data
    
    async def calculate_scenario_impact(
        self,
        scenario_id: str,
        inventory_buffer_days: int = 14,
        use_alternative_suppliers: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate the business impact of a disruption scenario.
        
        Args:
            scenario_id: ID of the disruption scenario to analyze
            inventory_buffer_days: Number of days inventory can buffer disruption
            use_alternative_suppliers: Whether to consider alternative suppliers
            
        Returns:
            Dictionary with impact analysis results
        """
        try:
            # Ensure cascade analyzer is initialized
            if not self.cascade_analyzer:
                await self.initialize()
            
            # Get scenario data
            scenario = self.cascade_analyzer.risk_scenarios.get(scenario_id)
            if not scenario:
                raise ValueError(f"Scenario {scenario_id} not found")
            
            # Get disrupted suppliers from scenario
            disrupted_suppliers = set()
            time_series = scenario.get("results", {}).get("time_series", [])
            
            for day_state in time_series:
                for supplier in day_state.get("disrupted_suppliers", []):
                    if supplier.get("disruption", 0) > 0.3:  # Significant disruption threshold
                        disrupted_suppliers.add(supplier.get("id"))
            
            # Build impact analysis
            impact = await self._calculate_impact(
                list(disrupted_suppliers),
                time_series,
                inventory_buffer_days,
                use_alternative_suppliers
            )
            
            # Add scenario information
            impact["scenario_id"] = scenario_id
            impact["scenario_details"] = {
                "initially_disrupted": scenario.get("disrupted_suppliers", []),
                "disruption_duration": scenario.get("disruption_duration", 0),
                "disruption_severity": scenario.get("disruption_severity", 0),
                "recovery_rate": scenario.get("recovery_rate", 0)
            }
            
            logger.info(f"Calculated impact for scenario {scenario_id}")
            
            return impact
            
        except Exception as e:
            logger.error(f"Error calculating scenario impact: {str(e)}")
            raise
    
    async def _calculate_impact(
        self,
        disrupted_suppliers: List[str],
        time_series: List[Dict[str, Any]],
        inventory_buffer_days: int = 14,
        use_alternative_suppliers: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate business impact of disruption.
        
        Args:
            disrupted_suppliers: List of disrupted supplier IDs
            time_series: Simulation time series data
            inventory_buffer_days: Inventory buffer in days
            use_alternative_suppliers: Whether to use alternative suppliers
            
        Returns:
            Dictionary with impact analysis
        """
        # Identify affected materials
        affected_materials = {}
        for material_id, material in self.material_data.items():
            suppliers = material.get("suppliers", [])
            
            # Check if any primary suppliers are disrupted
            primary_disrupted = False
            alternative_available = False
            
            for supplier in suppliers:
                supplier_id = supplier.get("supplier_id")
                is_primary = supplier.get("is_primary", False)
                
                if supplier_id in disrupted_suppliers and is_primary:
                    primary_disrupted = True
                
                if supplier_id not in disrupted_suppliers and not is_primary:
                    alternative_available = True
            
            # Mark material as affected if primary supplier is disrupted
            # and no alternative is available (or alternatives are not considered)
            if primary_disrupted and (not alternative_available or not use_alternative_suppliers):
                affected_materials[material_id] = material
        
        # Identify affected products
        affected_products = {}
        for product_id, product in self.product_data.items():
            materials = product.get("materials", [])
            
            # Check if any required materials are affected
            for material_item in materials:
                material_id = material_item.get("material_id")
                if material_id in affected_materials:
                    if product_id not in affected_products:
                        affected_products[product_id] = {
                            **product,
                            "affected_materials": []
                        }
                    
                    affected_products[product_id]["affected_materials"].append({
                        **material_item,
                        "material_data": affected_materials[material_id]
                    })
        
        # Calculate daily impact
        daily_impact = []
        max_simulation_days = len(time_series)
        
        for day in range(max_simulation_days):
            day_impact = {
                "day": day,
                "stock_outs": [],
                "revenue_impact": 0.0,
                "margin_impact": 0.0,
                "service_level_impact": 0.0
            }
            
            # Process each affected product
            total_sales = 0
            fulfilled_sales = 0
            
            for product_id, product in affected_products.items():
                current_stock = product.get("current_stock", 0)
                avg_daily_sales = product.get("avg_daily_sales", 0)
                contribution_margin = product.get("contribution_margin", 0)
                
                # Calculate remaining stock based on inventory and days passed
                remaining_stock = max(0, current_stock - (avg_daily_sales * max(0, day - inventory_buffer_days)))
                
                # Calculate daily demand and fulfillment
                daily_demand = avg_daily_sales
                total_sales += daily_demand
                
                if remaining_stock >= daily_demand:
                    # Can fulfill all demand
                    fulfilled_demand = daily_demand
                    fulfilled_sales += fulfilled_demand
                else:
                    # Partial or no fulfillment
                    fulfilled_demand = remaining_stock
                    fulfilled_sales += fulfilled_demand
                    
                    # Record stock out
                    if remaining_stock == 0:
                        day_impact["stock_outs"].append({
                            "product_id": product_id,
                            "product_name": product.get("name", "Unknown"),
                            "unfulfilled_demand": daily_demand - fulfilled_demand,
                            "revenue_impact": (daily_demand - fulfilled_demand) * product.get("sale_price", 0),
                            "margin_impact": (daily_demand - fulfilled_demand) * contribution_margin
                        })
            
            # Calculate daily financial impact
            day_impact["revenue_impact"] = sum(item["revenue_impact"] for item in day_impact["stock_outs"])
            day_impact["margin_impact"] = sum(item["margin_impact"] for item in day_impact["stock_outs"])
            
            # Calculate service level impact
            if total_sales > 0:
                day_impact["service_level_impact"] = 1.0 - (fulfilled_sales / total_sales)
            
            daily_impact.append(day_impact)
        
        # Calculate summary metrics
        total_revenue_impact = sum(day["revenue_impact"] for day in daily_impact)
        total_margin_impact = sum(day["margin_impact"] for day in daily_impact)
        avg_service_level_impact = sum(day["service_level_impact"] for day in daily_impact) / len(daily_impact) if daily_impact else 0
        
        stock_out_products = set()
        for day in daily_impact:
            for stock_out in day["stock_outs"]:
                stock_out_products.add(stock_out["product_id"])
        
        # Compile impact analysis
        impact_analysis = {
            "summary": {
                "affected_materials_count": len(affected_materials),
                "affected_products_count": len(affected_products),
                "stock_out_products_count": len(stock_out_products),
                "total_revenue_impact": total_revenue_impact,
                "total_margin_impact": total_margin_impact,
                "avg_service_level_impact": avg_service_level_impact
            },
            "affected_materials": [
                {
                    "id": material_id,
                    "name": material.get("name", "Unknown"),
                    "current_stock": material.get("current_stock", 0),
                    "lead_time": material.get("lead_time", 0)
                }
                for material_id, material in affected_materials.items()
            ],
            "affected_products": [
                {
                    "id": product_id,
                    "name": product.get("name", "Unknown"),
                    "current_stock": product.get("current_stock", 0),
                    "avg_daily_sales": product.get("avg_daily_sales", 0),
                    "buffer_days": math.floor(product.get("current_stock", 0) / max(1, product.get("avg_daily_sales", 1))),
                    "revenue_impact": sum(
                        day["revenue_impact"] 
                        for day in daily_impact 
                        for stock_out in day["stock_outs"] 
                        if stock_out["product_id"] == product_id
                    )
                }
                for product_id, product in affected_products.items()
            ],
            "daily_impact": daily_impact,
            "mitigation_recommendations": await self._generate_mitigation_recommendations(
                affected_materials,
                affected_products,
                use_alternative_suppliers
            )
        }
        
        return impact_analysis
    
    async def _generate_mitigation_recommendations(
        self,
        affected_materials: Dict[str, Any],
        affected_products: Dict[str, Any],
        use_alternative_suppliers: bool
    ) -> List[Dict[str, Any]]:
        """
        Generate mitigation recommendations based on impact analysis.
        
        Args:
            affected_materials: Dictionary of affected materials
            affected_products: Dictionary of affected products
            use_alternative_suppliers: Whether alternative suppliers are considered
            
        Returns:
            List of mitigation recommendations
        """
        recommendations = []
        
        # Prioritize products by impact
        prioritized_products = sorted(
            affected_products.values(),
            key=lambda p: p.get("avg_daily_sales", 0) * p.get("contribution_margin", 0),
            reverse=True
        )
        
        # Identify critical materials
        critical_materials = {}
        for product in prioritized_products:
            for material_item in product.get("affected_materials", []):
                material_id = material_item.get("material_id")
                if material_id not in critical_materials:
                    critical_materials[material_id] = []
                
                critical_materials[material_id].append({
                    "product_id": product.get("id"),
                    "product_name": product.get("name"),
                    "quantity_required": material_item.get("quantity_required", 0),
                    "daily_usage": material_item.get("quantity_required", 0) * product.get("avg_daily_sales", 0)
                })
        
        # Generate product-level recommendations
        for product in prioritized_products[:5]:  # Focus on top 5 products
            product_id = product.get("id")
            
            # Add prioritization recommendation
            recommendations.append({
                "type": "product_prioritization",
                "priority": "high",
                "product_id": product_id,
                "product_name": product.get("name", "Unknown"),
                "description": f"Prioritize production and allocation of {product.get('name', 'Unknown')} due to high financial impact",
                "potential_benefit": product.get("avg_daily_sales", 0) * product.get("contribution_margin", 0)
            })
            
            # Add inventory adjustment recommendation if stock is low
            buffer_days = math.floor(product.get("current_stock", 0) / max(1, product.get("avg_daily_sales", 1)))
            if buffer_days < 30:  # Less than 30 days of inventory
                recommendations.append({
                    "type": "inventory_adjustment",
                    "priority": "medium" if buffer_days > 14 else "high",
                    "product_id": product_id,
                    "product_name": product.get("name", "Unknown"),
                    "description": f"Increase safety stock for {product.get('name', 'Unknown')} to provide additional buffer",
                    "current_days": buffer_days,
                    "recommended_days": 30,
                    "additional_units": (30 - buffer_days) * product.get("avg_daily_sales", 0)
                })
        
        # Generate material-level recommendations
        for material_id, material in affected_materials.items():
            material_name = material.get("name", "Unknown")
            
            # Check for alternative suppliers
            suppliers = material.get("suppliers", [])
            alternative_suppliers = [s for s in suppliers if not s.get("is_primary", False)]
            
            if alternative_suppliers and not use_alternative_suppliers:
                # Recommend using alternative suppliers
                for alt_supplier in alternative_suppliers[:2]:  # Top 2 alternatives
                    recommendations.append({
                        "type": "alternative_supplier",
                        "priority": "high",
                        "material_id": material_id,
                        "material_name": material_name,
                        "supplier_id": alt_supplier.get("supplier_id"),
                        "supplier_name": alt_supplier.get("supplier_name", "Unknown"),
                        "description": f"Activate alternative supplier {alt_supplier.get('supplier_name', 'Unknown')} for {material_name}",
                        "lead_time": alt_supplier.get("lead_time", 0),
                        "price_difference_percent": ((alt_supplier.get("price_per_unit", 0) / 
                                                     max(1, next((s.get("price_per_unit", 0) for s in suppliers if s.get("is_primary", False)), 1))) - 1) * 100
                    })
            elif not alternative_suppliers:
                # Recommend finding new suppliers
                recommendations.append({
                    "type": "new_supplier",
                    "priority": "high",
                    "material_id": material_id,
                    "material_name": material_name,
                    "description": f"Initiate qualification of new suppliers for {material_name} to reduce single-source risk",
                    "affected_products_count": len(critical_materials.get(material_id, []))
                })
            
            # Recommend expedited shipping for critical materials
            if material.get("current_stock", 0) < material.get("safety_stock", 0):
                recommendations.append({
                    "type": "expedited_shipping",
                    "priority": "high",
                    "material_id": material_id,
                    "material_name": material_name,
                    "description": f"Arrange expedited shipping for {material_name} to replenish critical stock levels",
                    "current_stock": material.get("current_stock", 0),
                    "safety_stock": material.get("safety_stock", 0),
                    "standard_lead_time": material.get("lead_time", 0),
                    "expedited_lead_time": max(1, int(material.get("lead_time", 0) * 0.6))  # Estimate 40% reduction
                })
        
        # Add general recommendations
        recommendations.append({
            "type": "communication",
            "priority": "medium",
            "description": "Establish regular communication with affected suppliers to monitor recovery progress",
            "affected_suppliers_count": len(set(
                material.get("supplier_id") 
                for material in affected_materials.values() 
                for supplier in material.get("suppliers", []) 
                if supplier.get("is_primary", False)
            ))
        })
        
        recommendations.append({
            "type": "customer_management",
            "priority": "medium",
            "description": "Implement allocation strategy for affected products to prioritize key customers",
            "affected_products_count": len(affected_products)
        })
        
        # Sort recommendations by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 3))
        
        return recommendations
    
    async def calculate_financial_impact(
        self,
        scenario_id: str,
        include_indirect_costs: bool = True,
        cost_of_capital: float = 0.10  # 10% annual rate
    ) -> Dict[str, Any]:
        """
        Calculate detailed financial impact of a disruption scenario.
        
        Args:
            scenario_id: ID of the disruption scenario
            include_indirect_costs: Whether to include indirect costs
            cost_of_capital: Annual cost of capital for NPV calculations
            
        Returns:
            Dictionary with financial impact analysis
        """
        try:
            # Get basic impact analysis
            impact = await self.calculate_scenario_impact(scenario_id)
            
            # Extract key financial metrics
            total_revenue_impact = impact.get("summary", {}).get("total_revenue_impact", 0)
            total_margin_impact = impact.get("summary", {}).get("total_margin_impact", 0)
            
            # Calculate additional financial impacts
            additional_costs = {}
            
            # Calculate expedited shipping costs
            expedited_shipping_cost = 0
            for recommendation in impact.get("mitigation_recommendations", []):
                if recommendation.get("type") == "expedited_shipping":
                    # Estimate expedited shipping premium
                    material_id = recommendation.get("material_id")
                    if material_id in self.material_data:
                        material = self.material_data[material_id]
                        
                        # Estimate based on safety stock value
                        safety_stock = recommendation.get("safety_stock", 0)
                        price_per_unit = next(
                            (s.get("price_per_unit", 0) for s in material.get("suppliers", []) 
                             if s.get("is_primary", True)),
                            0
                        )
                        
                        # Typical premium for expedited shipping is 2-3x standard
                        expedited_shipping_cost += safety_stock * price_per_unit * 1.5  # 150% premium
            
            additional_costs["expedited_shipping"] = expedited_shipping_cost
            
            # Calculate alternative supplier premium
            alt_supplier_premium = 0
            for recommendation in impact.get("mitigation_recommendations", []):
                if recommendation.get("type") == "alternative_supplier":
                    # Get price difference percentage
                    price_diff_pct = recommendation.get("price_difference_percent", 0)
                    
                    # Estimate affected volume
                    material_id = recommendation.get("material_id")
                    if material_id in self.material_data:
                        # Estimate based on safety stock value
                        material = self.material_data[material_id]
                        safety_stock = material.get("safety_stock", 0)
                        price_per_unit = next(
                            (s.get("price_per_unit", 0) for s in material.get("suppliers", []) 
                             if s.get("is_primary", True)),
                            0
                        )
                        
                        alt_supplier_premium += safety_stock * price_per_unit * (price_diff_pct / 100)
            
            additional_costs["alternative_supplier_premium"] = alt_supplier_premium
            
            # Calculate inventory carrying costs for increased safety stock
            inventory_carrying_cost = 0
            for recommendation in impact.get("mitigation_recommendations", []):
                if recommendation.get("type") == "inventory_adjustment":
                    product_id = recommendation.get("product_id")
                    if product_id in self.product_data:
                        product = self.product_data[product_id]
                        
                        # Calculate additional inventory value
                        additional_units = recommendation.get("additional_units", 0)
                        unit_cost = product.get("cost", 0)
                        additional_inventory_value = additional_units * unit_cost
                        
                        # Annual inventory carrying cost is typically 20-30% of inventory value
                        carrying_cost_rate = 0.25  # 25%
                        
                        # Prorate for disruption duration
                        scenario = self.cascade_analyzer.risk_scenarios.get(scenario_id, {})
                        disruption_days = scenario.get("disruption_duration", 30)
                        
                        inventory_carrying_cost += additional_inventory_value * carrying_cost_rate * (disruption_days / 365)
            
            additional_costs["inventory_carrying_cost"] = inventory_carrying_cost
            
            # Include indirect costs if requested
            if include_indirect_costs:
                # Brand and reputation damage (estimated as % of revenue impact)
                reputation_impact = total_revenue_impact * 0.15  # 15% of revenue impact
                
                # Customer goodwill loss (estimated as future revenue impact)
                scenario = self.cascade_analyzer.risk_scenarios.get(scenario_id, {})
                time_series = scenario.get("results", {}).get("time_series", [])
                
                # Calculate average service level impact
                avg_service_level_impact = 0
                if time_series:
                    service_impacts = [
                        day.get("service_level_impact", 0) for day in impact.get("daily_impact", [])
                    ]
                    if service_impacts:
                        avg_service_level_impact = sum(service_impacts) / len(service_impacts)
                
                # Customer goodwill impact proportional to service level impact
                # Assume 5% of annual revenue is at risk for severe service issues
                # Get annual revenue from daily sales data
                annual_revenue = sum(
                    product.get("avg_daily_sales", 0) * product.get("sale_price", 0) * 365
                    for product in self.product_data.values()
                )
                
                goodwill_impact = annual_revenue * 0.05 * avg_service_level_impact
                
                # Add to additional costs
                additional_costs["reputation_impact"] = reputation_impact
                additional_costs["customer_goodwill_impact"] = goodwill_impact
            
            # Calculate NPV of total financial impact
            total_direct_costs = total_margin_impact + sum(additional_costs.values())
            
            # Discount rate per day (from annual rate)
            daily_rate = cost_of_capital / 365
            
            # Calculate NPV of daily impacts
            npv = 0
            for i, day_impact in enumerate(impact.get("daily_impact", [])):
                day_value = day_impact.get("margin_impact", 0)
                
                # Add prorated additional costs
                day_additional_costs = sum(additional_costs.values()) / len(impact.get("daily_impact", []))
                day_value += day_additional_costs
                
                # Discount to present value
                npv += day_value / ((1 + daily_rate) ** i)
            
            # Compile financial impact analysis
            financial_impact = {
                "summary": {
                    "total_revenue_impact": total_revenue_impact,
                    "total_margin_impact": total_margin_impact,
                    "total_additional_costs": sum(additional_costs.values()),
                    "total_direct_financial_impact": total_direct_costs,
                    "present_value_impact": npv
                },
                "additional_costs": additional_costs,
                "daily_financial_impact": [
                    {
                        "day": day.get("day"),
                        "margin_impact": day.get("margin_impact", 0),
                        "additional_costs": sum(additional_costs.values()) / len(impact.get("daily_impact", [])),
                        "cumulative_impact": sum(
                            d.get("margin_impact", 0) for d in impact.get("daily_impact", [])
                            if d.get("day") <= day.get("day")
                        ) + (sum(additional_costs.values()) * (day.get("day") + 1) / len(impact.get("daily_impact", [])))
                    }
                    for day in impact.get("daily_impact", [])
                ]
            }
            
            # Add indirect costs if included
            if include_indirect_costs:
                financial_impact["summary"]["total_indirect_costs"] = additional_costs.get("reputation_impact", 0) + additional_costs.get("customer_goodwill_impact", 0)
                financial_impact["summary"]["total_financial_impact"] = financial_impact["summary"]["total_direct_financial_impact"] + financial_impact["summary"]["total_indirect_costs"]
            
            logger.info(f"Calculated financial impact for scenario {scenario_id}")
            
            return financial_impact
            
        except Exception as e:
            logger.error(f"Error calculating financial impact: {str(e)}")
            raise
    
    def _find_table_by_domain(self, domain_concept: str) -> Optional[str]:
        """
        Find the table name corresponding to a domain concept.
        
        Args:
            domain_concept: Domain concept to look for
            
        Returns:
            Table name or None if not found
        """
        if not self.domain_mappings:
            return None
        
        # Look for exact match in domain mappings
        for mapping in self.domain_mappings:
            if mapping.get("domain_concept") == domain_concept and mapping.get("custom_column") is None:
                return mapping.get("custom_table")
        
        # Fall back to table names in schema that might match
        if self.schema and hasattr(self.schema, "tables"):
            for table in self.schema.tables:
                table_name = table.get("name", "").lower()
                if domain_concept in table_name:
                    return table.get("name")
        
        return None
    
    async def compare_mitigation_strategies(
        self,
        scenario_id: str,
        strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare different mitigation strategies for a disruption scenario.
        
        Args:
            scenario_id: ID of the disruption scenario
            strategies: List of strategy configurations to compare
            
        Returns:
            Comparative analysis of different strategies
        """
        try:
            # Validate scenario
            if scenario_id not in self.cascade_analyzer.risk_scenarios:
                raise ValueError(f"Scenario {scenario_id} not found")
            
            # Calculate impact for each strategy
            strategy_results = []
            
            for strategy in strategies:
                strategy_name = strategy.get("name", "Unnamed Strategy")
                inventory_buffer_days = strategy.get("inventory_buffer_days", 14)
                use_alternative_suppliers = strategy.get("use_alternative_suppliers", True)
                
                # Calculate impact with this strategy
                impact = await self.calculate_scenario_impact(
                    scenario_id,
                    inventory_buffer_days=inventory_buffer_days,
                    use_alternative_suppliers=use_alternative_suppliers
                )
                
                # Calculate financial impact
                financial_impact = await self.calculate_financial_impact(
                    scenario_id,
                    include_indirect_costs=strategy.get("include_indirect_costs", True)
                )
                
                # Store strategy results
                strategy_results.append({
                    "strategy": strategy_name,
                    "configuration": strategy,
                    "impact_summary": impact.get("summary", {}),
                    "financial_summary": financial_impact.get("summary", {})
                })
            
            # Compile comparison results
            comparison = {
                "scenario_id": scenario_id,
                "strategies": [s.get("strategy") for s in strategy_results],
                "financial_comparison": {},
                "operational_comparison": {},
                "recommendations": []
            }
            
            # Compare financial metrics
            financial_metrics = ["total_financial_impact", "present_value_impact", "total_additional_costs"]
            for metric in financial_metrics:
                comparison["financial_comparison"][metric] = []
                for strategy in strategy_results:
                    comparison["financial_comparison"][metric].append({
                        "strategy": strategy.get("strategy"),
                        "value": strategy.get("financial_summary", {}).get(metric, 0)
                    })
                
                # Sort by metric value (ascending is better for costs)
                comparison["financial_comparison"][metric].sort(key=lambda x: x["value"])
            
            # Compare operational metrics
            operational_metrics = ["affected_products_count", "stock_out_products_count", "avg_service_level_impact"]
            for metric in operational_metrics:
                comparison["operational_comparison"][metric] = []
                for strategy in strategy_results:
                    comparison["operational_comparison"][metric].append({
                        "strategy": strategy.get("strategy"),
                        "value": strategy.get("impact_summary", {}).get(metric, 0)
                    })
                
                # Sort by metric value (ascending is better)
                comparison["operational_comparison"][metric].sort(key=lambda x: x["value"])
            
            # Generate strategy recommendations
            # Find best strategy for each key metric
            best_financial_strategy = comparison["financial_comparison"]["total_financial_impact"][0]["strategy"]
            best_service_strategy = comparison["operational_comparison"]["avg_service_level_impact"][0]["strategy"]
            
            comparison["recommendations"].append({
                "metric": "cost_effectiveness",
                "best_strategy": best_financial_strategy,
                "description": f"The {best_financial_strategy} strategy offers the most cost-effective approach to managing disruption impact"
            })
            
            comparison["recommendations"].append({
                "metric": "service_level",
                "best_strategy": best_service_strategy,
                "description": f"The {best_service_strategy} strategy offers the best customer service level during disruption"
            })
            
            # Overall recommendation based on weighted score
            # Calculate weighted score for each strategy (lower is better)
            weighted_scores = []
            for strategy in strategy_results:
                strategy_name = strategy.get("strategy")
                
                # Get normalized metric values (0-1 scale, lower is better)
                financial_impact = strategy.get("financial_summary", {}).get("total_financial_impact", 0)
                max_financial = max(s.get("financial_summary", {}).get("total_financial_impact", 0) 
                                   for s in strategy_results)
                norm_financial = financial_impact / max_financial if max_financial > 0 else 0
                
                service_impact = strategy.get("impact_summary", {}).get("avg_service_level_impact", 0)
                max_service = max(s.get("impact_summary", {}).get("avg_service_level_impact", 0) 
                                 for s in strategy_results)
                norm_service = service_impact / max_service if max_service > 0 else 0
                
                # Calculate weighted score
                weighted_score = (norm_financial * 0.6) + (norm_service * 0.4)
                
                weighted_scores.append({
                    "strategy": strategy_name,
                    "weighted_score": weighted_score
                })
            
            # Sort by weighted score (ascending)
            weighted_scores.sort(key=lambda x: x["weighted_score"])
            
            # Best overall strategy
            best_overall = weighted_scores[0]["strategy"]
            
            comparison["recommendations"].append({
                "metric": "overall_effectiveness",
                "best_strategy": best_overall,
                "description": f"The {best_overall} strategy provides the best balance of cost effectiveness and service level maintenance"
            })
            
            logger.info(f"Compared {len(strategies)} mitigation strategies for scenario {scenario_id}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing mitigation strategies: {str(e)}")
            raise
    
    async def export_impact_analysis(
        self,
        scenario_id: str,
        output_format: str = "json"
    ) -> Any:
        """
        Export impact analysis in the specified format.
        
        Args:
            scenario_id: ID of the disruption scenario
            output_format: Format to export (json, csv, etc.)
            
        Returns:
            Exported impact analysis in the specified format
        """
        try:
            # Get impact analysis
            impact = await self.calculate_scenario_impact(scenario_id)
            
            # Get financial impact
            financial_impact = await self.calculate_financial_impact(scenario_id)
            
            # Combine for export
            export_data = {
                "scenario_id": scenario_id,
                "scenario_details": impact.get("scenario_details", {}),
                "impact_summary": impact.get("summary", {}),
                "financial_summary": financial_impact.get("summary", {}),
                "affected_materials": impact.get("affected_materials", []),
                "affected_products": impact.get("affected_products", []),
                "mitigation_recommendations": impact.get("mitigation_recommendations", [])
            }
            
            # Export in requested format
            if output_format == "json":
                return json.dumps(export_data, indent=2)
            elif output_format == "csv":
                # Create DataFrames for different components
                summary_df = pd.DataFrame([{**impact.get("summary", {}), **financial_impact.get("summary", {})}])
                materials_df = pd.DataFrame(impact.get("affected_materials", []))
                products_df = pd.DataFrame(impact.get("affected_products", []))
                recommendations_df = pd.DataFrame(impact.get("mitigation_recommendations", []))
                
                # Return a dict with all DataFrames
                return {
                    "summary": summary_df.to_csv(index=False),
                    "affected_materials": materials_df.to_csv(index=False),
                    "affected_products": products_df.to_csv(index=False),
                    "recommendations": recommendations_df.to_csv(index=False)
                }
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Error exporting impact analysis: {str(e)}")
            raise