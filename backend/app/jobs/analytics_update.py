# -*- coding: utf-8 -*-
# jobs/analytics_update.py

"""
Analytics calculation and update jobs
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional
from decimal import Decimal
import numpy as np
import pandas as pd
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, case
from sqlalchemy.sql import text

from app.db.database import get_db_session
from app.config import get_settings

from app.models import (


    Supplier, Product, Inventory, Order, OrderItem,
    Shipment, ShipmentItem, SupplierPerformanceMetric,
    InventoryMetric, DeliveryPerformance, ABCAnalysisResult,
    ForecastResult, SafetyStockCalculation, RiskAssessment,
    ComplianceCheck, ScheduledAnalytic, AnalyticsResult,
    BottleneckAnalysis, NetworkNode, NetworkEdge
)

# Get settings
# Lazy load settings to avoid circular import
_settings = None

def get_settings_cached():
    global _settings
    if _settings is None:
        from ..config import get_settings
        _settings = get_settings()
    return _settings

settings = property(lambda self: get_settings_cached())

# Get settings


logger = logging.getLogger(__name__)


class AnalyticsCalculator:
    """Calculate various analytics metrics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_supplier_performance(
        self,
        supplier_id: str,
        start_date: date,
        end_date: date
    ) -> SupplierPerformanceMetric:
        """Calculate supplier performance metrics"""
        
        # Get orders for supplier in period
        orders = self.db.query(Order).filter(
            and_(
                Order.supplier_id == supplier_id,
                Order.order_date >= start_date,
                Order.order_date <= end_date,
                Order.status.in_(["delivered", "closed"])
            )
        ).all()
        
        if not orders:
            return None
        
        # Calculate metrics
        total_orders = len(orders)
        on_time_deliveries = 0
        total_spend = Decimal("0")
        quality_issues = 0
        lead_times = []
        
        for order in orders:
            # Check on-time delivery
            if order.actual_delivery_date and order.requested_delivery_date:
                if order.actual_delivery_date <= order.requested_delivery_date:
                    on_time_deliveries += 1
                
                lead_time = (order.actual_delivery_date - order.order_date).days
                lead_times.append(lead_time)
            
            # Calculate spend
            total_spend += order.total_amount
            
            # Check for quality issues (simplified - would need quality data)
            # For now, check if any items were returned
            if order.status == "returned":
                quality_issues += 1
        
        # Calculate scores
        on_time_rate = (on_time_deliveries / total_orders) * 100 if total_orders > 0 else 0
        quality_score = ((total_orders - quality_issues) / total_orders) * 100 if total_orders > 0 else 100
        
        # Lead time variance
        if lead_times:
            avg_lead_time = np.mean(lead_times)
            lead_time_variance = np.std(lead_times)
        else:
            avg_lead_time = 0
            lead_time_variance = 0
        
        # Responsiveness score (simplified)
        responsiveness_score = min(100, 100 - (lead_time_variance * 2))
        
        # Price competitiveness (would need market data)
        price_competitiveness = 75.0  # Placeholder
        
        # Overall score (weighted average)
        overall_score = (
            on_time_rate * 0.3 +
            quality_score * 0.3 +
            responsiveness_score * 0.2 +
            price_competitiveness * 0.2
        )
        
        # Create or update metrics
        metrics = self.db.query(SupplierPerformanceMetric).filter(
            and_(
                SupplierPerformanceMetric.supplier_id == supplier_id,
                SupplierPerformanceMetric.period_start == start_date,
                SupplierPerformanceMetric.period_end == end_date
            )
        ).first()
        
        if not metrics:
            metrics = SupplierPerformanceMetric(
                supplier_id=supplier_id,
                period_start=start_date,
                period_end=end_date
            )
            self.db.add(metrics)
        
        metrics.on_time_delivery_rate = on_time_rate
        metrics.quality_score = quality_score
        metrics.responsiveness_score = responsiveness_score
        metrics.price_competitiveness = price_competitiveness
        metrics.overall_score = overall_score
        metrics.total_orders = total_orders
        metrics.total_spend = total_spend
        metrics.defect_rate = (quality_issues / total_orders) * 100 if total_orders > 0 else 0
        metrics.lead_time_variance = lead_time_variance
        
        self.db.commit()
        return metrics
    
    def calculate_inventory_metrics(
        self,
        location_id: str,
        product_id: Optional[str],
        start_date: date,
        end_date: date
    ) -> List[InventoryMetric]:
        """Calculate inventory metrics"""
        
        # Build query
        query = self.db.query(Inventory).filter(
            Inventory.location_id == location_id
        )
        
        if product_id:
            query = query.filter(Inventory.product_id == product_id)
        
        inventories = query.all()
        metrics_list = []
        
        for inventory in inventories:
            # Get inventory history (would need InventoryHistory table)
            # For now, use current values
            avg_inventory = float(inventory.quantity_on_hand)
            
            # Calculate turnover (would need sales data)
            # Simplified calculation
            order_items = self.db.query(
                func.sum(OrderItem.quantity_ordered)
            ).join(Order).filter(
                and_(
                    OrderItem.product_id == inventory.product_id,
                    Order.order_date >= start_date,
                    Order.order_date <= end_date,
                    Order.type == "sales"
                )
            ).scalar() or 0
            
            turnover_ratio = float(order_items) / avg_inventory if avg_inventory > 0 else 0
            
            # Calculate stockout days (simplified)
            stockout_days = 0
            if inventory.quantity_on_hand <= 0:
                stockout_days = (end_date - start_date).days
            
            # Calculate carrying cost (simplified)
            product = self.db.query(Product).filter(
                Product.id == inventory.product_id
            ).first()
            
            if product and product.unit_cost:
                carrying_cost = float(avg_inventory * product.unit_cost * Decimal("0.25"))  # 25% carrying cost
            else:
                carrying_cost = 0
            
            # Service level (simplified)
            service_level = 100 if stockout_days == 0 else max(0, 100 - (stockout_days * 10))
            
            # Create or update metrics
            metrics = self.db.query(InventoryMetric).filter(
                and_(
                    InventoryMetric.location_id == location_id,
                    InventoryMetric.product_id == inventory.product_id,
                    InventoryMetric.period_start == start_date,
                    InventoryMetric.period_end == end_date
                )
            ).first()
            
            if not metrics:
                metrics = InventoryMetric(
                    location_id=location_id,
                    product_id=inventory.product_id,
                    period_start=start_date,
                    period_end=end_date
                )
                self.db.add(metrics)
            
            metrics.average_inventory = avg_inventory
            metrics.turnover_ratio = turnover_ratio
            metrics.stockout_days = stockout_days
            metrics.carrying_cost = carrying_cost
            metrics.obsolescence_cost = 0  # Would need aging data
            metrics.service_level = service_level
            
            metrics_list.append(metrics)
        
        self.db.commit()
        return metrics_list
    
    def perform_abc_analysis(self, location_id: Optional[str] = None) -> List[ABCAnalysisResult]:
        """Perform ABC analysis on inventory"""
        
        # Get inventory with product details
        query = self.db.query(
            Inventory,
            Product
        ).join(Product, Inventory.product_id == Product.id)
        
        if location_id:
            query = query.filter(Inventory.location_id == location_id)
        
        inventory_data = query.all()
        
        # Calculate annual usage value
        usage_data = []
        
        for inventory, product in inventory_data:
            # Get annual usage (simplified - would need historical data)
            annual_usage = self.db.query(
                func.sum(OrderItem.quantity_ordered)
            ).join(Order).filter(
                and_(
                    OrderItem.product_id == product.id,
                    Order.order_date >= datetime.utcnow() - timedelta(days=365),
                    Order.type == "sales"
                )
            ).scalar() or 0
            
            if product.unit_cost:
                annual_value = float(annual_usage) * float(product.unit_cost)
            else:
                annual_value = 0
            
            usage_data.append({
                "product_id": product.id,
                "location_id": inventory.location_id,
                "annual_usage": float(annual_usage),
                "unit_cost": float(product.unit_cost) if product.unit_cost else 0,
                "annual_value": annual_value,
                "current_inventory": float(inventory.quantity_on_hand)
            })
        
        # Sort by annual value
        usage_data.sort(key=lambda x: x["annual_value"], reverse=True)
        
        # Calculate cumulative percentages
        total_value = sum(item["annual_value"] for item in usage_data)
        cumulative_value = 0
        
        results = []
        
        for i, item in enumerate(usage_data):
            cumulative_value += item["annual_value"]
            cumulative_percentage = (cumulative_value / total_value * 100) if total_value > 0 else 0
            
            # Determine class
            if cumulative_percentage <= 80:
                classification = "A"
            elif cumulative_percentage <= 95:
                classification = "B"
            else:
                classification = "C"
            
            # Create or update result
            result = self.db.query(ABCAnalysisResult).filter(
                and_(
                    ABCAnalysisResult.product_id == item["product_id"],
                    ABCAnalysisResult.location_id == item["location_id"]
                )
            ).first()
            
            if not result:
                result = ABCAnalysisResult(
                    product_id=item["product_id"],
                    location_id=item["location_id"]
                )
                self.db.add(result)
            
            result.classification = classification
            result.annual_usage_value = item["annual_value"]
            result.annual_usage_quantity = item["annual_usage"]
            result.percentage_of_total_value = (item["annual_value"] / total_value * 100) if total_value > 0 else 0
            result.cumulative_percentage = cumulative_percentage
            result.current_inventory_value = item["current_inventory"] * item["unit_cost"]
            result.analysis_date = datetime.utcnow()
            
            results.append(result)
        
        self.db.commit()
        return results
    
    def calculate_safety_stock(
        self,
        product_id: str,
        location_id: str,
        service_level: float = 0.95
    ) -> SafetyStockCalculation:
        """Calculate safety stock levels"""
        
        # Get historical demand data
        demand_data = self.db.query(
            func.date_trunc('day', Order.order_date).label('date'),
            func.sum(OrderItem.quantity_ordered).label('demand')
        ).join(OrderItem).filter(
            and_(
                OrderItem.product_id == product_id,
                Order.order_date >= datetime.utcnow() - timedelta(days=365),
                Order.type == "sales"
            )
        ).group_by(func.date_trunc('day', Order.order_date)).all()
        
        if not demand_data:
            return None
        
        # Convert to pandas for easier calculation
        df = pd.DataFrame(demand_data, columns=['date', 'demand'])
        df['demand'] = df['demand'].astype(float)
        
        # Calculate demand statistics
        avg_demand = df['demand'].mean()
        std_demand = df['demand'].std()
        
        # Get lead time
        supplier_product = self.db.query(Product).filter(
            Product.id == product_id
        ).first()
        
        lead_time = 7  # Default 7 days
        if supplier_product and supplier_product.suppliers:
            # Get average lead time from suppliers
            lead_times = [s.lead_time_days for s in supplier_product.suppliers if s.lead_time_days]
            if lead_times:
                lead_time = np.mean(lead_times)
        
        # Calculate lead time demand variability
        lead_time_demand_std = std_demand * np.sqrt(lead_time)
        
        # Get Z-score for service level
        from scipy import stats
        z_score = stats.norm.ppf(service_level)
        
        # Calculate safety stock
        safety_stock = z_score * lead_time_demand_std
        
        # Calculate reorder point
        reorder_point = (avg_demand * lead_time) + safety_stock
        
        # Create or update calculation
        calc = self.db.query(SafetyStockCalculation).filter(
            and_(
                SafetyStockCalculation.product_id == product_id,
                SafetyStockCalculation.location_id == location_id
            )
        ).first()
        
        if not calc:
            calc = SafetyStockCalculation(
                product_id=product_id,
                location_id=location_id
            )
            self.db.add(calc)
        
        calc.average_demand = avg_demand
        calc.demand_std_dev = std_demand
        calc.lead_time_days = lead_time
        calc.service_level = service_level
        calc.z_score = z_score
        calc.safety_stock_quantity = safety_stock
        calc.reorder_point = reorder_point
        calc.calculation_date = datetime.utcnow()
        calc.calculation_method = "statistical"
        
        self.db.commit()
        return calc
    def assess_supplier_risk(self, supplier_id: str) -> RiskAssessment:
        """Assess supplier risk"""
        
        supplier = self.db.query(Supplier).filter(
            Supplier.id == supplier_id
        ).first()
        
        if not supplier:
            return None
        
        risk_factors = {}
        
        # Financial risk (simplified - would need financial data)
        risk_factors["financial"] = 0.3
        
        # Geopolitical risk based on country
        high_risk_countries = ["XX", "YY", "ZZ"]  # Example
        if supplier.country in high_risk_countries:
            risk_factors["geopolitical"] = 0.8
        else:
            risk_factors["geopolitical"] = 0.2
        
        # Performance risk
        recent_metrics = self.db.query(SupplierPerformanceMetric).filter(
            SupplierPerformanceMetric.supplier_id == supplier_id
        ).order_by(SupplierPerformanceMetric.period_end.desc()).first()
        
        if recent_metrics:
            if recent_metrics.overall_score < 60:
                risk_factors["performance"] = 0.8
            elif recent_metrics.overall_score < 80:
                risk_factors["performance"] = 0.5
            else:
                risk_factors["performance"] = 0.2
        else:
            risk_factors["performance"] = 0.5
        
        # Dependency risk (single source)
        product_count = self.db.query(
            func.count(func.distinct(Product.id))
        ).filter(
            Product.suppliers.any(Supplier.id == supplier_id)
        ).scalar()
        
        if product_count > 10:
            risk_factors["dependency"] = 0.7
        elif product_count > 5:
            risk_factors["dependency"] = 0.5
        else:
            risk_factors["dependency"] = 0.3
        
        # Compliance risk
        recent_compliance = self.db.query(ComplianceCheck).filter(
            ComplianceCheck.supplier_id == supplier_id
        ).order_by(ComplianceCheck.check_date.desc()).first()
        
        if recent_compliance:
            if recent_compliance.status == "non_compliant":
                risk_factors["compliance"] = 0.9
            elif recent_compliance.status == "pending_review":
                risk_factors["compliance"] = 0.5
            else:
                risk_factors["compliance"] = 0.1
        else:
            risk_factors["compliance"] = 0.4
        
        # Calculate overall risk score
        risk_score = sum(risk_factors.values()) / len(risk_factors)
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Mitigation strategies
        mitigation_strategies = []
        
        if risk_factors["financial"] > 0.5:
            mitigation_strategies.append("Request financial statements and monitor credit ratings")
        
        if risk_factors["geopolitical"] > 0.5:
            mitigation_strategies.append("Identify alternative suppliers in stable regions")
        
        if risk_factors["performance"] > 0.5:
            mitigation_strategies.append("Implement supplier improvement program")
        
        if risk_factors["dependency"] > 0.5:
            mitigation_strategies.append("Diversify supplier base for critical components")
        
        if risk_factors["compliance"] > 0.5:
            mitigation_strategies.append("Conduct compliance audit and training")
        
        # Create or update assessment
        assessment = self.db.query(RiskAssessment).filter(
            RiskAssessment.supplier_id == supplier_id
        ).order_by(RiskAssessment.assessment_date.desc()).first()
        
        if not assessment or (datetime.utcnow() - assessment.assessment_date).days > 30:
            assessment = RiskAssessment(
                supplier_id=supplier_id,
                assessment_date=datetime.utcnow()
            )
            self.db.add(assessment)
        
        assessment.risk_level = risk_level
        assessment.risk_score = risk_score
        assessment.risk_factors = risk_factors
        assessment.mitigation_strategies = mitigation_strategies
        assessment.review_date = datetime.utcnow() + timedelta(days=90)
        
        self.db.commit()
        return assessment
    
    def identify_bottlenecks(self, network_id: str) -> List[BottleneckAnalysis]:
        """Identify bottlenecks in supply chain network"""
        
        # Get network nodes and edges
        nodes = self.db.query(NetworkNode).filter(
            NetworkNode.network_id == network_id
        ).all()
        
        edges = self.db.query(NetworkEdge).filter(
            NetworkEdge.network_id == network_id
        ).all()
        
        bottlenecks = []
        
        # Analyze each node
        for node in nodes:
            # Calculate input/output flow
            input_capacity = sum(
                edge.capacity for edge in edges
                if edge.target_node_id == node.id
            )
            
            output_capacity = sum(
                edge.capacity for edge in edges
                if edge.source_node_id == node.id
            )
            
            # Check if node is a bottleneck
            if input_capacity > output_capacity and output_capacity > 0:
                utilization = (input_capacity / output_capacity) * 100
                
                if utilization > 80:  # Bottleneck threshold
                    bottleneck = BottleneckAnalysis(
                        network_id=network_id,
                        node_id=node.id,
                        bottleneck_type="capacity",
                        severity="high" if utilization > 95 else "medium",
                        capacity_utilization=utilization,
                        throughput_limit=output_capacity,
                        impact_assessment={
                            "affected_downstream_nodes": len([e for e in edges if e.source_node_id == node.id]),
                            "potential_delay_days": int((utilization - 80) / 5),
                            "revenue_at_risk": 0  # Would need order data
                        },
                        recommendations=[
                            "Increase node processing capacity",
                            "Add parallel processing paths",
                            "Optimize node operations"
                        ],
                        identified_date=datetime.utcnow()
                    )
                    
                    self.db.add(bottleneck)
                    bottlenecks.append(bottleneck)
        
        self.db.commit()
        return bottlenecks


# Job functions
@scheduled_job(name="update_supplier_metrics", description="Update supplier performance metrics")
async def update_supplier_metrics():
    """Update supplier performance metrics for all active suppliers"""
    with get_db_session() as db:
        calculator = AnalyticsCalculator(db)
        
        # Get active suppliers
        suppliers = db.query(Supplier).filter(
            Supplier.status == "active"
        ).all()
        
        # Calculate last month's metrics
        end_date = date.today().replace(day=1) - timedelta(days=1)
        start_date = end_date.replace(day=1)
        
        # jobs/analytics_update.py (continued)
        
        results = []
        
        for supplier in suppliers:
            try:
                metrics = calculator.calculate_supplier_performance(
                    supplier_id=str(supplier.id),
                    start_date=start_date,
                    end_date=end_date
                )
                if metrics:
                    results.append({
                        "supplier_id": str(supplier.id),
                        "supplier_name": supplier.name,
                        "overall_score": metrics.overall_score
                    })
            except Exception as e:
                logger.error(f"Failed to calculate metrics for supplier {supplier.id}: {str(e)}")
        
        logger.info(f"Updated metrics for {len(results)} suppliers")
        return results


@scheduled_job(name="update_inventory_metrics", description="Update inventory metrics")
async def update_inventory_metrics():
    """Update inventory metrics for all locations"""
    with get_db_session() as db:
        calculator = AnalyticsCalculator(db)
        
        # Get unique locations
        locations = db.query(Inventory.location_id).distinct().all()
        
        # Calculate last month's metrics
        end_date = date.today().replace(day=1) - timedelta(days=1)
        start_date = end_date.replace(day=1)
        
        results = []
        
        for location in locations:
            try:
                metrics = calculator.calculate_inventory_metrics(
                    location_id=location[0],
                    product_id=None,  # All products
                    start_date=start_date,
                    end_date=end_date
                )
                results.extend(metrics)
            except Exception as e:
                logger.error(f"Failed to calculate metrics for location {location[0]}: {str(e)}")
        
        logger.info(f"Updated {len(results)} inventory metrics")
        return len(results)


@scheduled_job(name="update_delivery_performance", description="Update delivery performance metrics")
async def update_delivery_performance():
    """Update delivery performance metrics"""
    with get_db_session() as db:
        # Get completed shipments from last month
        end_date = date.today().replace(day=1) - timedelta(days=1)
        start_date = end_date.replace(day=1)
        
        shipments = db.query(Shipment).filter(
            and_(
                Shipment.actual_delivery_date >= start_date,
                Shipment.actual_delivery_date <= end_date,
                Shipment.status == "delivered"
            )
        ).all()
        
        metrics_by_carrier = defaultdict(lambda: {
            "total_shipments": 0,
            "on_time": 0,
            "late": 0,
            "damaged": 0,
            "total_cost": Decimal("0")
        })
        
        for shipment in shipments:
            carrier = shipment.carrier or "Unknown"
            metrics = metrics_by_carrier[carrier]
            
            metrics["total_shipments"] += 1
            metrics["total_cost"] += shipment.shipping_cost or Decimal("0")
            
            # Check on-time delivery
            if shipment.estimated_delivery_date and shipment.actual_delivery_date:
                if shipment.actual_delivery_date <= shipment.estimated_delivery_date:
                    metrics["on_time"] += 1
                else:
                    metrics["late"] += 1
        
        # Create performance records
        for carrier, metrics in metrics_by_carrier.items():
            on_time_rate = (metrics["on_time"] / metrics["total_shipments"] * 100) if metrics["total_shipments"] > 0 else 0
            
            perf = DeliveryPerformance(
                carrier=carrier,
                period_start=start_date,
                period_end=end_date,
                total_shipments=metrics["total_shipments"],
                on_time_deliveries=metrics["on_time"],
                late_deliveries=metrics["late"],
                damaged_deliveries=metrics["damaged"],
                on_time_rate=on_time_rate,
                average_transit_time=0,  # Would need to calculate
                average_cost=metrics["total_cost"] / metrics["total_shipments"] if metrics["total_shipments"] > 0 else 0)
            db.add(perf)
        
        db.commit()
        logger.info(f"Updated delivery performance for {len(metrics_by_carrier)} carriers")
        return len(metrics_by_carrier)


@scheduled_job(name="calculate_abc_analysis", description="Perform ABC analysis on inventory")
async def calculate_abc_analysis():
    """Perform ABC analysis for all locations"""
    with get_db_session() as db:
        calculator = AnalyticsCalculator(db)
        
        # Get unique locations
        locations = db.query(Inventory.location_id).distinct().all()
        
        all_results = []
        
        for location in locations:
            try:
                results = calculator.perform_abc_analysis(location_id=location[0])
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Failed to perform ABC analysis for location {location[0]}: {str(e)}")
        
        logger.info(f"Completed ABC analysis for {len(all_results)} products")
        return len(all_results)


@scheduled_job(name="update_risk_assessments", description="Update supplier risk assessments")
async def update_risk_assessments():
    """Update risk assessments for all suppliers"""
    with get_db_session() as db:
        calculator = AnalyticsCalculator(db)
        
        # Get suppliers needing assessment
        suppliers = db.query(Supplier).filter(
            Supplier.status == "active"
        ).all()
        
        results = []
        
        for supplier in suppliers:
            try:
                assessment = calculator.assess_supplier_risk(str(supplier.id))
                if assessment:
                    results.append({
                        "supplier_id": str(supplier.id),
                        "supplier_name": supplier.name,
                        "risk_level": assessment.risk_level,
                        "risk_score": assessment.risk_score
                    })
            except Exception as e:
                logger.error(f"Failed to assess risk for supplier {supplier.id}: {str(e)}")
        
        logger.info(f"Updated risk assessments for {len(results)} suppliers")
        return results


@scheduled_job(name="update_safety_stock", description="Update safety stock calculations")
async def update_safety_stock():
    """Update safety stock for all products"""
    with get_db_session() as db:
        calculator = AnalyticsCalculator(db)
        
        # Get active inventory items
        inventory_items = db.query(Inventory).filter(
            Inventory.quantity_on_hand > 0
        ).all()
        
        results = []
        
        for item in inventory_items:
            try:
                calc = calculator.calculate_safety_stock(
                    product_id=str(item.product_id),
                    location_id=item.location_id,
                    service_level=0.95
                )
                if calc:
                    results.append({
                        "product_id": str(item.product_id),
                        "location_id": item.location_id,
                        "safety_stock": calc.safety_stock_quantity,
                        "reorder_point": calc.reorder_point
                    })
            except Exception as e:
                logger.error(f"Failed to calculate safety stock for product {item.product_id}: {str(e)}")
        
        logger.info(f"Updated safety stock for {len(results)} items")
        return len(results)


@scheduled_job(name="identify_network_bottlenecks", description="Identify supply chain bottlenecks")
async def identify_network_bottlenecks():
    """Identify bottlenecks in all supply chain networks"""
    with get_db_session() as db:
        calculator = AnalyticsCalculator(db)
        
        # Get active networks
        from app.models import SupplyChainNetwork
        networks = db.query(SupplyChainNetwork).filter(
            SupplyChainNetwork.is_active == True
        ).all()
        
        all_bottlenecks = []
        
        for network in networks:
            try:
                bottlenecks = calculator.identify_bottlenecks(str(network.id))
                all_bottlenecks.extend(bottlenecks)
            except Exception as e:
                logger.error(f"Failed to identify bottlenecks for network {network.id}: {str(e)}")
        
        logger.info(f"Identified {len(all_bottlenecks)} bottlenecks")
        return len(all_bottlenecks)


@scheduled_job(name="run_all_analytics", description="Run all analytics updates")
async def run_all_analytics():
    """Run all analytics calculations"""
    results = {
        "supplier_metrics": await update_supplier_metrics(),
        "inventory_metrics": await update_inventory_metrics(),
        "delivery_performance": await update_delivery_performance(),
        "abc_analysis": await calculate_abc_analysis(),
        "risk_assessments": await update_risk_assessments(),
        "safety_stock": await update_safety_stock(),
        "bottlenecks": await identify_network_bottlenecks()
    }
    
    logger.info(f"Completed all analytics updates: {results}")
    return results


async def run_analytics_by_id(analytics_id: str):
    """Run a specific scheduled analytics job"""
    with get_db_session() as db:
        # Get scheduled analytics
        scheduled = db.query(ScheduledAnalytic).filter(
            ScheduledAnalytic.id == analytics_id
        ).first()
        
        if not scheduled:
            raise ValueError(f"Scheduled analytics {analytics_id} not found")
        
        # Run based on type
        analytics_type = scheduled.analytics_type
        parameters = scheduled.parameters or {}
        
        calculator = AnalyticsCalculator(db)
        
        if analytics_type == "supplier_performance":
            supplier_id = parameters.get("supplier_id")
            if supplier_id:
                result = calculator.calculate_supplier_performance(
                    supplier_id=supplier_id,
                    start_date=date.today() - timedelta(days=30),
                    end_date=date.today()
                )
            else:
                result = await update_supplier_metrics()
        
        elif analytics_type == "inventory_metrics":
            location_id = parameters.get("location_id")
            product_id = parameters.get("product_id")
            result = calculator.calculate_inventory_metrics(
                location_id=location_id,
                product_id=product_id,
                start_date=date.today() - timedelta(days=30),
                end_date=date.today()
            )
        
        elif analytics_type == "abc_analysis":
            location_id = parameters.get("location_id")
            result = calculator.perform_abc_analysis(location_id=location_id)
        
        elif analytics_type == "risk_assessment":
            supplier_id = parameters.get("supplier_id")
            if supplier_id:
                result = calculator.assess_supplier_risk(supplier_id)
            else:
                result = await update_risk_assessments()
        
        else:
            raise ValueError(f"Unknown analytics type: {analytics_type}")
        
        # Store result
        analytics_result = AnalyticsResult(
            analytics_type=analytics_type,
            parameters=parameters,
            result_data={"result": result},
            created_by_id=scheduled.created_by_id
        )
        db.add(analytics_result)
        
        # Update last run time
        scheduled.last_run_time = datetime.utcnow()
        scheduled.next_run_time = datetime.utcnow() + timedelta(
            seconds=scheduled.schedule_config.get("interval_seconds", 86400)
        )
        
        db.commit()
        
        return result


async def generate_scheduled_report(report_type: str, recipients: List[str]):
    """Generate and send scheduled report"""
    with get_db_session() as db:
        # Generate report based on type
        if report_type == "supplier_performance":
            # Get latest supplier metrics
            metrics = db.query(SupplierPerformanceMetric).filter(
                SupplierPerformanceMetric.period_end >= date.today() - timedelta(days=35)
            ).all()
            
            # Create report data
            report_data = {
                "report_type": "supplier_performance",
                "period": f"{metrics[0].period_start} to {metrics[0].period_end}" if metrics else "N/A",
                "summary": {
                    "total_suppliers": len(metrics),
                    "average_score": sum(m.overall_score for m in metrics) / len(metrics) if metrics else 0,
                    "high_performers": len([m for m in metrics if m.overall_score >= 90]),
                    "at_risk": len([m for m in metrics if m.overall_score < 60])
                },
                "details": [
                    {
                        "supplier_id": str(m.supplier_id),
                        "overall_score": m.overall_score,
                        "on_time_rate": m.on_time_delivery_rate,
                        "quality_score": m.quality_score
                    }
                    for m in sorted(metrics, key=lambda x: x.overall_score, reverse=True)[:10]
                ]
            }
        
        elif report_type == "inventory_status":
            # Get inventory metrics and ABC analysis
            abc_results = db.query(ABCAnalysisResult).all()
            
            report_data = {
                "report_type": "inventory_status",
                "generated_at": datetime.utcnow().isoformat(),
                "summary": {
                    "total_products": len(abc_results),
                    "a_items": len([r for r in abc_results if r.classification == "A"]),
                    "b_items": len([r for r in abc_results if r.classification == "B"]),
                    "c_items": len([r for r in abc_results if r.classification == "C"]),
                    "total_inventory_value": sum(r.current_inventory_value for r in abc_results)
                }
            }
        
        else:
            raise ValueError(f"Unknown report type: {report_type}")
        
        # Create report record
        from app.models import Report
        report = Report(
            report_type=report_type,
            name=f"{report_type.replace('_', ' ').title()} - {date.today()}",
            parameters={"recipients": recipients},
            status="completed",
            result_data=report_data,
            generated_at=datetime.utcnow(),
            created_by_id=None  # System generated
        )
        db.add(report)
        db.commit()
        
        # Send report to recipients (would integrate with email service)
        logger.info(f"Generated {report_type} report for {len(recipients)} recipients")
        
        return report

