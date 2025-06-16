"""
Inventory repository for inventory management operations
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from uuid import UUID
import logging

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, or_, and_, desc, case
from sqlalchemy.exc import IntegrityError

from app.models import (
    Inventory, Product, Material,
    InventoryHistory, ABCAnalysisResult, SafetyStockCalculation
)

logger = logging.getLogger(__name__)


class InventoryRepository:
    """Repository for inventory-related database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # =====================================================
    # Inventory CRUD Operations
    # =====================================================
    
    def create_inventory(
        self,
        location_code: str,
        quantity_on_hand: Decimal,
        product_id: Optional[UUID] = None,
        material_id: Optional[UUID] = None,
        warehouse_id: Optional[UUID] = None,
        reorder_point: Optional[Decimal] = None,
        reorder_quantity: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Inventory:
        """Create new inventory record"""
        if not product_id and not material_id:
            raise ValueError("Either product_id or material_id must be provided")
        
        if product_id and material_id:
            raise ValueError("Only one of product_id or material_id should be provided")
        
        inventory = Inventory(
            product_id=product_id,
            material_id=material_id,
            location_code=location_code,
            quantity_on_hand=quantity_on_hand,
            quantity_reserved=Decimal('0'),
            reorder_point=reorder_point,
            reorder_quantity=reorder_quantity,
            metadata=metadata or {},
            last_counted_date=date.today()
        )
        
        self.db.add(inventory)
        self.db.commit()
        self.db.refresh(inventory)
        
        # Create initial history record
        self._create_history(
            inventory_id=inventory.id,
            transaction_type='initial',
            quantity_change=quantity_on_hand,
            quantity_before=Decimal('0'),
            quantity_after=quantity_on_hand,
            reason="Initial inventory creation"
        )
        
        return inventory
    
    def get_inventory_by_id(self, inventory_id: UUID) -> Optional[Inventory]:
        """Get inventory by ID"""
        return self.db.query(Inventory).options(
            joinedload(Inventory.product),
            joinedload(Inventory.material)
        ).filter(Inventory.id == inventory_id).first()
    
    def get_inventory_by_item(
        self,
        location_code: str,
        product_id: Optional[UUID] = None,
        material_id: Optional[UUID] = None
    ) -> Optional[Inventory]:
        """Get inventory for specific item and location"""
        query = self.db.query(Inventory).filter(
            Inventory.location_code == location_code
        )
        
        if product_id:
            query = query.filter(Inventory.product_id == product_id)
        elif material_id:
            query = query.filter(Inventory.material_id == material_id)
        else:
            return None
        
        return query.first()
    
    def get_inventory_list(
        self,
        skip: int = 0,
        limit: int = 100,
        location_code: Optional[str] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
        low_stock_only: bool = False,
        include_zero_stock: bool = True
    ) -> Tuple[List[Inventory], int]:
        """Get inventory list with filters"""
        query = self.db.query(Inventory).options(
            joinedload(Inventory.product),
            joinedload(Inventory.material)
        )
        
        # Apply filters
        if location_code:
            query = query.filter(Inventory.location_code == location_code)
        
        if not include_zero_stock:
            query = query.filter(Inventory.quantity_on_hand > 0)
        
        if low_stock_only:
            query = query.filter(
                and_(
                    Inventory.reorder_point.isnot(None),
                    Inventory.quantity_available <= Inventory.reorder_point
                )
            )
        
        # Join with product/material for category and search filters
        if category or search:
            query = query.outerjoin(Product).outerjoin(Material)
            
            if category:
                query = query.filter(
                    or_(
                        Product.category == category,
                        Material.type == category
                    )
                )
            
            if search:
                search_filter = f"%{search}%"
                query = query.filter(
                    or_(
                        Product.name.ilike(search_filter),
                        Product.sku.ilike(search_filter),
                        Material.name.ilike(search_filter),
                        Material.code.ilike(search_filter)
                    )
                )
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        inventory_list = query.offset(skip).limit(limit).all()
        
        return inventory_list, total
    
    def update_inventory_quantity(
        self,
        inventory_id: UUID,
        new_quantity: Decimal,
        reason: str,
        performed_by: Optional[UUID] = None,
        reference_type: Optional[str] = None,
        reference_id: Optional[UUID] = None
    ) -> Optional[Inventory]:
        """Update inventory quantity (absolute)"""
        inventory = self.get_inventory_by_id(inventory_id)
        if not inventory:
            return None
        
        old_quantity = inventory.quantity_on_hand
        quantity_change = new_quantity - old_quantity
        
        # Update inventory
        inventory.quantity_on_hand = new_quantity
        inventory.last_movement_date = datetime.utcnow()
        inventory.updated_at = datetime.utcnow()
        
        # Create history record
        self._create_history(
            inventory_id=inventory_id,
            transaction_type='adjustment',
            quantity_change=quantity_change,
            quantity_before=old_quantity,
            quantity_after=new_quantity,
            reason=reason,
            performed_by=performed_by,
            reference_type=reference_type,
            reference_id=reference_id
        )
        
        self.db.commit()
        self.db.refresh(inventory)
        return inventory
    
    def adjust_inventory_quantity(
        self,
        inventory_id: UUID,
        quantity_change: Decimal,
        transaction_type: str,
        reason: str,
        performed_by: Optional[UUID] = None,
        reference_type: Optional[str] = None,
        reference_id: Optional[UUID] = None
    ) -> Optional[Inventory]:
        """Adjust inventory quantity (relative)"""
        inventory = self.get_inventory_by_id(inventory_id)
        if not inventory:
            return None
        
        old_quantity = inventory.quantity_on_hand
        new_quantity = old_quantity + quantity_change
        
        # Validate new quantity
        if new_quantity < 0:
            raise ValueError(f"Insufficient inventory. Available: {old_quantity}, Requested: {abs(quantity_change)}")
        
        # Update inventory
        inventory.quantity_on_hand = new_quantity
        inventory.last_movement_date = datetime.utcnow()
        inventory.updated_at = datetime.utcnow()
        
        # Create history record
        self._create_history(
            inventory_id=inventory_id,
            transaction_type=transaction_type,
            quantity_change=quantity_change,
            quantity_before=old_quantity,
            quantity_after=new_quantity,
            reason=reason,
            performed_by=performed_by,
            reference_type=reference_type,
            reference_id=reference_id
        )
        
        self.db.commit()
        self.db.refresh(inventory)
        return inventory
    
    def reserve_inventory(
        self,
        inventory_id: UUID,
        quantity_to_reserve: Decimal,
        reference_type: str,
        reference_id: UUID
    ) -> Optional[Inventory]:
        """Reserve inventory quantity"""
        inventory = self.get_inventory_by_id(inventory_id)
        if not inventory:
            return None
        
        # Check availability
        if inventory.quantity_available < quantity_to_reserve:
            raise ValueError(f"Insufficient available inventory. Available: {inventory.quantity_available}")
        
        # Update reserved quantity
        inventory.quantity_reserved += quantity_to_reserve
        inventory.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(inventory)
        return inventory
    
    def release_reservation(
        self,
        inventory_id: UUID,
        quantity_to_release: Decimal,
        reference_type: str,
        reference_id: UUID
    ) -> Optional[Inventory]:
        """Release reserved inventory"""
        inventory = self.get_inventory_by_id(inventory_id)
        if not inventory:
            return None
        
        # Validate release quantity
        if inventory.quantity_reserved < quantity_to_release:
            raise ValueError(f"Cannot release more than reserved. Reserved: {inventory.quantity_reserved}")
        
        # Update reserved quantity
        inventory.quantity_reserved -= quantity_to_release
        inventory.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(inventory)
        return inventory
    
    def update_reorder_parameters(
        self,
        inventory_id: UUID,
        reorder_point: Optional[Decimal] = None,
        reorder_quantity: Optional[Decimal] = None
    ) -> Optional[Inventory]:
        """Update reorder parameters"""
        inventory = self.get_inventory_by_id(inventory_id)
        if not inventory:
            return None
        
        if reorder_point is not None:
            inventory.reorder_point = reorder_point
        if reorder_quantity is not None:
            inventory.reorder_quantity = reorder_quantity
        
        inventory.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(inventory)
        return inventory
    
    # =====================================================
    # Inventory Analysis Operations
    # =====================================================
    
    def get_low_stock_items(
        self,
        location_code: Optional[str] = None,
        threshold_percentage: float = 20.0
    ) -> List[Dict[str, Any]]:
        """Get items below reorder point or threshold"""
        query = self.db.query(
            Inventory,
            case(
                (Inventory.product_id.isnot(None), Product.name),
                else_=Material.name
            ).label('item_name'),
            case(
                (Inventory.product_id.isnot(None), Product.sku),
                else_=Material.code
            ).label('item_code')
        ).outerjoin(
            Product, Inventory.product_id == Product.id
        ).outerjoin(
            Material, Inventory.material_id == Material.id
        )
        
        if location_code:
            query = query.filter(Inventory.location_code == location_code)
        
        # Filter for low stock
        query = query.filter(
            or_(
                # Below reorder point
                and_(
                    Inventory.reorder_point.isnot(None),
                    Inventory.quantity_available <= Inventory.reorder_point
                ),
                # Below threshold percentage of reorder quantity
                and_(
                    Inventory.reorder_quantity.isnot(None),
                    Inventory.quantity_available <= (Inventory.reorder_quantity * threshold_percentage / 100)
                )
            )
        )
        
        results = query.all()
        
        return [
            {
                "inventory_id": inv.id,
                "item_name": item_name,
                "item_code": item_code,
                "location_code": inv.location_code,
                "quantity_on_hand": float(inv.quantity_on_hand),
                "quantity_available": float(inv.quantity_available),
                "reorder_point": float(inv.reorder_point) if inv.reorder_point else None,
                "reorder_quantity": float(inv.reorder_quantity) if inv.reorder_quantity else None,
                "shortage": float(inv.reorder_point - inv.quantity_available) if inv.reorder_point else None
            }
            for inv, item_name, item_code in results
        ]
    
    def get_inventory_value(
        self,
        location_code: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate total inventory value"""
        query = self.db.query(
            func.sum(
                case(
                    (Inventory.product_id.isnot(None), 
                     Inventory.quantity_on_hand * Product.unit_cost),
                    else_=Inventory.quantity_on_hand * Material.unit_cost
                )
            ).label('total_value'),
            func.count(Inventory.id).label('item_count'),
            func.sum(Inventory.quantity_on_hand).label('total_quantity')
        ).outerjoin(
            Product, Inventory.product_id == Product.id
        ).outerjoin(
            Material, Inventory.material_id == Material.id
        )
        
        if location_code:
            query = query.filter(Inventory.location_code == location_code)
        
        if category:
            query = query.filter(
                or_(
                    Product.category == category,
                    Material.type == category
                )
            )
        
        result = query.first()
        
        return {
            "total_value": float(result.total_value or 0),
            "item_count": result.item_count or 0,
            "total_quantity": float(result.total_quantity or 0)
        }
    
    def get_inventory_turnover(
        self,
        start_date: date,
        end_date: date,
        location_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate inventory turnover ratio"""
        # Get average inventory value
        avg_inventory = self.db.query(
            func.avg(
                case(
                    (Inventory.product_id.isnot(None),
                     Inventory.quantity_on_hand * Product.unit_cost),
                    else_=Inventory.quantity_on_hand * Material.unit_cost
                )
            )
        ).outerjoin(
            Product, Inventory.product_id == Product.id
        ).outerjoin(
            Material, Inventory.material_id == Material.id
        )
        
        if location_code:
            avg_inventory = avg_inventory.filter(Inventory.location_code == location_code)
        
        avg_value = avg_inventory.scalar() or 0
        
        # Get cost of goods sold (approximated from inventory movements)
        cogs_query = self.db.query(
            func.sum(
                case(
                    (InventoryHistory.quantity_change < 0,
                     func.abs(InventoryHistory.quantity_change) * 
                     case(
                         (Inventory.product_id.isnot(None), Product.unit_cost),
                         else_=Material.unit_cost
                     )),
                    else_=0
                )
            )
        ).join(
            Inventory, InventoryHistory.inventory_id == Inventory.id
        ).outerjoin(
            Product, Inventory.product_id == Product.id
        ).outerjoin(
            Material, Inventory.material_id == Material.id
        ).filter(
            InventoryHistory.recorded_at.between(start_date, end_date),
            InventoryHistory.transaction_type.in_(['sale', 'consumption'])
        )
        
        if location_code:
            cogs_query = cogs_query.filter(Inventory.location_code == location_code)
        
        cogs = cogs_query.scalar() or 0
        
        # Calculate turnover ratio
        turnover_ratio = (cogs / avg_value) if avg_value > 0 else 0
        days_in_period = (end_date - start_date).days
        days_of_inventory = (days_in_period / turnover_ratio) if turnover_ratio > 0 else 0
        
        return {
            "turnover_ratio": float(turnover_ratio),
            "days_of_inventory": float(days_of_inventory),
            "average_inventory_value": float(avg_value),
            "cost_of_goods_sold": float(cogs)
        }
    
    # =====================================================
    # ABC Analysis Operations
    # =====================================================
    
    def perform_abc_analysis(
        self,
        analysis_date: date,
        location_code: Optional[str] = None,
        analysis_type: str = 'value'  # 'value' or 'quantity'
    ) -> List[ABCAnalysisResult]:
        """Perform ABC analysis on inventory"""
        # Get inventory items with values
        query = self.db.query(
            Inventory,
            case(
                (Inventory.product_id.isnot(None), Product.unit_cost),
                else_=Material.unit_cost
            ).label('unit_cost')
        ).outerjoin(
            Product, Inventory.product_id == Product.id
        ).outerjoin(
            Material, Inventory.material_id == Material.id
        )
        
        if location_code:
            query = query.filter(Inventory.location_code == location_code)
        
        items = query.all()
        
        # Calculate annual values/quantities
        item_data = []
        for inv, unit_cost in items:
            # Get annual movement (simplified - using current quantity * 12)
            annual_quantity = inv.quantity_on_hand * 12  # Simplified
            annual_value = annual_quantity * (unit_cost or 0)
            
            item_data.append({
                'inventory': inv,
                'annual_quantity': annual_quantity,
                'annual_value': annual_value,
                'sort_value': annual_value if analysis_type == 'value' else annual_quantity
            })
        
        # Sort by value/quantity descending
        item_data.sort(key=lambda x: x['sort_value'], reverse=True)
        
        # Calculate cumulative percentages and assign categories
        total = sum(item['sort_value'] for item in item_data)
        if total == 0:
            return []
        
        results = []
        cumulative = 0
        
        for item in item_data:
            percentage = (item['sort_value'] / total) * 100
            cumulative += percentage
            
            # Assign category
            if cumulative <= 80:
                category = 'A'
            elif cumulative <= 95:
                category = 'B'
            else:
                category = 'C'
            
            # Create result record
            inv = item['inventory']
            result = ABCAnalysisResult(
                analysis_date=analysis_date,
                analysis_type=analysis_type,
                item_type='product' if inv.product_id else 'material',
                item_id=inv.product_id or inv.material_id,
                category=category,
                annual_value=item['annual_value'],
                annual_quantity=item['annual_quantity'],
                percentage_of_total=percentage,
                cumulative_percentage=cumulative,
                recommendations=self._generate_abc_recommendations(category)
            )
            results.append(result)
            self.db.add(result)
        
        self.db.commit()
        return results
    
    def _generate_abc_recommendations(self, category: str) -> List[str]:
        """Generate recommendations based on ABC category"""
        recommendations = {
            'A': [
                "Monitor closely with daily/weekly reviews",
                "Maintain higher service levels (95-99%)",
                "Consider vendor-managed inventory",
                "Implement automatic reordering"
            ],
            'B': [
                "Review bi-weekly or monthly",
                "Maintain moderate service levels (85-95%)",
                "Use economic order quantity (EOQ) models",
                "Consider safety stock optimization"
            ],
            'C': [
                "Review quarterly",
                "Maintain lower service levels (70-85%)",
                "Consider reducing variety",
                "Evaluate discontinuation of slow-moving items"
            ]
        }
        return recommendations.get(category, [])
    
    # =====================================================
    # Safety Stock Operations
    # =====================================================
    
    def calculate_safety_stock(
        self,
        inventory_id: UUID,
        service_level: float = 95.0,
        lead_time_days: int = 7,
        calculation_method: str = 'basic'
    ) -> Optional[SafetyStockCalculation]:
        """Calculate safety stock for an inventory item"""
        inventory = self.get_inventory_by_id(inventory_id)
        if not inventory:
            return None
        
        # Get historical demand data (simplified - using recent movements)
        demand_history = self.db.query(
            func.avg(func.abs(InventoryHistory.quantity_change)).label('avg_demand'),
            func.stddev(func.abs(InventoryHistory.quantity_change)).label('std_demand')
        ).filter(
            InventoryHistory.inventory_id == inventory_id,
            InventoryHistory.quantity_change < 0,
            InventoryHistory.recorded_at >= datetime.utcnow() - timedelta(days=90)
        ).first()
        
        avg_demand = float(demand_history.avg_demand or 0)
        std_demand = float(demand_history.std_demand or 0)
        
        # Z-score for service level
        z_scores = {
            90.0: 1.28,
            95.0: 1.65,
            97.5: 1.96,
            99.0: 2.33
        }
        z_score = z_scores.get(service_level, 1.65)
        
        # Calculate safety stock
        if calculation_method == 'basic':
            safety_stock = z_score * std_demand * (lead_time_days ** 0.5)
        else:
            # Advanced calculation could include lead time variability
            safety_stock = z_score * std_demand * (lead_time_days ** 0.5)
        
        # Create calculation record
        calculation = SafetyStockCalculation(
            calculation_date=date.today(),
            product_id=inventory.product_id,
            material_id=inventory.material_id,
            location_code=inventory.location_code,
            service_level=service_level,
            lead_time_days=lead_time_days,
            demand_mean=avg_demand,
            demand_std_dev=std_demand,
            calculated_safety_stock=safety_stock,
            current_stock=float(inventory.quantity_on_hand),
            recommendation=self._generate_safety_stock_recommendation(
                safety_stock, inventory.quantity_on_hand
            ),
            calculation_method=calculation_method,
            parameters={
                "z_score": z_score,
                "historical_days": 90
            }
        )
        
        self.db.add(calculation)
        self.db.commit()
        self.db.refresh(calculation)
        return calculation
    
    def _generate_safety_stock_recommendation(
        self,
        calculated_safety_stock: float,
        current_stock: Decimal
    ) -> str:
        """Generate safety stock recommendation"""
        if current_stock < calculated_safety_stock:
            deficit = calculated_safety_stock - float(current_stock)
            return f"Increase safety stock by {deficit:.0f} units to meet target"
        elif current_stock > calculated_safety_stock * 1.5:
            excess = float(current_stock) - calculated_safety_stock
            return f"Consider reducing stock by {excess:.0f} units to optimize carrying costs"
        else:
            return "Current stock level is within acceptable range"
    
    # =====================================================
    # Inventory History Operations
    # =====================================================
    
    def _create_history(
        self,
        inventory_id: UUID,
        transaction_type: str,
        quantity_change: Decimal,
        quantity_before: Decimal,
        quantity_after: Decimal,
        reason: str,
        performed_by: Optional[UUID] = None,
        reference_type: Optional[str] = None,
        reference_id: Optional[UUID] = None
    ):
        """Create inventory history record"""
        history = InventoryHistory(
            inventory_id=inventory_id,
            transaction_type=transaction_type,
            quantity_change=quantity_change,
            quantity_before=quantity_before,
            quantity_after=quantity_after,
            reason=reason,
            performed_by=performed_by,
            reference_type=reference_type,
            reference_id=reference_id
        )
        self.db.add(history)
    
    def get_inventory_history(
        self,
        inventory_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        transaction_type: Optional[str] = None,
        limit: int = 100
    ) -> List[InventoryHistory]:
        """Get inventory movement history"""
        query = self.db.query(InventoryHistory).filter(
            InventoryHistory.inventory_id == inventory_id
        )
        
        if start_date:
            query = query.filter(InventoryHistory.recorded_at >= start_date)
        
        if end_date:
            query = query.filter(InventoryHistory.recorded_at <= end_date)
        
        if transaction_type:
            query = query.filter(InventoryHistory.transaction_type == transaction_type)
        
        return query.order_by(
            desc(InventoryHistory.recorded_at)
        ).limit(limit).all()
    
    def get_inventory_movements_summary(
        self,
        start_date: datetime,
        end_date: datetime,
        location_code: Optional[str] = None,
        group_by: str = 'day'  # 'day', 'week', 'month'
    ) -> List[Dict[str, Any]]:
        """Get inventory movements summary"""
        # Date truncation based on grouping
        if group_by == 'day':
            date_trunc = func.date_trunc('day', InventoryHistory.recorded_at)
        elif group_by == 'week':
            date_trunc = func.date_trunc('week', InventoryHistory.recorded_at)
        else:
            date_trunc = func.date_trunc('month', InventoryHistory.recorded_at)
        
        query = self.db.query(
            date_trunc.label('period'),
            func.sum(
                case(
                    (InventoryHistory.quantity_change > 0, InventoryHistory.quantity_change),
                    else_=0
                )
            ).label('total_inbound'),
            func.sum(
                case(
                    (InventoryHistory.quantity_change < 0, func.abs(InventoryHistory.quantity_change)),
                    else_=0
                )
            ).label('total_outbound'),
            func.count(InventoryHistory.id).label('transaction_count')
        ).join(
            Inventory, InventoryHistory.inventory_id == Inventory.id
        ).filter(
            InventoryHistory.recorded_at.between(start_date, end_date)
        )
        
        if location_code:
            query = query.filter(Inventory.location_code == location_code)
        
        results = query.group_by('period').order_by('period').all()
        
        return [
            {
                "period": period.isoformat() if period else None,
                "total_inbound": float(total_inbound or 0),
                "total_outbound": float(total_outbound or 0),
                "net_change": float((total_inbound or 0) - (total_outbound or 0)),
                "transaction_count": transaction_count
            }
            for period, total_inbound, total_outbound, transaction_count in results
        ]
    
    # =====================================================
    # Bulk Operations
    # =====================================================
    
    def bulk_update_inventory(
        self,
        updates: List[Dict[str, Any]],
        performed_by: UUID,
        reason: str = "Bulk update"
    ) -> int:
        """Bulk update inventory quantities"""
        updated_count = 0
        
        for update in updates:
            inventory_id = update.get('inventory_id')
            new_quantity = update.get('quantity')
            
            if inventory_id and new_quantity is not None:
                inventory = self.update_inventory_quantity(
                    inventory_id=inventory_id,
                    new_quantity=Decimal(str(new_quantity)),
                    reason=reason,
                    performed_by=performed_by
                )
                if inventory:
                    updated_count += 1
        
        return updated_count
    
    def perform_cycle_count(
        self,
        location_code: str,
        count_date: date,
        counts: List[Dict[str, Any]],
        performed_by: UUID
    ) -> Dict[str, Any]:
        """Perform cycle count for a location"""
        results = {
            "counted": 0,
            "adjusted": 0,
            "errors": [],
            "adjustments": []
        }
        
        for count in counts:
            try:
                inventory_id = count.get('inventory_id')
                counted_quantity = Decimal(str(count.get('counted_quantity', 0)))
                
                inventory = self.get_inventory_by_id(inventory_id)
                if not inventory or inventory.location_code != location_code:
                    results["errors"].append(f"Invalid inventory ID: {inventory_id}")
                    continue
                
                results["counted"] += 1
                
                # Check if adjustment needed
                if inventory.quantity_on_hand != counted_quantity:
                    variance = counted_quantity - inventory.quantity_on_hand
                    
                    # Update inventory
                    self.update_inventory_quantity(
                        inventory_id=inventory_id,
                        new_quantity=counted_quantity,
                        reason=f"Cycle count adjustment on {count_date}",
                        performed_by=performed_by,
                        reference_type="cycle_count"
                    )
                    
                    results["adjusted"] += 1
                    results["adjustments"].append({
                        "inventory_id": str(inventory_id),
                        "item_name": getattr(inventory.product or inventory.material, 'name', 'Unknown'),
                        "previous_quantity": float(inventory.quantity_on_hand),
                        "counted_quantity": float(counted_quantity),
                        "variance": float(variance)
                    })
                
                # Update last counted date
                inventory.last_counted_date = count_date
                self.db.commit()
                
            except Exception as e:
                results["errors"].append(str(e))
        
        return results
    
    # =====================================================
    # Analytics and Reporting
    # =====================================================
    
    def get_inventory_metrics(
        self,
        location_code: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get comprehensive inventory metrics"""
        # Base query
        query = self.db.query(Inventory).options(
            joinedload(Inventory.product),
            joinedload(Inventory.material)
        )
        
        if location_code:
            query = query.filter(Inventory.location_code == location_code)
        
        # Calculate metrics
        total_items = query.count()
        
        # Items with zero stock
        zero_stock = query.filter(Inventory.quantity_on_hand == 0).count()
        
        # Items below reorder point
        below_reorder = query.filter(
            and_(
                Inventory.reorder_point.isnot(None),
                Inventory.quantity_available <= Inventory.reorder_point
            )
        ).count()
        
        # Get value metrics
        value_metrics = self.get_inventory_value(location_code)
        
        # Get turnover metrics if dates provided
        turnover_metrics = {}
        if start_date and end_date:
            turnover_metrics = self.get_inventory_turnover(
                start_date, end_date, location_code
            )
        
        return {
            "total_items": total_items,
            "zero_stock_items": zero_stock,
            "below_reorder_point": below_reorder,
            "stockout_rate": (zero_stock / total_items * 100) if total_items > 0 else 0,
            **value_metrics,
            **turnover_metrics
        }
    
    def get_location_summary(self) -> List[Dict[str, Any]]:
        """Get inventory summary by location"""
        results = self.db.query(
            Inventory.location_code,
            func.count(Inventory.id).label('item_count'),
            func.sum(Inventory.quantity_on_hand).label('total_quantity'),
            func.sum(
                case(
                    (Inventory.quantity_on_hand == 0, 1),
                    else_=0
                )
            ).label('zero_stock_count'),
            func.sum(
                case(
                    (and_(
                        Inventory.reorder_point.isnot(None),
                        Inventory.quantity_available <= Inventory.reorder_point
                    ), 1),
                    else_=0
                )
            ).label('low_stock_count')
        ).group_by(
            Inventory.location_code
        ).all()
        
        return [
            {
                "location_code": location,
                "item_count": item_count,
                "total_quantity": float(total_quantity or 0),
                "zero_stock_count": zero_stock_count,
                "low_stock_count": low_stock_count
            }
            for location, item_count, total_quantity, zero_stock_count, low_stock_count in results
        ]