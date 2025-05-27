"""
Order repository for order management operations
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from uuid import UUID
import logging

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, or_, and_, desc, asc
from sqlalchemy.exc import IntegrityError

from app.models import (
    Order, OrderItem, Shipment, ShipmentItem,
    Supplier, Product, Material, Inventory
)

logger = logging.getLogger(__name__)


class OrderRepository:
    """Repository for order-related database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # =====================================================
    # Order CRUD Operations
    # =====================================================
    
    def create_order(
        self,
        order_data: Dict[str, Any],
        items: List[Dict[str, Any]],
        created_by: UUID
    ) -> Order:
        """Create a new order with items"""
        try:
            # Generate order number if not provided
            if 'order_number' not in order_data:
                order_data['order_number'] = self._generate_order_number(
                    order_data.get('order_type', 'purchase')
                )
            
            # Create order
            order = Order(
                **order_data,
                created_by=created_by,
                status='draft' if 'status' not in order_data else order_data['status']
            )
            self.db.add(order)
            self.db.flush()  # Get order ID without committing
            
            # Create order items
            total_amount = Decimal('0')
            for idx, item_data in enumerate(items, 1):
                item = OrderItem(
                    order_id=order.id,
                    line_number=item_data.get('line_number', idx),
                    **{k: v for k, v in item_data.items() if k != 'line_number'}
                )
                self.db.add(item)
                
                # Calculate line total
                quantity = Decimal(str(item.quantity))
                unit_price = Decimal(str(item.unit_price))
                discount_percent = Decimal(str(item.discount_percent or 0))
                tax_percent = Decimal(str(item.tax_percent or 0))
                
                line_total = quantity * unit_price * (1 - discount_percent/100) * (1 + tax_percent/100)
                total_amount += line_total
            
            # Update order total
            order.total_amount = total_amount
            
            self.db.commit()
            self.db.refresh(order)
            return order
            
        except IntegrityError as e:
            self.db.rollback()
            if 'order_number' in str(e.orig):
                raise ValueError(f"Order number already exists: {order_data.get('order_number')}")
            raise
    
    def _generate_order_number(self, order_type: str) -> str:
        """Generate unique order number"""
        prefix = 'PO' if order_type == 'purchase' else 'SO'
        today = date.today()
        
        # Get count of orders today
        count = self.db.query(Order).filter(
            func.date(Order.created_at) == today
        ).count()
        
        return f"{prefix}-{today.strftime('%Y%m%d')}-{str(count + 1).zfill(4)}"
    
    def get_order_by_id(self, order_id: UUID) -> Optional[Order]:
        """Get order by ID with related data"""
        return self.db.query(Order).options(
            joinedload(Order.supplier),
            joinedload(Order.items).joinedload(OrderItem.product),
            joinedload(Order.items).joinedload(OrderItem.material),
            joinedload(Order.shipments)
        ).filter(Order.id == order_id).first()
    
    def get_order_by_number(self, order_number: str) -> Optional[Order]:
        """Get order by order number"""
        return self.db.query(Order).filter(
            Order.order_number == order_number
        ).first()
    
    def get_orders(
        self,
        skip: int = 0,
        limit: int = 100,
        order_type: Optional[str] = None,
        status: Optional[str] = None,
        supplier_id: Optional[UUID] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        search: Optional[str] = None,
        sort_by: str = 'order_date',
        sort_order: str = 'desc'
    ) -> Tuple[List[Order], int]:
        """Get orders with filters and pagination"""
        query = self.db.query(Order).options(
            joinedload(Order.supplier)
        )
        
        # Apply filters
        if order_type:
            query = query.filter(Order.order_type == order_type)
        
        if status:
            query = query.filter(Order.status == status)
        
        if supplier_id:
            query = query.filter(Order.supplier_id == supplier_id)
        
        if start_date:
            query = query.filter(Order.order_date >= start_date)
        
        if end_date:
            query = query.filter(Order.order_date <= end_date)
        
        if search:
            search_filter = f"%{search}%"
            query = query.filter(
                or_(
                    Order.order_number.ilike(search_filter),
                    Order.notes.ilike(search_filter)
                )
            )
        
        # Get total count
        total = query.count()
        
        # Apply sorting
        sort_column = getattr(Order, sort_by, Order.order_date)
        if sort_order == 'desc':
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Apply pagination
        orders = query.offset(skip).limit(limit).all()
        
        return orders, total
    
    def update_order(
        self,
        order_id: UUID,
        update_data: Dict[str, Any]
    ) -> Optional[Order]:
        """Update order information"""
        order = self.get_order_by_id(order_id)
        if not order:
            return None
        
        # Don't allow status changes through this method
        if 'status' in update_data:
            del update_data['status']
        
        for key, value in update_data.items():
            if hasattr(order, key):
                setattr(order, key, value)
        
        order.updated_at = datetime.utcnow()
        
        # Recalculate total if needed
        if 'recalculate_total' in update_data and update_data['recalculate_total']:
            order.total_amount = order.calculate_total()
        
        self.db.commit()
        self.db.refresh(order)
        return order
    
    def update_order_status(
        self,
        order_id: UUID,
        new_status: str,
        notes: Optional[str] = None
    ) -> Optional[Order]:
        """Update order status with validation"""
        order = self.get_order_by_id(order_id)
        if not order:
            return None
        
        # Validate status transition
        valid_transitions = {
            'draft': ['submitted', 'cancelled'],
            'submitted': ['approved', 'rejected', 'cancelled'],
            'approved': ['in_progress', 'cancelled'],
            'in_progress': ['partially_shipped', 'shipped', 'cancelled'],
            'partially_shipped': ['shipped', 'cancelled'],
            'shipped': ['delivered', 'returned'],
            'delivered': ['closed', 'returned'],
            'rejected': ['draft'],
            'cancelled': [],
            'returned': ['closed'],
            'closed': []
        }
        
        current_status = order.status
        if new_status not in valid_transitions.get(current_status, []):
            raise ValueError(f"Invalid status transition from {current_status} to {new_status}")
        
        order.status = new_status
        order.updated_at = datetime.utcnow()
        
        if notes:
            order.notes = (order.notes or '') + f"\n[{datetime.utcnow()}] Status changed: {notes}"
        
        # Update related dates
        if new_status == 'delivered':
            order.actual_delivery_date = date.today()
        
        self.db.commit()
        self.db.refresh(order)
        return order
    
    def delete_order(self, order_id: UUID) -> bool:
        """Delete order (only if draft status)"""
        order = self.get_order_by_id(order_id)
        if not order or order.status != 'draft':
            return False
        
        self.db.delete(order)
        self.db.commit()
        return True
    
    # =====================================================
    # Order Item Operations
    # =====================================================
    
    def add_order_item(
        self,
        order_id: UUID,
        item_data: Dict[str, Any]
    ) -> Optional[OrderItem]:
        """Add item to existing order"""
        order = self.get_order_by_id(order_id)
        if not order or order.status not in ['draft', 'submitted']:
            return None
        
        # Get next line number
        max_line = self.db.query(func.max(OrderItem.line_number)).filter(
            OrderItem.order_id == order_id
        ).scalar() or 0
        
        item = OrderItem(
            order_id=order_id,
            line_number=item_data.get('line_number', max_line + 1),
            **{k: v for k, v in item_data.items() if k != 'line_number'}
        )
        self.db.add(item)
        
        # Update order total
        order.total_amount = order.calculate_total()
        
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def update_order_item(
        self,
        item_id: UUID,
        update_data: Dict[str, Any]
    ) -> Optional[OrderItem]:
        """Update order item"""
        item = self.db.query(OrderItem).filter(
            OrderItem.id == item_id
        ).first()
        
        if not item:
            return None
        
        # Check if order is editable
        order = item.order
        if order.status not in ['draft', 'submitted']:
            raise ValueError("Cannot update items for order in current status")
        
        for key, value in update_data.items():
            if hasattr(item, key):
                setattr(item, key, value)
        
        # Update order total
        order.total_amount = order.calculate_total()
        order.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def delete_order_item(self, item_id: UUID) -> bool:
        """Delete order item"""
        item = self.db.query(OrderItem).filter(
            OrderItem.id == item_id
        ).first()
        
        if not item:
            return False
        
        # Check if order is editable
        order = item.order
        if order.status not in ['draft', 'submitted']:
            return False
        
        self.db.delete(item)
        
        # Update order total
        order.total_amount = order.calculate_total()
        order.updated_at = datetime.utcnow()
        
        self.db.commit()
        return True
    
    # =====================================================
    # Shipment Operations
    # =====================================================
    
    def create_shipment(
        self,
        shipment_data: Dict[str, Any],
        items: List[Dict[str, Any]]
    ) -> Shipment:
        """Create shipment for order"""
        try:
            # Generate shipment number if not provided
            if 'shipment_number' not in shipment_data:
                shipment_data['shipment_number'] = self._generate_shipment_number()
            
            shipment = Shipment(**shipment_data)
            self.db.add(shipment)
            self.db.flush()
            
            # Create shipment items
            for item_data in items:
                shipment_item = ShipmentItem(
                    shipment_id=shipment.id,
                    **item_data
                )
                self.db.add(shipment_item)
            
            self.db.commit()
            self.db.refresh(shipment)
            return shipment
            
        except IntegrityError:
            self.db.rollback()
            raise
    
    def _generate_shipment_number(self) -> str:
        """Generate unique shipment number"""
        today = date.today()
        count = self.db.query(Shipment).filter(
            func.date(Shipment.created_at) == today
        ).count()
        
        return f"SHP-{today.strftime('%Y%m%d')}-{str(count + 1).zfill(4)}"
    
    def get_shipment_by_id(self, shipment_id: UUID) -> Optional[Shipment]:
        """Get shipment by ID"""
        return self.db.query(Shipment).options(
            joinedload(Shipment.order),
            joinedload(Shipment.items)
        ).filter(Shipment.id == shipment_id).first()
    
    def get_shipments_for_order(self, order_id: UUID) -> List[Shipment]:
        """Get all shipments for an order"""
        return self.db.query(Shipment).filter(
            Shipment.order_id == order_id
        ).order_by(Shipment.ship_date).all()
    
    def update_shipment_status(
        self,
        shipment_id: UUID,
        status: str,
        tracking_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Shipment]:
        """Update shipment status"""
        shipment = self.get_shipment_by_id(shipment_id)
        if not shipment:
            return None
        
        shipment.status = status
        shipment.updated_at = datetime.utcnow()
        
        # Update dates based on status
        if status == 'in_transit' and not shipment.ship_date:
            shipment.ship_date = datetime.utcnow()
        elif status == 'delivered' and not shipment.actual_delivery_date:
            shipment.actual_delivery_date = datetime.utcnow()
        
        # Update tracking info
        if tracking_info:
            if 'tracking_number' in tracking_info:
                shipment.tracking_number = tracking_info['tracking_number']
            if 'carrier_name' in tracking_info:
                shipment.carrier_name = tracking_info['carrier_name']
        
        # Update order status if needed
        order = shipment.order
        if order and status == 'delivered':
            # Check if all items are delivered
            total_ordered = sum(item.quantity for item in order.items)
            total_shipped = sum(
                sum(si.quantity_shipped for si in s.items)
                for s in order.shipments
                if s.status == 'delivered'
            )
            
            if total_shipped >= total_ordered:
                order.status = 'delivered'
                order.actual_delivery_date = date.today()
            elif total_shipped > 0:
                order.status = 'partially_shipped'
        
        self.db.commit()
        self.db.refresh(shipment)
        return shipment
    
    # =====================================================
    # Order Fulfillment Operations
    # =====================================================
    
    def check_order_fulfillment(self, order_id: UUID) -> Dict[str, Any]:
        """Check order fulfillment status"""
        order = self.get_order_by_id(order_id)
        if not order:
            return {}
        
        fulfillment = {
            "order_id": str(order.id),
            "order_number": order.order_number,
            "items": []
        }
        
        for item in order.items:
            # Get shipped quantity
            shipped_qty = self.db.query(
                func.sum(ShipmentItem.quantity_shipped)
            ).join(
                Shipment
            ).filter(
                ShipmentItem.order_item_id == item.id,
                Shipment.status.in_(['in_transit', 'delivered'])
            ).scalar() or 0
            
            fulfillment["items"].append({
                "item_id": str(item.id),
                "item_name": item.item_name,
                "ordered_quantity": float(item.quantity),
                "shipped_quantity": float(shipped_qty),
                "remaining_quantity": float(item.quantity - shipped_qty),
                "fulfillment_percentage": (shipped_qty / item.quantity * 100) if item.quantity > 0 else 0
            })
        
        # Calculate overall fulfillment
        total_ordered = sum(item["ordered_quantity"] for item in fulfillment["items"])
        total_shipped = sum(item["shipped_quantity"] for item in fulfillment["items"])
        
        fulfillment["overall_fulfillment_percentage"] = (
            (total_shipped / total_ordered * 100) if total_ordered > 0 else 0
        )
        
        return fulfillment
    
    def allocate_inventory_to_order(
        self,
        order_id: UUID,
        allocation_strategy: str = 'fifo'
    ) -> Dict[str, Any]:
        """Allocate inventory to order items"""
        order = self.get_order_by_id(order_id)
        if not order or order.status != 'approved':
            return {"error": "Order not found or not approved"}
        
        allocations = []
        errors = []
        
        for item in order.items:
            # Get available inventory
            inventory_query = self.db.query(Inventory).filter(
                Inventory.quantity_available > 0
            )
            
            if item.product_id:
                inventory_query = inventory_query.filter(
                    Inventory.product_id == item.product_id
                )
            elif item.material_id:
                inventory_query = inventory_query.filter(
                    Inventory.material_id == item.material_id
                )
            else:
                continue
            
            # Apply allocation strategy
            if allocation_strategy == 'fifo':
                inventory_query = inventory_query.order_by(Inventory.created_at)
            
            inventory_list = inventory_query.all()
            
            remaining_qty = item.quantity
            item_allocations = []
            
            for inv in inventory_list:
                if remaining_qty <= 0:
                    break
                
                allocate_qty = min(inv.quantity_available, remaining_qty)
                
                # Reserve inventory
                inv.quantity_reserved += allocate_qty
                remaining_qty -= allocate_qty
                
                item_allocations.append({
                    "inventory_id": str(inv.id),
                    "location": inv.location_code,
                    "allocated_quantity": float(allocate_qty)
                })
            
            if remaining_qty > 0:
                errors.append(f"Insufficient inventory for {item.item_name}. Short by {remaining_qty}")
            
            allocations.append({
                "item_id": str(item.id),
                "item_name": item.item_name,
                "requested_quantity": float(item.quantity),
                "allocated_quantity": float(item.quantity - remaining_qty),
                "allocations": item_allocations
            })
        
        if not errors:
            order.status = 'in_progress'
            self.db.commit()
        else:
            self.db.rollback()
        
        return {
            "order_id": str(order.id),
            "allocations": allocations,
            "errors": errors,
            "success": len(errors) == 0
        }
    
    # =====================================================
    # Analytics and Reporting
    # =====================================================
    
    def get_order_statistics(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        order_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get order statistics"""
        query = self.db.query(Order)
        
        if start_date:
            query = query.filter(Order.order_date >= start_date)
        
        if end_date:
            query = query.filter(Order.order_date <= end_date)
        
        if order_type:
            query = query.filter(Order.order_type == order_type)
        
        # Basic counts
        total_orders = query.count()
        
        status_distribution = dict(
            query.with_entities(
                Order.status,
                func.count(Order.id)
            ).group_by(Order.status).all()
        )
        
        # Value statistics
        value_stats = query.with_entities(
            func.sum(Order.total_amount),
            func.avg(Order.total_amount),
            func.min(Order.total_amount),
            func.max(Order.total_amount)
        ).first()
        
        # Delivery performance
        delivered_orders = query.filter(
            Order.status == 'delivered',
            Order.requested_delivery_date.isnot(None),
            Order.actual_delivery_date.isnot(None)
        ).all()
        
        on_time_deliveries = sum(
            1 for o in delivered_orders
            if o.actual_delivery_date <= o.requested_delivery_date
        )
        
        return {
            "total_orders": total_orders,
            "status_distribution": status_distribution,
            "total_value": float(value_stats[0] or 0),
            "average_value": float(value_stats[1] or 0),
            "min_value": float(value_stats[2] or 0),
            "max_value": float(value_stats[3] or 0),
            "on_time_delivery_rate": (
                (on_time_deliveries / len(delivered_orders) * 100)
                if delivered_orders else 0
            )
        }
    
    def get_pending_orders(
        self,
        supplier_id: Optional[UUID] = None,
        due_within_days: int = 7
    ) -> List[Order]:
        """Get pending orders due soon"""
        due_date = date.today() + timedelta(days=due_within_days)
        
        query = self.db.query(Order).filter(
            Order.status.in_(['submitted', 'approved', 'in_progress']),
            Order.requested_delivery_date <= due_date
        )
        
        if supplier_id:
            query = query.filter(Order.supplier_id == supplier_id)
        
        return query.order_by(Order.requested_delivery_date).all()
    
    def get_order_timeline(self, order_id: UUID) -> List[Dict[str, Any]]:
        """Get order timeline events"""
        order = self.get_order_by_id(order_id)
        if not order:
            return []
        
        timeline = []
        
        # Order creation
        timeline.append({
            "timestamp": order.created_at,
            "event": "Order created",
            "status": "draft",
            "details": f"Order {order.order_number} created"
        })
        
        # Status changes (simplified - in production, track in audit log)
        if order.status != 'draft':
            timeline.append({
                "timestamp": order.updated_at,
                "event": "Status changed",
                "status": order.status,
                "details": f"Order status changed to {order.status}"
            })
        
        # Shipments
        for shipment in order.shipments:
            if shipment.ship_date:
                timeline.append({
                    "timestamp": shipment.ship_date,
                    "event": "Shipment dispatched",
                    "status": "shipped",
                    "details": f"Shipment {shipment.shipment_number} dispatched"
                })
            
            if shipment.actual_delivery_date:
                timeline.append({
                    "timestamp": shipment.actual_delivery_date,
                    "event": "Shipment delivered",
                    "status": "delivered",
                    "details": f"Shipment {shipment.shipment_number} delivered"
                })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        
        return timeline