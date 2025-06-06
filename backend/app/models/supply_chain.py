"""
Supply chain core business data models
from app.models.base import Base

"""

from datetime import datetime, date
from typing import Optional, List
from uuid import uuid4
from decimal import Decimal

from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, ForeignKey,
    Text, Date, DECIMAL, CheckConstraint, UniqueConstraint,
    Computed, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship



class Supplier(Base):
    """Supplier master data"""
    __tablename__ = 'suppliers'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    code = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100))
    status = Column(String(50), default='active')
    tier = Column(Integer, default=1, index=True)
    country = Column(String(2))
    city = Column(String(100))
    address = Column(Text)
    contact_name = Column(String(255))
    contact_email = Column(String(255))
    contact_phone = Column(String(50))
    payment_terms = Column(String(100))
    lead_time_days = Column(Integer)
    minimum_order_value = Column(DECIMAL(15, 2))
    rating = Column(DECIMAL(3, 2))
    certifications = Column(JSONB, default=list)
    query_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tiers = relationship('SupplierTier', back_populates='supplier', cascade='all, delete-orphan')
    parent_relationships = relationship('SupplierRelationship', 
                                       foreign_keys='SupplierRelationship.child_supplier_id',
                                       back_populates='child_supplier')
    child_relationships = relationship('SupplierRelationship',
                                      foreign_keys='SupplierRelationship.parent_supplier_id',
                                      back_populates='parent_supplier')
    orders = relationship('Order', back_populates='supplier')
    performance_metrics = relationship('SupplierPerformanceMetric', back_populates='supplier')
    
    def __repr__(self):
        return f"<Supplier(code={self.code}, name={self.name})>"
    
    @property
    def is_active(self) -> bool:
        """Check if supplier is active"""
        return self.status == 'active'
    
    def get_tier_level(self, verified_only: bool = False) -> Optional[int]:
        """Get supplier tier level"""
        if not self.tiers:
            return self.tier
        
        if verified_only:
            verified_tiers = [t for t in self.tiers if t.verified]
            if verified_tiers:
                return min(t.tier_level for t in verified_tiers)
        
        return min(t.tier_level for t in self.tiers)


class SupplierTier(Base):
    """Supplier tier classifications"""
    __tablename__ = 'supplier_tiers'
    
    supplier_id = Column(UUID(as_uuid=True), ForeignKey('suppliers.id', ondelete='CASCADE'), primary_key=True)
    tier_level = Column(Integer, primary_key=True, nullable=False)
    classification = Column(String(100))
    verified = Column(Boolean, default=False)
    verified_date = Column(DateTime(timezone=True))
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    supplier = relationship('Supplier', back_populates='tiers')
    
    def __repr__(self):
        return f"<SupplierTier(supplier_id={self.supplier_id}, tier_level={self.tier_level})>"


class SupplierRelationship(Base):
    """Supplier to supplier relationships"""
    __tablename__ = 'supplier_relationships'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    parent_supplier_id = Column(UUID(as_uuid=True), ForeignKey('suppliers.id'), nullable=False)
    child_supplier_id = Column(UUID(as_uuid=True), ForeignKey('suppliers.id'), nullable=False)
    relationship_type = Column(String(100), nullable=False)
    strength = Column(DECIMAL(3, 2))
    start_date = Column(Date)
    end_date = Column(Date)
    is_active = Column(Boolean, default=True)
    query_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    parent_supplier = relationship('Supplier', foreign_keys=[parent_supplier_id], back_populates='child_relationships')
    child_supplier = relationship('Supplier', foreign_keys=[child_supplier_id], back_populates='parent_relationships')
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('parent_supplier_id', 'child_supplier_id', 'relationship_type', 
                        name='uq_supplier_relationship'),
        CheckConstraint('parent_supplier_id != child_supplier_id', name='ck_different_suppliers'),
    )
    
    def __repr__(self):
        return f"<SupplierRelationship(parent={self.parent_supplier_id}, child={self.child_supplier_id})>"


class Product(Base):
    """Product master data"""
    __tablename__ = 'products'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sku = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    subcategory = Column(String(100))
    unit_of_measure = Column(String(50))
    weight_kg = Column(DECIMAL(10, 3))
    volume_m3 = Column(DECIMAL(10, 3))
    unit_cost = Column(DECIMAL(15, 2))
    selling_price = Column(DECIMAL(15, 2))
    status = Column(String(50), default='active')
    launch_date = Column(Date)
    discontinue_date = Column(Date)
    attributes = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    materials = relationship('ProductMaterial', back_populates='product', cascade='all, delete-orphan')
    inventory = relationship('Inventory', back_populates='product')
    order_items = relationship('OrderItem', back_populates='product')
    
    def __repr__(self):
        return f"<Product(sku={self.sku}, name={self.name})>"
    
    @property
    def is_active(self) -> bool:
        """Check if product is active"""
        return self.status == 'active' and (self.discontinue_date is None or self.discontinue_date > date.today())
    
    @property
    def margin(self) -> Optional[Decimal]:
        """Calculate profit margin"""
        if self.selling_price and self.unit_cost and self.unit_cost > 0:
            return ((self.selling_price - self.unit_cost) / self.unit_cost) * 100
        return None


class Material(Base):
    """Material/component master data"""
    __tablename__ = 'materials'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    code = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    type = Column(String(100))
    unit_of_measure = Column(String(50))
    unit_cost = Column(DECIMAL(15, 2))
    lead_time_days = Column(Integer)
    minimum_order_quantity = Column(DECIMAL(15, 3))
    is_critical = Column(Boolean, default=False)
    specifications = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    products = relationship('ProductMaterial', back_populates='material')
    inventory = relationship('Inventory', back_populates='material')
    order_items = relationship('OrderItem', back_populates='material')
    
    def __repr__(self):
        return f"<Material(code={self.code}, name={self.name})>"


class ProductMaterial(Base):
    """Bill of Materials (BOM) - links products to materials"""
    __tablename__ = 'product_materials'
    
    product_id = Column(UUID(as_uuid=True), ForeignKey('products.id', ondelete='CASCADE'), primary_key=True)
    material_id = Column(UUID(as_uuid=True), ForeignKey('materials.id'), primary_key=True)
    quantity = Column(DECIMAL(15, 3), nullable=False)
    unit_of_measure = Column(String(50))
    is_primary = Column(Boolean, default=False)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    product = relationship('Product', back_populates='materials')
    material = relationship('Material', back_populates='products')
    
    def __repr__(self):
        return f"<ProductMaterial(product_id={self.product_id}, material_id={self.material_id})>"


class Inventory(Base):
    """Current inventory levels"""
    __tablename__ = 'inventory'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey('products.id'), index=True)
    material_id = Column(UUID(as_uuid=True), ForeignKey('materials.id'), index=True)
    location_code = Column(String(100), nullable=False)
    quantity_on_hand = Column(DECIMAL(15, 3), nullable=False, default=0)
    quantity_reserved = Column(DECIMAL(15, 3), nullable=False, default=0)
    quantity_available = Column(DECIMAL(15, 3), 
                               Computed('quantity_on_hand - quantity_reserved'))
    reorder_point = Column(DECIMAL(15, 3))
    reorder_quantity = Column(DECIMAL(15, 3))
    last_counted_date = Column(Date)
    last_movement_date = Column(DateTime(timezone=True))
    query_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    product = relationship('Product', back_populates='inventory')
    material = relationship('Material', back_populates='inventory')
    history = relationship('InventoryHistory', back_populates='inventory', cascade='all, delete-orphan')
    
    # Constraints
    __table_args__ = (
        CheckConstraint('(product_id IS NOT NULL AND material_id IS NULL) OR '
                       '(product_id IS NULL AND material_id IS NOT NULL)',
                       name='ck_inventory_product_or_material'),
    )
    
    def __repr__(self):
        item_type = 'product' if self.product_id else 'material'
        item_id = self.product_id or self.material_id
        return f"<Inventory({item_type}={item_id}, location={self.location_code})>"
    
    @property
    def needs_reorder(self) -> bool:
        """Check if inventory needs reordering"""
        if self.reorder_point is None:
            return False
        return self.quantity_available <= self.reorder_point
    
    @property
    def item_name(self) -> Optional[str]:
        """Get the name of the inventory item"""
        if self.product:
            return self.product.name
        elif self.material:
            return self.material.name
        return None


class InventoryHistory(Base):
    """Inventory transaction history"""
    __tablename__ = 'inventory_history'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    inventory_id = Column(UUID(as_uuid=True), ForeignKey('inventory.id'), nullable=False)
    transaction_type = Column(String(50), nullable=False)
    quantity_change = Column(DECIMAL(15, 3), nullable=False)
    quantity_before = Column(DECIMAL(15, 3), nullable=False)
    quantity_after = Column(DECIMAL(15, 3), nullable=False)
    reference_type = Column(String(50))
    reference_id = Column(UUID(as_uuid=True))
    reason = Column(String(255))
    performed_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    recorded_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    inventory = relationship('Inventory', back_populates='history')
    user = relationship('User')
    
    def __repr__(self):
        return f"<InventoryHistory(id={self.id}, type={self.transaction_type}, change={self.quantity_change})>"


class Order(Base):
    """Purchase and sales orders"""
    __tablename__ = 'orders'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    order_number = Column(String(100), unique=True, nullable=False, index=True)
    order_type = Column(String(50), nullable=False)
    supplier_id = Column(UUID(as_uuid=True), ForeignKey('suppliers.id'), index=True)
    status = Column(String(50), nullable=False, default='draft', index=True)
    order_date = Column(Date, nullable=False)
    requested_delivery_date = Column(Date)
    actual_delivery_date = Column(Date)
    total_amount = Column(DECIMAL(15, 2))
    currency = Column(String(3), default='USD')
    payment_status = Column(String(50), default='pending')
    shipping_address = Column(Text)
    billing_address = Column(Text)
    notes = Column(Text)
    query_metadata = Column(JSONB, default={})
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    supplier = relationship('Supplier', back_populates='orders')
    items = relationship('OrderItem', back_populates='order', cascade='all, delete-orphan')
    shipments = relationship('Shipment', back_populates='order')
    created_by_user = relationship('User')
    
    def __repr__(self):
        return f"<Order(order_number={self.order_number}, status={self.status})>"
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete"""
        return self.status in ['delivered', 'completed', 'closed']
    
    @property
    def is_overdue(self) -> bool:
        """Check if order is overdue"""
        if self.requested_delivery_date and not self.is_complete:
            return date.today() > self.requested_delivery_date
        return False
    
    def calculate_total(self) -> Decimal:
        """Calculate order total from items"""
        if not self.items:
            return Decimal('0.00')
        return sum(item.line_total for item in self.items if item.line_total)


class OrderItem(Base):
    """Order line items"""
    __tablename__ = 'order_items'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    order_id = Column(UUID(as_uuid=True), ForeignKey('orders.id', ondelete='CASCADE'), nullable=False)
    line_number = Column(Integer, nullable=False)
    product_id = Column(UUID(as_uuid=True), ForeignKey('products.id'))
    material_id = Column(UUID(as_uuid=True), ForeignKey('materials.id'))
    quantity = Column(DECIMAL(15, 3), nullable=False)
    unit_price = Column(DECIMAL(15, 2), nullable=False)
    discount_percent = Column(DECIMAL(5, 2), default=0)
    tax_percent = Column(DECIMAL(5, 2), default=0)
    line_total = Column(DECIMAL(15, 2), 
                       Computed('quantity * unit_price * (1 - discount_percent/100) * (1 + tax_percent/100)'))
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    order = relationship('Order', back_populates='items')
    product = relationship('Product', back_populates='order_items')
    material = relationship('Material', back_populates='order_items')
    shipment_items = relationship('ShipmentItem', back_populates='order_item')
    
    # Constraints
    __table_args__ = (
        CheckConstraint('(product_id IS NOT NULL AND material_id IS NULL) OR '
                       '(product_id IS NULL AND material_id IS NOT NULL)',
                       name='ck_order_item_product_or_material'),
    )
    
    def __repr__(self):
        return f"<OrderItem(order_id={self.order_id}, line={self.line_number})>"
    
    @property
    def item_name(self) -> Optional[str]:
        """Get the name of the ordered item"""
        if self.product:
            return self.product.name
        elif self.material:
            return self.material.name
        return None
    
    @property
    def subtotal(self) -> Decimal:
        """Calculate line subtotal before tax"""
        return self.quantity * self.unit_price * (1 - self.discount_percent/100)


class Shipment(Base):
    """Shipment tracking"""
    __tablename__ = 'shipments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    shipment_number = Column(String(100), unique=True, nullable=False, index=True)
    order_id = Column(UUID(as_uuid=True), ForeignKey('orders.id'))
    carrier_name = Column(String(255))
    tracking_number = Column(String(255), index=True)
    status = Column(String(50), nullable=False, default='pending')
    ship_date = Column(DateTime(timezone=True))
    estimated_delivery_date = Column(DateTime(timezone=True))
    actual_delivery_date = Column(DateTime(timezone=True))
    origin_address = Column(Text)
    destination_address = Column(Text)
    shipping_cost = Column(DECIMAL(15, 2))
    weight_kg = Column(DECIMAL(10, 3))
    query_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    order = relationship('Order', back_populates='shipments')
    items = relationship('ShipmentItem', back_populates='shipment', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Shipment(shipment_number={self.shipment_number}, status={self.status})>"
    
    @property
    def is_delivered(self) -> bool:
        """Check if shipment is delivered"""
        return self.status == 'delivered' and self.actual_delivery_date is not None
    
    @property
    def is_in_transit(self) -> bool:
        """Check if shipment is in transit"""
        return self.status == 'in_transit' and self.ship_date is not None
    
    @property
    def delivery_variance_hours(self) -> Optional[float]:
        """Calculate delivery time variance in hours"""
        if self.estimated_delivery_date and self.actual_delivery_date:
            delta = self.actual_delivery_date - self.estimated_delivery_date
            return delta.total_seconds() / 3600
        return None


class ShipmentItem(Base):
    """Items in a shipment"""
    __tablename__ = 'shipment_items'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    shipment_id = Column(UUID(as_uuid=True), ForeignKey('shipments.id', ondelete='CASCADE'), nullable=False)
    order_item_id = Column(UUID(as_uuid=True), ForeignKey('order_items.id'))
    quantity_shipped = Column(DECIMAL(15, 3), nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    shipment = relationship('Shipment', back_populates='items')
    order_item = relationship('OrderItem', back_populates='shipment_items')
    
    def __repr__(self):
        return f"<ShipmentItem(shipment_id={self.shipment_id}, quantity={self.quantity_shipped})>"