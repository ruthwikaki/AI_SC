"""

Supply Chain domain models

"""



from datetime import datetime

from decimal import Decimal

from uuid import uuid4

from typing import Optional, List



from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DECIMAL,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint
)

from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

from sqlalchemy.orm import relationship, backref

from sqlalchemy.ext.hybrid import hybrid_property



from app.models.base import Base





class Supplier(Base):

    """Supplier/Vendor model"""

    __tablename__ = 'suppliers'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    code = Column(String(50), unique=True, nullable=False, index=True)

    name = Column(String(200), nullable=False)

    contact_name = Column(String(200))

    email = Column(String(255))

    phone = Column(String(50))

    address = Column(Text)

    city = Column(String(100))

    country = Column(String(100))

    postal_code = Column(String(20))

    tax_id = Column(String(50))

    payment_terms = Column(String(50))  # 'Net 30', 'Net 60', etc.

    rating = Column(Numeric(3, 2))  # 0.00 to 5.00

    is_active = Column(Boolean, default=True)

    meta_data = Column(JSONB, default={})

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    # Relationships

    products = relationship('Product', back_populates='supplier')

    

    def __repr__(self):

        return f"<Supplier(code={self.code}, name={self.name})>"





class Product(Base):

    """Product/Item model"""

    __tablename__ = 'products'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    sku = Column(String(100), unique=True, nullable=False, index=True)

    name = Column(String(200), nullable=False)

    description = Column(Text)

    category = Column(String(100))

    unit_price = Column(Numeric(10, 2), nullable=False)

    cost_price = Column(Numeric(10, 2))

    weight = Column(Numeric(10, 3))  # in kg

    dimensions = Column(String(50))  # LxWxH

    barcode = Column(String(50))

    reorder_level = Column(Integer, default=0)

    reorder_quantity = Column(Integer, default=0)

    lead_time_days = Column(Integer, default=0)

    supplier_id = Column(UUID(as_uuid=True), ForeignKey('suppliers.id'))

    is_active = Column(Boolean, default=True)

    meta_data = Column(JSONB, default={})

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    # Relationships

    supplier = relationship('Supplier', back_populates='products')

    inventory_items = relationship('Inventory', back_populates='product')

    order_items = relationship('OrderItem', back_populates='product')

    

    def __repr__(self):

        return f"<Product(sku={self.sku}, name={self.name})>"





class Warehouse(Base):

    """Warehouse/Distribution Center model"""

    __tablename__ = 'warehouses'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    code = Column(String(50), unique=True, nullable=False)

    name = Column(String(200), nullable=False)

    type = Column(String(50))  # 'central', 'regional', 'local'

    address = Column(Text)

    city = Column(String(100))

    state = Column(String(100))

    country = Column(String(100))

    postal_code = Column(String(20))

    latitude = Column(Numeric(10, 8))

    longitude = Column(Numeric(11, 8))

    capacity = Column(Numeric(10, 2))

    available_capacity = Column(Numeric(10, 2))

    manager_name = Column(String(200))

    manager_email = Column(String(255))

    phone = Column(String(50))

    operating_hours = Column(JSONB)

    is_active = Column(Boolean, default=True)

    meta_data = Column(JSONB, default={})

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    # Relationships

    inventory_items = relationship('Inventory', back_populates='warehouse')

    

    def __repr__(self):

        return f"<Warehouse(code={self.code}, name={self.name})>"





class Inventory(Base):

    """Inventory tracking model"""

    __tablename__ = 'inventory'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    product_id = Column(UUID(as_uuid=True), ForeignKey('products.id'), nullable=False)

    warehouse_id = Column(UUID(as_uuid=True), ForeignKey('warehouses.id'), nullable=False)

    quantity = Column(Integer, nullable=False, default=0)

    reserved_quantity = Column(Integer, default=0)

    available_quantity = Column(Integer, default=0)

    reorder_point = Column(Integer, default=0)

    reorder_quantity = Column(Integer, default=0)

    location = Column(String(50))  # Bin/Shelf location

    batch_number = Column(String(50))

    expiry_date = Column(Date)

    last_restock_date = Column(DateTime(timezone=True))

    last_count_date = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    # Relationships

    product = relationship('Product', back_populates='inventory_items')

    warehouse = relationship('Warehouse', back_populates='inventory_items')

    

    # Constraints

    __table_args__ = (

        UniqueConstraint('product_id', 'warehouse_id', 'batch_number', name='uq_inventory_product_warehouse_batch'),

        CheckConstraint('quantity >= 0', name='check_quantity_positive'),

        CheckConstraint('reserved_quantity >= 0', name='check_reserved_positive'),

        CheckConstraint('reserved_quantity <= quantity', name='check_reserved_not_exceed_quantity'),

    )

    

    @hybrid_property

    def available_quantity(self):

        return self.quantity - self.reserved_quantity

    

    def __repr__(self):

        return f"<Inventory(product_id={self.product_id}, warehouse_id={self.warehouse_id}, quantity={self.quantity})>"





class Customer(Base):

    """Customer model for supply chain"""

    __tablename__ = 'customers'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    code = Column(String(50), unique=True, nullable=False, index=True)

    name = Column(String(200), nullable=False)

    contact_name = Column(String(200))

    email = Column(String(255))

    phone = Column(String(50))

    billing_address = Column(Text)

    shipping_address = Column(Text)

    city = Column(String(100))

    state = Column(String(100))

    country = Column(String(100))

    postal_code = Column(String(20))

    credit_limit = Column(Numeric(12, 2))

    payment_terms = Column(String(50))  # 'Net 30', 'Net 60', 'COD', etc.

    tax_id = Column(String(50))

    customer_type = Column(String(50))  # 'retail', 'wholesale', 'distributor'

    is_active = Column(Boolean, default=True)

    meta_data = Column(JSONB, default={})

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    # Relationships

    orders = relationship('Order', back_populates='customer', cascade='all, delete-orphan')

    

    def __repr__(self):

        return f"<Customer(code={self.code}, name={self.name})>"





class Order(Base):

    """Sales/Purchase Order model"""

    __tablename__ = 'orders'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    order_number = Column(String(50), unique=True, nullable=False, index=True)

    order_type = Column(String(20), default='sales')  # 'sales' or 'purchase'

    customer_id = Column(UUID(as_uuid=True), ForeignKey('customers.id'), nullable=False)

    order_date = Column(DateTime(timezone=True), default=datetime.utcnow)

    required_date = Column(DateTime(timezone=True))

    shipped_date = Column(DateTime(timezone=True))

    status = Column(String(20), default='pending')  # pending, confirmed, shipped, delivered, cancelled

    total_amount = Column(Numeric(12, 2), default=0)

    tax_amount = Column(Numeric(10, 2), default=0)

    shipping_amount = Column(Numeric(10, 2), default=0)

    discount_amount = Column(Numeric(10, 2), default=0)

    payment_status = Column(String(20), default='pending')  # pending, partial, paid, refunded

    payment_method = Column(String(50))

    shipping_address = Column(Text)

    billing_address = Column(Text)

    notes = Column(Text)

    meta_data = Column(JSONB, default={})

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    # Relationships

    customer = relationship('Customer', back_populates='orders')

    items = relationship('OrderItem', back_populates='order', cascade='all, delete-orphan')

    shipments = relationship('Shipment', back_populates='order')

    

    def __repr__(self):

        return f"<Order(order_number={self.order_number}, status={self.status})>"





class OrderItem(Base):

    """Individual items within an order"""

    __tablename__ = 'order_items'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    order_id = Column(UUID(as_uuid=True), ForeignKey('orders.id', ondelete='CASCADE'), nullable=False)

    product_id = Column(UUID(as_uuid=True), ForeignKey('products.id'), nullable=False)

    sku = Column(String(100))

    quantity = Column(Integer, nullable=False)

    unit_price = Column(Numeric(10, 2), nullable=False)

    discount_amount = Column(Numeric(10, 2), default=0)

    tax_amount = Column(Numeric(10, 2), default=0)

    notes = Column(Text)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    # Relationships

    order = relationship('Order', back_populates='items')

    product = relationship('Product', back_populates='order_items')

    

    @hybrid_property

    def line_total(self):

        """Calculate line total"""

        return (self.quantity * self.unit_price) - self.discount_amount + self.tax_amount

    

    def __repr__(self):

        return f"<OrderItem(order_id={self.order_id}, product_id={self.product_id}, quantity={self.quantity})>"





class Shipment(Base):

    """Shipment/Delivery tracking model"""

    __tablename__ = 'shipments'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    shipment_number = Column(String(50), unique=True, nullable=False, index=True)

    order_id = Column(UUID(as_uuid=True), ForeignKey('orders.id'), nullable=False)

    carrier = Column(String(100))

    tracking_number = Column(String(100))

    shipped_date = Column(DateTime(timezone=True))

    estimated_delivery = Column(DateTime(timezone=True))

    actual_delivery = Column(DateTime(timezone=True))

    status = Column(String(20), default='pending')  # pending, in_transit, delivered, returned

    shipping_cost = Column(Numeric(10, 2))

    weight = Column(Numeric(10, 3))

    notes = Column(Text)

    meta_data = Column(JSONB, default={})

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    # Relationships

    order = relationship('Order', back_populates='shipments')

    

    def __repr__(self):

        return f"<Shipment(shipment_number={self.shipment_number}, status={self.status})>"

class SupplierTier(Base):

    """SupplierTier model"""

    __tablename__ = 'supplier_tiers'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<SupplierTier(id={self.id})>"



class SupplierRelationship(Base):

    """SupplierRelationship model"""

    __tablename__ = 'supplier_relationships'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<SupplierRelationship(id={self.id})>"



class Material(Base):

    """Material model"""

    __tablename__ = 'materials'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<Material(id={self.id})>"



class ProductMaterial(Base):

    """ProductMaterial model"""

    __tablename__ = 'product_materials'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<ProductMaterial(id={self.id})>"



class InventoryHistory(Base):

    """InventoryHistory model"""

    __tablename__ = 'inventory_history'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<InventoryHistory(id={self.id})>"



class ShipmentItem(Base):

    """ShipmentItem model"""

    __tablename__ = 'shipment_items'

    

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String(255))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    

    def __repr__(self):

        return f"<ShipmentItem(id={self.id})>"



class SupplierProduct(Base):
    """Supplier-Product relationship (which suppliers provide which products)"""
    __tablename__ = 'supplier_products'
    
    supplier_id = Column(UUID(as_uuid=True), ForeignKey('suppliers.id'), primary_key=True)
    product_id = Column(UUID(as_uuid=True), ForeignKey('products.id'), primary_key=True)
    supplier_sku = Column(String(100))
    lead_time_days = Column(Integer)
    minimum_order_quantity = Column(DECIMAL(15, 3))
    unit_price = Column(DECIMAL(15, 2))
    is_preferred = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    supplier = relationship('Supplier')
    product = relationship('Product')
    
    def __repr__(self):
        return f"<SupplierProduct(supplier_id={self.supplier_id}, product_id={self.product_id})>"
