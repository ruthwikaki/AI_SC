"""
Supply chain business entity schemas
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, condecimal
from enum import Enum


# =====================================================
# Enums
# =====================================================

class SupplierStatus(str, Enum):
    """Supplier status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class OrderStatus(str, Enum):
    """Order status"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    PARTIALLY_SHIPPED = "partially_shipped"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"
    CLOSED = "closed"


class ShipmentStatus(str, Enum):
    """Shipment status"""
    PENDING = "pending"
    READY = "ready"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    RETURNED = "returned"
    LOST = "lost"


class ComplianceStatus(str, Enum):
    """Compliance status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    EXPIRED = "expired"


# =====================================================
# Supplier Schemas
# =====================================================

class SupplierBase(BaseModel):
    """Base supplier schema"""
    code: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    category: Optional[str] = Field(None, max_length=100)
    tier: int = Field(default=1, ge=1, le=5)
    country: Optional[str] = Field(None, min_length=2, max_length=2)
    city: Optional[str] = Field(None, max_length=100)
    address: Optional[str] = None
    contact_name: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    contact_phone: Optional[str] = Field(None, max_length=50)
    payment_terms: Optional[str] = Field(None, max_length=100)
    lead_time_days: Optional[int] = Field(None, ge=0)
    minimum_order_value: Optional[condecimal(max_digits=15, decimal_places=2)] = None


class SupplierCreate(SupplierBase):
    """Create supplier request"""
    status: SupplierStatus = SupplierStatus.ACTIVE
    rating: Optional[condecimal(max_digits=3, decimal_places=2, ge=0, le=5)] = None
    certifications: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SupplierUpdate(BaseModel):
    """Update supplier request"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    category: Optional[str] = Field(None, max_length=100)
    status: Optional[SupplierStatus] = None
    tier: Optional[int] = Field(None, ge=1, le=5)
    country: Optional[str] = Field(None, min_length=2, max_length=2)
    city: Optional[str] = Field(None, max_length=100)
    address: Optional[str] = None
    contact_name: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    contact_phone: Optional[str] = Field(None, max_length=50)
    payment_terms: Optional[str] = Field(None, max_length=100)
    lead_time_days: Optional[int] = Field(None, ge=0)
    minimum_order_value: Optional[condecimal(max_digits=15, decimal_places=2)] = None
    rating: Optional[condecimal(max_digits=3, decimal_places=2, ge=0, le=5)] = None
    certifications: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SupplierResponse(SupplierBase):
    """Supplier response schema"""
    id: UUID
    status: SupplierStatus
    rating: Optional[float] = None
    certifications: List[str] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    performance_score: Optional[float] = None
    risk_level: Optional[str] = None
    
    class Config:
        from_attributes = True  # Changed from orm_mode
        use_enum_values = True


class SupplierTierUpdate(BaseModel):
    """Update supplier tier"""
    supplier_id: UUID
    tier_level: int = Field(..., ge=1, le=5)
    classification: str
    verified: bool = False
    notes: Optional[str] = None


class SupplierRelationshipCreate(BaseModel):
    """Create supplier relationship"""
    parent_supplier_id: UUID
    child_supplier_id: UUID
    relationship_type: str
    strength: Optional[float] = Field(None, ge=0, le=1)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =====================================================
# Product and Material Schemas
# =====================================================

class ProductBase(BaseModel):
    """Base product schema"""
    sku: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = Field(None, max_length=100)
    subcategory: Optional[str] = Field(None, max_length=100)
    unit_of_measure: Optional[str] = Field(None, max_length=50)
    weight_kg: Optional[condecimal(max_digits=10, decimal_places=3)] = None
    volume_m3: Optional[condecimal(max_digits=10, decimal_places=3)] = None
    unit_cost: Optional[condecimal(max_digits=15, decimal_places=2)] = None
    selling_price: Optional[condecimal(max_digits=15, decimal_places=2)] = None


class ProductCreate(ProductBase):
    """Create product request"""
    status: str = "active"
    launch_date: Optional[date] = None
    discontinue_date: Optional[date] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ProductUpdate(BaseModel):
    """Update product request"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = Field(None, max_length=100)
    subcategory: Optional[str] = Field(None, max_length=100)
    unit_of_measure: Optional[str] = Field(None, max_length=50)
    weight_kg: Optional[condecimal(max_digits=10, decimal_places=3)] = None
    volume_m3: Optional[condecimal(max_digits=10, decimal_places=3)] = None
    unit_cost: Optional[condecimal(max_digits=15, decimal_places=2)] = None
    selling_price: Optional[condecimal(max_digits=15, decimal_places=2)] = None
    status: Optional[str] = None
    launch_date: Optional[date] = None
    discontinue_date: Optional[date] = None
    attributes: Optional[Dict[str, Any]] = None


class ProductResponse(ProductBase):
    """Product response schema"""
    id: UUID
    status: str
    launch_date: Optional[date] = None
    discontinue_date: Optional[date] = None
    attributes: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    margin: Optional[float] = None
    is_active: bool
    
    class Config:
        from_attributes = True  # Changed from orm_mode


# schemas/supply_chain.py (continued)

class MaterialBase(BaseModel):
    """Base material schema"""
    code: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    type: Optional[str] = Field(None, max_length=100)
    unit_of_measure: Optional[str] = Field(None, max_length=50)
    unit_cost: Optional[condecimal(max_digits=15, decimal_places=2)] = None
    lead_time_days: Optional[int] = Field(None, ge=0)
    minimum_order_quantity: Optional[condecimal(max_digits=15, decimal_places=3)] = None


class MaterialCreate(MaterialBase):
    """Create material request"""
    hazmat: bool = False
    perishable: bool = False
    specifications: Dict[str, Any] = Field(default_factory=dict)


class MaterialUpdate(BaseModel):
    """Update material request"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    type: Optional[str] = Field(None, max_length=100)
    unit_of_measure: Optional[str] = Field(None, max_length=50)
    unit_cost: Optional[condecimal(max_digits=15, decimal_places=2)] = None
    lead_time_days: Optional[int] = Field(None, ge=0)
    minimum_order_quantity: Optional[condecimal(max_digits=15, decimal_places=3)] = None
    hazmat: Optional[bool] = None
    perishable: Optional[bool] = None
    specifications: Optional[Dict[str, Any]] = None


class MaterialResponse(MaterialBase):
    """Material response schema"""
    id: UUID
    hazmat: bool
    perishable: bool
    specifications: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True  # Changed from orm_mode


class BillOfMaterials(BaseModel):
    """Bill of materials entry"""
    product_id: UUID
    material_id: UUID
    quantity: condecimal(max_digits=15, decimal_places=3) = Field(..., gt=0)
    unit_of_measure: Optional[str] = Field(None, max_length=50)
    scrap_percentage: Optional[condecimal(max_digits=5, decimal_places=2)] = Field(None, ge=0, le=100)
    notes: Optional[str] = None


# =====================================================
# Inventory Schemas
# =====================================================

class InventoryBase(BaseModel):
    """Base inventory schema"""
    location_id: str = Field(..., min_length=1, max_length=100)
    product_id: Optional[UUID] = None
    material_id: Optional[UUID] = None
    quantity_on_hand: condecimal(max_digits=15, decimal_places=3) = Field(default=0)
    quantity_available: condecimal(max_digits=15, decimal_places=3) = Field(default=0)
    quantity_allocated: condecimal(max_digits=15, decimal_places=3) = Field(default=0)
    quantity_in_transit: condecimal(max_digits=15, decimal_places=3) = Field(default=0)
    reorder_point: Optional[condecimal(max_digits=15, decimal_places=3)] = None
    reorder_quantity: Optional[condecimal(max_digits=15, decimal_places=3)] = None
    safety_stock: Optional[condecimal(max_digits=15, decimal_places=3)] = None


class InventoryCreate(InventoryBase):
    """Create inventory request"""
    lot_number: Optional[str] = Field(None, max_length=100)
    expiry_date: Optional[date] = None
    cost_per_unit: Optional[condecimal(max_digits=15, decimal_places=2)] = None
    
    @field_validator('material_id')
    @classmethod
    def validate_item_reference(cls, v, info):
        """Validate that either product_id or material_id is specified, but not both"""
        product_id = info.data.get('product_id')
        if product_id and v:
            raise ValueError("Cannot specify both product_id and material_id")
        if not product_id and not v:
            raise ValueError("Must specify either product_id or material_id")
        return v


class InventoryUpdate(BaseModel):
    """Update inventory request"""
    quantity_on_hand: Optional[condecimal(max_digits=15, decimal_places=3)] = None
    quantity_allocated: Optional[condecimal(max_digits=15, decimal_places=3)] = None
    quantity_in_transit: Optional[condecimal(max_digits=15, decimal_places=3)] = None
    reorder_point: Optional[condecimal(max_digits=15, decimal_places=3)] = None
    reorder_quantity: Optional[condecimal(max_digits=15, decimal_places=3)] = None
    safety_stock: Optional[condecimal(max_digits=15, decimal_places=3)] = None
    lot_number: Optional[str] = Field(None, max_length=100)
    expiry_date: Optional[date] = None
    cost_per_unit: Optional[condecimal(max_digits=15, decimal_places=2)] = None


class InventoryResponse(InventoryBase):
    """Inventory response schema"""
    id: UUID
    lot_number: Optional[str] = None
    expiry_date: Optional[date] = None
    cost_per_unit: Optional[float] = None
    last_counted_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    item_name: Optional[str] = None
    item_sku: Optional[str] = None
    
    class Config:
        from_attributes = True  # Changed from orm_mode


class InventoryAdjustment(BaseModel):
    """Inventory adjustment request"""
    inventory_id: UUID
    adjustment_quantity: condecimal(max_digits=15, decimal_places=3)
    adjustment_type: str = Field(..., pattern="^(manual|cycle_count|damage|theft|found)$")  # Changed from regex
    reason: str = Field(..., min_length=1)
    reference_number: Optional[str] = Field(None, max_length=100)


class InventoryTransfer(BaseModel):
    """Inventory transfer request"""
    from_location: str
    to_location: str
    product_id: Optional[UUID] = None
    material_id: Optional[UUID] = None
    quantity: condecimal(max_digits=15, decimal_places=3) = Field(..., gt=0)
    transfer_date: Optional[datetime] = None
    notes: Optional[str] = None


class InventoryReservation(BaseModel):
    """Inventory reservation request"""
    inventory_id: UUID
    reserved_quantity: condecimal(max_digits=15, decimal_places=3) = Field(..., gt=0)
    reserved_for: str
    reservation_type: str
    expires_at: Optional[datetime] = None
    notes: Optional[str] = None


# =====================================================
# Order Schemas
# =====================================================

class OrderBase(BaseModel):
    """Base order schema"""
    order_number: str = Field(..., min_length=1, max_length=100)
    type: str = Field(..., pattern="^(purchase|sales|transfer|return)$")  # Changed from regex
    supplier_id: Optional[UUID] = None
    customer_id: Optional[UUID] = None
    order_date: datetime
    requested_delivery_date: Optional[datetime] = None
    shipping_address: Optional[str] = None
    billing_address: Optional[str] = None
    currency: str = Field(default="USD", min_length=3, max_length=3)
    payment_terms: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = None


class OrderCreate(OrderBase):
    """Create order request"""
    status: OrderStatus = OrderStatus.DRAFT
    priority: str = Field(default="normal", pattern="^(low|normal|high|urgent)$")  # Changed from regex
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrderUpdate(BaseModel):
    """Update order request"""
    status: Optional[OrderStatus] = None
    requested_delivery_date: Optional[datetime] = None
    shipping_address: Optional[str] = None
    billing_address: Optional[str] = None
    payment_terms: Optional[str] = Field(None, max_length=100)
    priority: Optional[str] = Field(None, pattern="^(low|normal|high|urgent)$")  # Changed from regex
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class OrderResponse(OrderBase):
    """Order response schema"""
    id: UUID
    status: OrderStatus
    priority: str
    total_amount: float
    tax_amount: float
    shipping_amount: float
    discount_amount: float
    net_amount: float
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    created_by_id: UUID
    items_count: int = 0
    
    class Config:
        from_attributes = True  # Changed from orm_mode
        use_enum_values = True


class OrderItemBase(BaseModel):
    """Base order item schema"""
    product_id: Optional[UUID] = None
    material_id: Optional[UUID] = None
    quantity_ordered: condecimal(max_digits=15, decimal_places=3) = Field(..., gt=0)
    unit_price: condecimal(max_digits=15, decimal_places=2) = Field(..., ge=0)
    discount_percentage: condecimal(max_digits=5, decimal_places=2) = Field(default=0, ge=0, le=100)
    tax_percentage: condecimal(max_digits=5, decimal_places=2) = Field(default=0, ge=0)


class OrderItemCreate(OrderItemBase):
    """Create order item request"""
    order_id: UUID
    line_number: Optional[int] = None
    notes: Optional[str] = None


class OrderItemResponse(OrderItemBase):
    """Order item response schema"""
    id: UUID
    order_id: UUID
    line_number: int
    quantity_shipped: float
    quantity_received: float
    quantity_invoiced: float
    line_total: float
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    item_name: Optional[str] = None
    item_sku: Optional[str] = None
    
    class Config:
        from_attributes = True  # Changed from orm_mode


# =====================================================
# Shipment Schemas
# =====================================================

class ShipmentBase(BaseModel):
    """Base shipment schema"""
    shipment_number: str = Field(..., min_length=1, max_length=100)
    order_id: UUID
    carrier: Optional[str] = Field(None, max_length=255)
    tracking_number: Optional[str] = Field(None, max_length=255)
    ship_date: Optional[datetime] = None
    estimated_delivery_date: Optional[datetime] = None
    actual_delivery_date: Optional[datetime] = None
    shipping_cost: Optional[condecimal(max_digits=15, decimal_places=2)] = None
    weight_kg: Optional[condecimal(max_digits=10, decimal_places=3)] = None
    volume_m3: Optional[condecimal(max_digits=10, decimal_places=3)] = None


class ShipmentCreate(ShipmentBase):
    """Create shipment request"""
    status: ShipmentStatus = ShipmentStatus.PENDING
    from_location: str
    to_location: str
    notes: Optional[str] = None


class ShipmentUpdate(BaseModel):
    """Update shipment request"""
    status: Optional[ShipmentStatus] = None
    carrier: Optional[str] = Field(None, max_length=255)
    tracking_number: Optional[str] = Field(None, max_length=255)
    ship_date: Optional[datetime] = None
    estimated_delivery_date: Optional[datetime] = None
    actual_delivery_date: Optional[datetime] = None
    shipping_cost: Optional[condecimal(max_digits=15, decimal_places=2)] = None
    notes: Optional[str] = None


class ShipmentResponse(ShipmentBase):
    """Shipment response schema"""
    id: UUID
    status: ShipmentStatus
    from_location: str
    to_location: str
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    items_count: int = 0
    
    class Config:
        from_attributes = True  # Changed from orm_mode
        use_enum_values = True


class ShipmentItemCreate(BaseModel):
    """Create shipment item request"""
    shipment_id: UUID
    order_item_id: UUID
    quantity_shipped: condecimal(max_digits=15, decimal_places=3) = Field(..., gt=0)
    lot_number: Optional[str] = Field(None, max_length=100)
    serial_numbers: List[str] = Field(default_factory=list)


# =====================================================
# Analytics and Metrics Schemas
# =====================================================

class SupplierPerformanceMetrics(BaseModel):
    """Supplier performance metrics"""
    supplier_id: UUID
    period_start: date
    period_end: date
    on_time_delivery_rate: float
    quality_score: float
    responsiveness_score: float
    price_competitiveness: float
    overall_score: float
    total_orders: int
    total_spend: float
    defect_rate: float
    lead_time_variance: float
    
    class Config:
        from_attributes = True  # Changed from orm_mode


class SupplierPerformanceUpdate(BaseModel):
    """Update supplier performance metrics"""
    on_time_delivery_rate: Optional[float] = None
    quality_score: Optional[float] = None
    responsiveness_score: Optional[float] = None
    price_competitiveness: Optional[float] = None
    overall_score: Optional[float] = None
    defect_rate: Optional[float] = None
    lead_time_variance: Optional[float] = None


class InventoryMetric(BaseModel):
    """Inventory metrics"""
    location_id: str
    product_id: Optional[UUID] = None
    material_id: Optional[UUID] = None
    period_start: date
    period_end: date
    average_inventory: float
    turnover_ratio: float
    stockout_days: int
    carrying_cost: float
    obsolescence_cost: float
    service_level: float
    
    class Config:
        from_attributes = True  # Changed from orm_mode


class ComplianceCheckCreate(BaseModel):
    """Create compliance check"""
    supplier_id: UUID
    check_type: str
    findings: List[str] = []
    recommendations: List[str] = []
    score: Optional[float] = None
    next_check_date: Optional[date] = None


class ComplianceCheck(BaseModel):
    """Compliance check result"""
    supplier_id: UUID
    check_type: str
    check_date: datetime
    status: ComplianceStatus
    score: Optional[float] = None
    findings: List[str] = []
    recommendations: List[str] = []
    next_check_date: Optional[date] = None
    
    class Config:
        from_attributes = True  # Changed from orm_mode
        use_enum_values = True


class RiskAssessment(BaseModel):
    """Risk assessment result"""
    supplier_id: UUID
    assessment_date: datetime
    risk_level: str
    risk_score: float
    risk_factors: Dict[str, float]
    mitigation_strategies: List[str]
    review_date: Optional[date] = None
    
    class Config:
        from_attributes = True  # Changed from orm_mode


# =====================================================
# Network Analysis and Simulation Schemas (NEW)
# =====================================================

class NetworkGraphResponse(BaseModel):
    """Supply chain network graph response"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    bottlenecks: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class RiskAnalysisResponse(BaseModel):
    """Risk analysis response with propagation paths"""
    risk_scores: Dict[str, float]
    propagation_paths: List[Dict[str, Any]]
    impact_metrics: Dict[str, Any]
    recommendations: List[str]
    analysis_timestamp: datetime


class ScenarioSimulationRequest(BaseModel):
    """Request for scenario simulation"""
    disrupted_suppliers: List[UUID]
    severity: float
    duration_days: int
    disruption_type: str
    target_recovery_time: Optional[int] = None


class ScenarioSimulationResponse(BaseModel):
    """Scenario simulation results"""
    scenario_id: str
    affected_suppliers: List[Dict[str, Any]]
    supply_impact: Dict[str, Any]
    financial_impact: Dict[str, Any]
    recovery_strategies: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]


class SupplierTierResponse(BaseModel):
    """Supplier tier information response"""
    supplier_id: UUID
    supplier_name: str
    tier_level: int
    tier_score: float
    connections_count: int
    risk_level: str


# =====================================================
# Aggregated Response Schemas
# =====================================================

class SupplierWithMetrics(SupplierResponse):
    """Supplier with performance metrics"""
    latest_performance: Optional[SupplierPerformanceMetrics] = None
    risk_assessment: Optional[RiskAssessment] = None
    compliance_status: Optional[ComplianceCheck] = None


class ProductWithInventory(ProductResponse):
    """Product with inventory summary"""
    total_inventory: float = 0
    locations_count: int = 0
    inventory_value: float = 0
    stockout_risk: Optional[str] = None


class OrderWithDetails(OrderResponse):
    """Order with full details"""
    items: List[OrderItemResponse] = []
    shipments: List[ShipmentResponse] = []
    supplier: Optional[SupplierResponse] = None


class InventoryMovement(BaseModel):
    """Inventory movement record"""
    movement_type: str
    reference_type: str
    reference_id: UUID
    product_id: Optional[UUID] = None
    material_id: Optional[UUID] = None
    location_id: str
    quantity: float
    movement_date: datetime
    description: str
    
    class Config:
        from_attributes = True  # Changed from orm_mode