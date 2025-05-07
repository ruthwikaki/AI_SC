from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import asyncio
import re
import json

from app.utils.logger import get_logger
from app.config import get_settings
from app.llm.controller.active_model_manager import get_active_model
from app.llm.prompt.template_manager import get_template

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Domain concept mapping cache
_domain_mapping_cache: Dict[str, List[Dict[str, Any]]] = {}

# Supply chain domain concepts
SUPPLY_CHAIN_CONCEPTS = {
    # Inventory concepts
    "inventory": {
        "description": "Current stock levels of products",
        "attributes": ["product", "quantity", "location", "unit_of_measure"]
    },
    "product": {
        "description": "Item or material that is sold, manufactured, or stored",
        "attributes": ["name", "description", "sku", "category", "subcategory", "price", "cost"]
    },
    "warehouse": {
        "description": "Location where inventory is stored",
        "attributes": ["name", "address", "capacity", "type"]
    },
    "stock_level": {
        "description": "Current inventory level at a specific location",
        "attributes": ["product", "warehouse", "quantity", "min_level", "max_level", "reorder_point"]
    },
    
    # Supplier concepts
    "supplier": {
        "description": "Entity that provides products or services",
        "attributes": ["name", "contact", "address", "performance_rating", "tier"]
    },
    "manufacturer": {
        "description": "Entity that creates or assembles products",
        "attributes": ["name", "capabilities", "capacity", "location"]
    },
    "vendor": {
        "description": "Business entity that sells products",
        "attributes": ["name", "type", "specialty", "reliability"]
    },
    
    # Order concepts
    "purchase_order": {
        "description": "Request to purchase products from suppliers",
        "attributes": ["order_number", "supplier", "order_date", "expected_delivery", "status", "total_amount"]
    },
    "sales_order": {
        "description": "Customer order for products",
        "attributes": ["order_number", "customer", "order_date", "status", "delivery_date", "total_amount"]
    },
    "order_line": {
        "description": "Individual line item in an order",
        "attributes": ["order", "product", "quantity", "price", "discount", "line_amount"]
    },
    
    # Logistics concepts
    "shipment": {
        "description": "Movement of goods from one location to another",
        "attributes": ["shipment_id", "origin", "destination", "carrier", "tracking_number", "status", "estimated_arrival"]
    },
    "delivery": {
        "description": "Final delivery of goods to customer or location",
        "attributes": ["delivery_id", "order", "address", "status", "delivery_date", "recipient"]
    },
    "carrier": {
        "description": "Transportation provider that moves goods",
        "attributes": ["name", "service_type", "cost_structure", "reliability_score"]
    },
    "route": {
        "description": "Path taken for transportation of goods",
        "attributes": ["origin", "destination", "distance", "estimated_time", "mode_of_transport"]
    },
    
    # Planning concepts
    "forecast": {
        "description": "Prediction of future demand or requirements",
        "attributes": ["product", "period", "quantity", "confidence_level", "source"]
    },
    "production_schedule": {
        "description": "Plan for manufacturing products",
        "attributes": ["product", "quantity", "start_date", "end_date", "facility", "status"]
    },
    "safety_stock": {
        "description": "Buffer inventory to account for variability",
        "attributes": ["product", "location", "quantity", "min_days_coverage", "max_days_coverage"]
    }
}