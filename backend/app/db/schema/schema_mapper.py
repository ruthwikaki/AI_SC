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
_domain_mapping_cache: Dict[str, Dict[str, Any]] = {}

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

def get_domain_mappings(client_id: str, connection_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get domain concept mappings for a specific client and connection.
    
    Args:
        client_id: Client ID
        connection_id: Optional connection ID
        
    Returns:
        Dictionary of domain mappings
    """
    global _domain_mapping_cache
    
    # Create cache key
    cache_key = f"{client_id}:{connection_id or 'default'}"
    
    # Check if mappings are in cache
    if cache_key in _domain_mapping_cache:
        return _domain_mapping_cache[cache_key]
    
    # For now, return the predefined supply chain concepts
    # In a real implementation, this would fetch client-specific mappings
    # from a database or configuration store
    mappings = {
        "concepts": SUPPLY_CHAIN_CONCEPTS,
        "table_mappings": {
            # Example table mappings - these would be customized per client in production
            "products": {
                "domain": "product",
                "column_mappings": {
                    "id": "product_id",
                    "name": "product_name",
                    "sku": "product_sku",
                    "description": "product_description",
                    "price": "product_price"
                }
            },
            "inventory": {
                "domain": "inventory",
                "column_mappings": {
                    "id": "inventory_id",
                    "product_id": "product_id",
                    "quantity": "inventory_quantity",
                    "location_id": "warehouse_id"
                }
            },
            "suppliers": {
                "domain": "supplier",
                "column_mappings": {
                    "id": "supplier_id",
                    "name": "supplier_name",
                    "contact": "supplier_contact",
                    "address": "supplier_address",
                    "rating": "performance_rating"
                }
            }
        }
    }
    
    # Cache the result
    _domain_mapping_cache[cache_key] = mappings
    
    logger.debug(f"Generated domain mappings for {client_id}")
    return mappings

async def update_domain_mappings(
    client_id: str,
    mappings: Dict[str, Any],
    connection_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update domain concept mappings for a specific client and connection.
    
    Args:
        client_id: Client ID
        mappings: Dictionary of domain mappings to update
        connection_id: Optional connection ID
        
    Returns:
        Updated dictionary of domain mappings
    """
    global _domain_mapping_cache
    
    # Create cache key
    cache_key = f"{client_id}:{connection_id or 'default'}"
    
    try:
        # In a production environment, this would persist the mappings to a database
        # For now, just update the in-memory cache
        
        # Validate the mappings structure
        if not isinstance(mappings, dict):
            raise ValueError("Mappings must be a dictionary")
        
        # Update the cache with the new mappings
        if cache_key in _domain_mapping_cache:
            # Update existing mappings
            existing_mappings = _domain_mapping_cache[cache_key]
            
            # Update concepts
            if "concepts" in mappings:
                existing_mappings["concepts"].update(mappings["concepts"])
            
            # Update table mappings
            if "table_mappings" in mappings:
                if "table_mappings" not in existing_mappings:
                    existing_mappings["table_mappings"] = {}
                existing_mappings["table_mappings"].update(mappings["table_mappings"])
        else:
            # Create new mappings entry
            _domain_mapping_cache[cache_key] = mappings
        
        logger.info(f"Updated domain mappings for {client_id}")
        return _domain_mapping_cache[cache_key]
        
    except Exception as e:
        logger.error(f"Error updating domain mappings for {client_id}: {str(e)}")
        raise