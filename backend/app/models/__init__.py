"""Models package - NO direct imports to prevent circular dependencies"""

# Only import the base
from .base import Base, get_db

# DO NOT import any models directly here!
# Use the lazy loading functions below

_models_cache = {}

def _get_model(module_name, model_name):
    """Universal lazy model getter"""
    cache_key = f"{module_name}.{model_name}"
    if cache_key not in _models_cache:
        # Dynamic import
        module = __import__(f'app.models.{module_name}', fromlist=[model_name])
        _models_cache[cache_key] = getattr(module, model_name)
    return _models_cache[cache_key]

# User models
def get_user_model():
    return _get_model('user', 'User')

# Supply chain models
def get_product_model():
    return _get_model('supply_chain', 'Product')

def get_supplier_model():
    return _get_model('supply_chain', 'Supplier')

def get_order_model():
    return _get_model('supply_chain', 'Order')

def get_inventory_model():
    return _get_model('supply_chain', 'Inventory')

# Analytics models
def get_dashboard_model():
    return _get_model('analytics', 'Dashboard')

def get_chart_model():
    return _get_model('analytics', 'Chart')

def get_query_model():
    return _get_model('analytics', 'Query')

def get_report_model():
    return _get_model('analytics', 'Report')

# Extended models - DO NOT IMPORT DIRECTLY
def get_extended_product_model():
    return _get_model('extended_models', 'ExtendedProduct')

def get_extended_supplier_model():
    return _get_model('extended_models', 'ExtendedSupplier')

def get_extended_order_model():
    return _get_model('extended_models', 'ExtendedOrder')

def get_extended_report_model():
    return _get_model('extended_models', 'ExtendedReport')

# System models
def get_system_config_model():
    return _get_model('system', 'SystemConfig')

def get_audit_log_model():
    return _get_model('system', 'AuditLog')

# Visualization models
def get_visualization_model():
    return _get_model('visualization', 'Visualization')

def get_chart_config_model():
    return _get_model('visualization', 'ChartConfig')

# Query models
def get_query_history_model():
    return _get_model('query', 'QueryHistory')

# Export only functions, never models directly
__all__ = [
    'Base', 'get_db',
    # User
    'get_user_model',
    # Supply chain
    'get_product_model', 'get_supplier_model', 'get_order_model', 'get_inventory_model',
    # Analytics
    'get_dashboard_model', 'get_chart_model', 'get_query_model', 'get_report_model',
    # Extended
    'get_extended_product_model', 'get_extended_supplier_model', 
    'get_extended_order_model', 'get_extended_report_model',
    # System
    'get_system_config_model', 'get_audit_log_model',
    # Visualization
    'get_visualization_model', 'get_chart_config_model',
    # Query
    'get_query_history_model'
]
