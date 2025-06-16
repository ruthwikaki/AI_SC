"""Models package - complete lazy loading to prevent ALL circular imports"""

# Base imports only
from .base import Base, get_db

# Lazy loading for ALL models to prevent circular imports
_models_cache = {}

def _lazy_import(module_name, class_names):
    """Generic lazy importer to prevent circular imports"""
    if module_name not in _models_cache:
        module = __import__(f'app.models.{module_name}', fromlist=class_names)
        _models_cache[module_name] = {name: getattr(module, name) for name in class_names}
    return _models_cache[module_name]

# User models
def get_user_model():
    return _lazy_import('user', ['User'])['User']

# Supply chain models
def get_product_model():
    return _lazy_import('supply_chain', ['Product'])['Product']

def get_supplier_model():
    return _lazy_import('supply_chain', ['Supplier'])['Supplier']

def get_order_model():
    return _lazy_import('supply_chain', ['Order'])['Order']

def get_inventory_model():
    return _lazy_import('supply_chain', ['Inventory'])['Inventory']

# Analytics models
def get_dashboard_model():
    return _lazy_import('analytics', ['Dashboard'])['Dashboard']

def get_chart_model():
    return _lazy_import('analytics', ['Chart'])['Chart']

def get_query_model():
    return _lazy_import('analytics', ['Query'])['Query']

def get_report_model():
    return _lazy_import('analytics', ['Report'])['Report']

# Extended models
def get_extended_models():
    models = _lazy_import('extended_models', ['ExtendedProduct', 'ExtendedSupplier', 'ExtendedOrder', 'ExtendedReport'])
    return models

# System models
def get_system_config_model():
    return _lazy_import('system', ['SystemConfig'])['SystemConfig']

def get_audit_log_model():
    return _lazy_import('system', ['AuditLog'])['AuditLog']

# Visualization models
def get_visualization_model():
    return _lazy_import('visualization', ['Visualization'])['Visualization']

def get_chart_config_model():
    return _lazy_import('visualization', ['ChartConfig'])['ChartConfig']

# Query models
def get_query_history_model():
    return _lazy_import('query', ['QueryHistory'])['QueryHistory']

__all__ = [
    'Base', 'get_db',
    'get_user_model', 'get_product_model', 'get_supplier_model',
    'get_order_model', 'get_inventory_model', 'get_dashboard_model',
    'get_chart_model', 'get_query_model', 'get_report_model',
    'get_extended_models', 'get_system_config_model', 'get_audit_log_model',
    'get_visualization_model', 'get_chart_config_model', 'get_query_history_model'
]
