"""Models package - Using lazy imports only"""

# Base imports
from .base import Base, get_db

# Model registry for lazy loading
_model_registry = {}

def _lazy_import(module_name, model_name):
    """Lazy import models to avoid circular dependencies"""
    key = f"{module_name}.{model_name}"
    if key not in _model_registry:
        module = __import__(f"app.models.{module_name}", fromlist=[model_name])
        _model_registry[key] = getattr(module, model_name)
    return _model_registry[key]

# Lazy getters for all models
def get_user():
    return _lazy_import("user", "User")

def get_product():
    return _lazy_import("supply_chain", "Product")

def get_supplier():
    return _lazy_import("supply_chain", "Supplier")

def get_order():
    return _lazy_import("supply_chain", "Order")

def get_inventory():
    return _lazy_import("supply_chain", "Inventory")

def get_dashboard():
    return _lazy_import("analytics", "Dashboard")

def get_chart():
    return _lazy_import("analytics", "Chart")

def get_query():
    return _lazy_import("analytics", "Query")

def get_report():
    return _lazy_import("analytics", "Report")

def get_analytics_metric():
    return _lazy_import("analytics", "AnalyticsMetric")

# DO NOT import extended_models directly!
def get_extended_product():
    return _lazy_import("extended_models", "ExtendedProduct")

def get_extended_supplier():
    return _lazy_import("extended_models", "ExtendedSupplier")

def get_extended_order():
    return _lazy_import("extended_models", "ExtendedOrder")

def get_extended_report():
    return _lazy_import("extended_models", "ExtendedReport")

def get_extended_analytics_metric():
    return _lazy_import("extended_models", "ExtendedAnalyticsMetric")

# System models
def get_system_config():
    return _lazy_import("system", "SystemConfig")

def get_audit_log():
    return _lazy_import("system", "AuditLog")

# Visualization models
def get_visualization():
    return _lazy_import("visualization", "Visualization")

def get_chart_config():
    return _lazy_import("visualization", "ChartConfig")

# Query history
def get_query_history():
    return _lazy_import("query", "QueryHistory")

__all__ = [
    "Base", "get_db",
    # User
    "get_user",
    # Supply chain
    "get_product", "get_supplier", "get_order", "get_inventory",
    # Analytics
    "get_dashboard", "get_chart", "get_query", "get_report", "get_analytics_metric",
    # Extended (lazy only!)
    "get_extended_product", "get_extended_supplier", "get_extended_order",
    "get_extended_report", "get_extended_analytics_metric",
    # System
    "get_system_config", "get_audit_log",
    # Visualization
    "get_visualization", "get_chart_config",
    # Query
    "get_query_history"
]
