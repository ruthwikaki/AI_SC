"""Repository package - ALL imports must be lazy to prevent circular dependencies"""

# Cache for lazy-loaded repositories
_repository_cache = {}

def _get_repository(repo_name, class_name):
    """Generic lazy repository loader"""
    if repo_name not in _repository_cache:
        # Dynamic import at runtime
        module = __import__(f'app.db.repositories.{repo_name}', fromlist=[class_name])
        _repository_cache[repo_name] = getattr(module, class_name)
    return _repository_cache[repo_name]

# User repository
def get_user_repository():
    """Get UserRepository lazily"""
    return _get_repository('user_repository', 'UserRepository')

# Inventory repository
def get_inventory_repository():
    """Get InventoryRepository lazily"""
    return _get_repository('inventory_repository', 'InventoryRepository')

# Order repository
def get_order_repository():
    """Get OrderRepository lazily"""
    return _get_repository('order_repository', 'OrderRepository')

# Supplier repository
def get_supplier_repository():
    """Get SupplierRepository lazily"""
    return _get_repository('supplier_repository', 'SupplierRepository')

# Analytics repository
def get_analytics_repository():
    """Get AnalyticsRepository lazily"""
    return _get_repository('analytics_repository', 'AnalyticsRepository')

# Dashboard repository
def get_dashboard_repository():
    """Get DashboardRepository lazily"""
    return _get_repository('dashboard_repository', 'DashboardRepository')

# Query repository
def get_query_repository():
    """Get QueryRepository lazily"""
    return _get_repository('query_repository', 'QueryRepository')

# Chart repository
def get_chart_repository():
    """Get ChartRepository lazily"""
    return _get_repository('chart_repository', 'ChartRepository')

# NO DIRECT IMPORTS! Only export the getter functions
__all__ = [
    'get_user_repository',
    'get_inventory_repository', 
    'get_order_repository',
    'get_supplier_repository',
    'get_analytics_repository',
    'get_dashboard_repository',
    'get_query_repository',
    'get_chart_repository'
]
