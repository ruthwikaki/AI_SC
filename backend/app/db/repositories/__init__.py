"""Repository package with lazy loading"""

# Lazy loading repositories to prevent circular imports
_repo_cache = {}

def get_user_repository():
    if 'user' not in _repo_cache:
        from .user_repository import UserRepository
        _repo_cache['user'] = UserRepository
    return _repo_cache['user']

def get_inventory_repository():
    if 'inventory' not in _repo_cache:
        from .inventory_repository import InventoryRepository
        _repo_cache['inventory'] = InventoryRepository
    return _repo_cache['inventory']

def get_order_repository():
    if 'order' not in _repo_cache:
        from .order_repository import OrderRepository
        _repo_cache['order'] = OrderRepository
    return _repo_cache['order']

def get_supplier_repository():
    if 'supplier' not in _repo_cache:
        from .supplier_repository import SupplierRepository
        _repo_cache['supplier'] = SupplierRepository
    return _repo_cache['supplier']

def get_analytics_repository():
    if 'analytics' not in _repo_cache:
        from .analytics_repository import AnalyticsRepository
        _repo_cache['analytics'] = AnalyticsRepository
    return _repo_cache['analytics']

def get_dashboard_repository():
    if 'dashboard' not in _repo_cache:
        from .dashboard_repository import DashboardRepository
        _repo_cache['dashboard'] = DashboardRepository
    return _repo_cache['dashboard']

def get_query_repository():
    if 'query' not in _repo_cache:
        from .query_repository import QueryRepository
        _repo_cache['query'] = QueryRepository
    return _repo_cache['query']

__all__ = [
    'get_user_repository', 'get_inventory_repository', 'get_order_repository',
    'get_supplier_repository', 'get_analytics_repository', 'get_dashboard_repository',
    'get_query_repository'
]
