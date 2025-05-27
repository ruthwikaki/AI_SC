# jobs/__init__.py

"""
Background jobs for SSA application
"""

from .scheduler import (
    JobScheduler,
    schedule_analytics_update,
    schedule_report_generation,
    schedule_data_sync,
    schedule_cleanup,
    start_scheduler,
    stop_scheduler
)

from .data_sync import (
    sync_supplier_data,
    sync_inventory_data,
    sync_order_data,
    run_full_sync,
    check_sync_status
)

from .analytics_update import (
    update_supplier_metrics,
    update_inventory_metrics,
    update_delivery_performance,
    calculate_abc_analysis,
    update_risk_assessments,
    run_all_analytics
)

from .cleanup import (
    cleanup_old_logs,
    cleanup_expired_sessions,
    cleanup_old_analytics,
    archive_old_data,
    vacuum_database
)

__all__ = [
    # Scheduler
    'JobScheduler',
    'start_scheduler',
    'stop_scheduler',
    
    # Data sync
    'sync_supplier_data',
    'sync_inventory_data',
    'sync_order_data',
    'run_full_sync',
    
    # Analytics
    'update_supplier_metrics',
    'update_inventory_metrics',
    'run_all_analytics',
    
    # Cleanup
    'cleanup_old_logs',
    'cleanup_expired_sessions',
    'vacuum_database'
]