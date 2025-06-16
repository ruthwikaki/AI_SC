# jobs/cleanup.py

"""
Data cleanup and maintenance jobs
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import and_, text

from app.db.database import get_db_session, engine
from app.config import get_settings
from app.models import (
    UserSession, 
    AnalyticsResult,
    QueryResultCache, 
    SyncHistory
)

# Get settings
# Lazy load settings to avoid circular import
_settings = None

def get_settings_cached():
    global _settings
    if _settings is None:
        from ..config import get_settings
        _settings = get_settings()
    return _settings

settings = property(lambda self: get_settings_cached())

logger = logging.getLogger(__name__)


@scheduled_job(name="cleanup_expired_sessions", description="Remove expired user sessions")
async def cleanup_expired_sessions():
    """Remove expired user sessions from database"""
    with get_db_session() as db:
        # Get expired sessions
        expired_count = db.query(UserSession).filter(
            UserSession.expires_at < datetime.utcnow()
        ).delete()
        
        db.commit()
        logger.info(f"Cleaned up {expired_count} expired sessions")
        return expired_count


@scheduled_job(name="cleanup_old_analytics", description="Remove old analytics results")
async def cleanup_old_analytics(retention_days: int = 90):
    """Clean up old analytics results"""
    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
    
    with get_db_session() as db:
        # Analytics results
        analytics_count = db.query(AnalyticsResult).filter(
            AnalyticsResult.created_at < cutoff_date
        ).delete()
        
        # Query result cache
        cache_count = db.query(QueryResultCache).filter(
            QueryResultCache.created_at < cutoff_date
        ).delete()
        
        # Old sync history
        sync_count = db.query(SyncHistory).filter(
            SyncHistory.completed_at < cutoff_date
        ).delete()
        
        db.commit()
        
        logger.info(f"Cleaned up old analytics: {analytics_count} results, "
                   f"{cache_count} cached queries, {sync_count} sync records")
        
        return {
            "analytics_results": analytics_count,
            "cached_queries": cache_count,
            "sync_history": sync_count
        }


@scheduled_job(name="archive_old_data", description="Archive old transactional data")
async def archive_old_data(archive_days: int = 365):
    """Archive old transactional data to archive tables"""
    cutoff_date = datetime.utcnow() - timedelta(days=archive_days)
    
    with engine.begin() as conn:
        try:
            # Archive old orders
            archive_orders = conn.execute(text("""
                INSERT INTO archived_orders 
                SELECT * FROM orders 
                WHERE order_date < :cutoff_date
                AND status IN ('closed', 'cancelled')
                ON CONFLICT (id) DO NOTHING
            """), {"cutoff_date": cutoff_date})
            
            orders_archived = archive_orders.rowcount
            
            # Delete archived orders
            if orders_archived > 0:
                conn.execute(text("""
                    DELETE FROM orders 
                    WHERE order_date < :cutoff_date
                    AND status IN ('closed', 'cancelled')
                    AND id IN (SELECT id FROM archived_orders)
                """), {"cutoff_date": cutoff_date})
            
            # Archive old shipments
            archive_shipments = conn.execute(text("""
                INSERT INTO archived_shipments 
                SELECT * FROM shipments 
                WHERE actual_delivery_date < :cutoff_date
                AND status = 'delivered'
                ON CONFLICT (id) DO NOTHING
            """), {"cutoff_date": cutoff_date})
            
            shipments_archived = archive_shipments.rowcount
            
            # Delete archived shipments
            if shipments_archived > 0:
                conn.execute(text("""
                    DELETE FROM shipments 
                    WHERE actual_delivery_date < :cutoff_date
                    AND status = 'delivered'
                    AND id IN (SELECT id FROM archived_shipments)
                """), {"cutoff_date": cutoff_date})
            
            logger.info(f"Archived {orders_archived} orders and {shipments_archived} shipments")
            
            return {
                "orders_archived": orders_archived,
                "shipments_archived": shipments_archived
            }
        except Exception as e:
            logger.error(f"Archive operation failed: {str(e)}")
            return {
                "orders_archived": 0,
                "shipments_archived": 0,
                "error": str(e)
            }


@scheduled_job(name="vacuum_database", description="Vacuum and analyze database tables")
async def vacuum_database():
    """Run VACUUM and ANALYZE on database tables for performance"""
    tables_processed = []
    
    try:
        with engine.connect() as conn:
            # Get all tables
            result = conn.execute(text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                AND tablename NOT LIKE 'pg_%'
                AND tablename NOT LIKE 'sql_%'
            """))
            
            tables = [row[0] for row in result]
            
            # VACUUM and ANALYZE each table
            for table in tables:
                try:
                    # PostgreSQL requires VACUUM to be run outside transaction
                    conn.execute(text("COMMIT"))
                    conn.execute(text(f"VACUUM ANALYZE {table}"))
                    tables_processed.append(table)
                    logger.info(f"Vacuumed table: {table}")
                except Exception as e:
                    logger.error(f"Failed to vacuum table {table}: {str(e)}")
        
        logger.info(f"Vacuum completed for {len(tables_processed)} tables")
        return {
            "tables_processed": len(tables_processed),
            "tables": tables_processed
        }
    except Exception as e:
        logger.error(f"Vacuum operation failed: {str(e)}")
        return {
            "tables_processed": 0,
            "error": str(e)
        }


@scheduled_job(name="cleanup_orphaned_data", description="Remove orphaned records")
async def cleanup_orphaned_data():
    """Clean up orphaned records in database"""
    with get_db_session() as db:
        try:
            # Remove order items without orders
            orphaned_items = db.execute(text("""
                DELETE FROM order_items 
                WHERE order_id NOT IN (SELECT id FROM orders)
            """))
            
            items_deleted = orphaned_items.rowcount
            
            # Remove shipment items without shipments
            orphaned_shipments = db.execute(text("""
                DELETE FROM shipment_items 
                WHERE shipment_id NOT IN (SELECT id FROM shipments)
            """))
            
            shipments_deleted = orphaned_shipments.rowcount
            
            # Remove inventory without products
            orphaned_inventory = db.execute(text("""
                DELETE FROM inventory 
                WHERE product_id IS NOT NULL 
                AND product_id NOT IN (SELECT id FROM products)
            """))
            
            inventory_deleted = orphaned_inventory.rowcount
            
            db.commit()
            
            total_deleted = items_deleted + shipments_deleted + inventory_deleted
            
            logger.info(f"Cleaned up {total_deleted} orphaned records")
            
            return {
                "total_deleted": total_deleted,
                "order_items": items_deleted,
                "shipment_items": shipments_deleted,
                "inventory": inventory_deleted
            }
        except Exception as e:
            logger.error(f"Cleanup orphaned data failed: {str(e)}")
            return {
                "total_deleted": 0,
                "error": str(e)
            }


@scheduled_job(name="optimize_indexes", description="Rebuild database indexes")
async def optimize_indexes():
    """Rebuild database indexes for performance"""
    indexes_rebuilt = []
    
    try:
        with engine.connect() as conn:
            # Get all indexes
            result = conn.execute(text("""
                SELECT indexname, tablename 
                FROM pg_indexes 
                WHERE schemaname = 'public'
                AND indexname NOT LIKE 'pg_%'
            """))
            
            indexes = [(row[0], row[1]) for row in result]
            
            # Rebuild each index
            for index_name, table_name in indexes:
                try:
                    conn.execute(text(f"REINDEX INDEX {index_name}"))
                    indexes_rebuilt.append(index_name)
                    logger.info(f"Rebuilt index: {index_name} on {table_name}")
                except Exception as e:
                    logger.error(f"Failed to rebuild index {index_name}: {str(e)}")
        
        logger.info(f"Rebuilt {len(indexes_rebuilt)} indexes")
        return {
            "indexes_rebuilt": len(indexes_rebuilt),
            "indexes": indexes_rebuilt
        }
    except Exception as e:
        logger.error(f"Index optimization failed: {str(e)}")
        return {
            "indexes_rebuilt": 0,
            "error": str(e)
        }


async def run_cleanup(cleanup_type: str, retention_days: int = 90):
    """Run specific cleanup job"""
    if cleanup_type == "analytics":
        return await cleanup_old_analytics(retention_days)
    elif cleanup_type == "sessions":
        return await cleanup_expired_sessions()
    elif cleanup_type == "archive":
        return await archive_old_data(retention_days)
    elif cleanup_type == "orphaned":
        return await cleanup_orphaned_data()
    else:
        raise ValueError(f"Unknown cleanup type: {cleanup_type}")


# Utility functions for manual cleanup
def get_database_size() -> dict:
    """Get current database size information"""
    try:
        with engine.connect() as conn:
            # Total database size
            result = conn.execute(text("""
                SELECT pg_database_size(current_database()) as total_size
            """))
            total_size = result.scalar()
            
            # Table sizes
            result = conn.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                LIMIT 20
            """))
            
            table_sizes = [
                {
                    "table": row[1],
                    "size": row[2],
                    "size_bytes": row[3]
                }
                for row in result
            ]
            
            return {
                "total_size": total_size,
                "total_size_pretty": f"{total_size / (1024**3):.2f} GB",
                "largest_tables": table_sizes
            }
    except Exception as e:
        logger.error(f"Failed to get database size: {str(e)}")
        return {
            "error": str(e)
        }


def estimate_cleanup_impact(retention_days: int = 30) -> dict:
    """Estimate the impact of running cleanup"""
    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
    
    try:
        with get_db_session() as db:
            # Count records to be deleted
            analytics = db.query(AnalyticsResult).filter(
                AnalyticsResult.created_at < cutoff_date
            ).count()
            
            sessions = db.query(UserSession).filter(
                UserSession.expires_at < datetime.utcnow()
            ).count()
            
            return {
                "retention_days": retention_days,
                "estimated_deletions": {
                    "analytics_results": analytics,
                    "expired_sessions": sessions,
                    "total": analytics + sessions
                }
            }
    except Exception as e:
        logger.error(f"Failed to estimate cleanup impact: {str(e)}")
        return {
            "error": str(e)
        }

