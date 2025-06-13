"""
Application metrics collection and reporting
Located at: /backend/app/utils/metrics.py
"""
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.models.query import NaturalLanguageQuery
from app.models.user import User, AuditLog
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class MetricsCollector:
    """Collect and report application metrics"""
    
    @staticmethod
    async def get_system_metrics() -> Dict[str, Any]:
        """Get system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    @staticmethod
    async def get_application_metrics() -> Dict[str, Any]:
        """Get application-specific metrics"""
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            last_hour = now - timedelta(hours=1)
            last_day = now - timedelta(days=1)
            
            # User metrics
            total_users = db.query(User).count()
            active_users = db.query(User).filter(User.is_active == True).count()
            
            # Query metrics
            total_queries = db.query(NaturalLanguageQuery).count()
            successful_queries = db.query(NaturalLanguageQuery).filter(
                NaturalLanguageQuery.success == True
            ).count()
            
            # Recent activity
            queries_last_hour = db.query(NaturalLanguageQuery).filter(
                NaturalLanguageQuery.created_at >= last_hour
            ).count()
            
            queries_last_day = db.query(NaturalLanguageQuery).filter(
                NaturalLanguageQuery.created_at >= last_day
            ).count()
            
            # Average execution time
            avg_execution_time = db.query(
                func.avg(NaturalLanguageQuery.execution_time)
            ).filter(
                NaturalLanguageQuery.success == True,
                NaturalLanguageQuery.execution_time.isnot(None)
            ).scalar() or 0
            
            # Most active users
            top_users = db.query(
                User.username,
                func.count(NaturalLanguageQuery.id).label('query_count')
            ).join(
                NaturalLanguageQuery
            ).group_by(
                User.id, User.username
            ).order_by(
                func.count(NaturalLanguageQuery.id).desc()
            ).limit(5).all()
            
            # Most common errors
            recent_errors = db.query(
                NaturalLanguageQuery.error_message,
                func.count(NaturalLanguageQuery.id).label('count')
            ).filter(
                NaturalLanguageQuery.success == False,
                NaturalLanguageQuery.error_message.isnot(None),
                NaturalLanguageQuery.created_at >= last_day
            ).group_by(
                NaturalLanguageQuery.error_message
            ).order_by(
                func.count(NaturalLanguageQuery.id).desc()
            ).limit(5).all()
            
            return {
                "users": {
                    "total": total_users,
                    "active": active_users,
                    "inactive": total_users - active_users
                },
                "queries": {
                    "total": total_queries,
                    "successful": successful_queries,
                    "failed": total_queries - successful_queries,
                    "success_rate": (successful_queries / total_queries * 100) if total_queries > 0 else 0,
                    "last_hour": queries_last_hour,
                    "last_day": queries_last_day,
                    "avg_execution_time": round(avg_execution_time, 3)
                },
                "top_users": [
                    {"username": user[0], "query_count": user[1]}
                    for user in top_users
                ],
                "recent_errors": [
                    {"error": error[0][:100], "count": error[1]}
                    for error in recent_errors
                ],
                "timestamp": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            return {}
        finally:
            db.close()
    
    @staticmethod
    async def get_database_metrics() -> Dict[str, Any]:
        """Get database connection and performance metrics"""
        from app.db.database import engine
        
        try:
            pool = engine.pool
            
            return {
                "connections": {
                    "size": pool.size(),
                    "checked_out": pool.checked_out(),
                    "overflow": pool.overflow(),
                    "total": pool.size() + pool.overflow()
                },
                "status": "connected"
            }
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")
            return {"status": "error", "error": str(e)}
    
    @staticmethod
    async def get_cache_metrics() -> Dict[str, Any]:
        """Get cache metrics if Redis is configured"""
        try:
            from app.cache.query_cache import QueryCache
            cache = QueryCache()
            
            info = await cache.get_info()
            
            return {
                "status": "connected",
                "memory_used": info.get("used_memory_human", "N/A"),
                "keys": info.get("db0", {}).get("keys", 0),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)) * 100
                ) if info.get("keyspace_hits", 0) > 0 else 0
            }
        except Exception as e:
            logger.debug(f"Cache metrics not available: {e}")
            return {"status": "not_configured"}

async def get_application_metrics() -> Dict[str, Any]:
    """Get comprehensive application metrics"""
    collector = MetricsCollector()
    
    # Collect all metrics concurrently
    results = await asyncio.gather(
        collector.get_system_metrics(),
        collector.get_application_metrics(),
        collector.get_database_metrics(),
        collector.get_cache_metrics(),
        return_exceptions=True
    )
    
    metrics = {
        "system": results[0] if not isinstance(results[0], Exception) else {},
        "application": results[1] if not isinstance(results[1], Exception) else {},
        "database": results[2] if not isinstance(results[2], Exception) else {},
        "cache": results[3] if not isinstance(results[3], Exception) else {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return metrics

async def log_metrics():
    """Log metrics periodically (for monitoring)"""
    metrics = await get_application_metrics()
    logger.info(f"Application metrics: {metrics}")
    return metrics