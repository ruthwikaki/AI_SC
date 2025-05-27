# jobs/scheduler.py

"""
Job scheduler setup using APScheduler
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from functools import wraps
import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from sqlalchemy.orm import Session

from app.db.database import get_db_session, engine
from app.core.config import settings
from app.models import ScheduledAnalytics, SystemSettings
from app.db.repositories.analytics_repository import AnalyticsRepository

logger = logging.getLogger(__name__)


class JobScheduler:
    """Main job scheduler for background tasks"""
    
    def __init__(self, use_async: bool = True):
        """
        Initialize job scheduler
        
        Args:
            use_async: Use async scheduler (True) or background scheduler (False)
        """
        self.use_async = use_async
        
        # Configure job stores
        jobstores = {
            'default': SQLAlchemyJobStore(
                engine=engine,
                tablename='apscheduler_jobs'
            )
        }
        
        # Configure executors
        executors = {
            'default': ThreadPoolExecutor(20),
            'processpool': ProcessPoolExecutor(5)
        }
        
        # Job defaults
        job_defaults = {
            'coalesce': True,
            'max_instances': 3,
            'misfire_grace_time': 300  # 5 minutes
        }
        
        # Create scheduler
        if use_async:
            self.scheduler = AsyncIOScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=settings.TIMEZONE
            )
        else:
            self.scheduler = BackgroundScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=settings.TIMEZONE
            )
        
        # Add event listeners
        self.scheduler.add_listener(
            self._job_executed_listener,
            EVENT_JOB_EXECUTED
        )
        self.scheduler.add_listener(
            self._job_error_listener,
            EVENT_JOB_ERROR
        )
        
        self._initialized = False
    
    def start(self):
        """Start the scheduler"""
        if not self._initialized:
            self._initialize_jobs()
            self._initialized = True
        
        self.scheduler.start()
        logger.info("Job scheduler started")
    
    def shutdown(self, wait: bool = True):
        """Shutdown the scheduler"""
        self.scheduler.shutdown(wait=wait)
        logger.info("Job scheduler stopped")
    
    def _initialize_jobs(self):
        """Initialize all scheduled jobs"""
        # System maintenance jobs
        self.scheduler.add_job(
            func=cleanup_expired_sessions,
            trigger=IntervalTrigger(hours=1),
            id='cleanup_sessions',
            name='Cleanup expired sessions',
            replace_existing=True
        )
        
        self.scheduler.add_job(
            func=cleanup_old_logs,
            trigger=CronTrigger(hour=2, minute=0),  # Daily at 2 AM
            id='cleanup_logs',
            name='Cleanup old logs',
            replace_existing=True
        )
        
        # Analytics update jobs
        self.scheduler.add_job(
            func=update_all_analytics,
            trigger=CronTrigger(hour=1, minute=0),  # Daily at 1 AM
            id='update_analytics',
            name='Update all analytics',
            replace_existing=True
        )
        
        self.scheduler.add_job(
            func=update_supplier_metrics,
            trigger=IntervalTrigger(hours=6),
            id='update_supplier_metrics',
            name='Update supplier metrics',
            replace_existing=True
        )
        
        self.scheduler.add_job(
            func=update_inventory_metrics,
            trigger=IntervalTrigger(hours=4),
            id='update_inventory_metrics',
            name='Update inventory metrics',
            replace_existing=True
        )
        
        # Data synchronization jobs
        self.scheduler.add_job(
            func=sync_inventory_data,
            trigger=IntervalTrigger(minutes=30),
            id='sync_inventory',
            name='Sync inventory data',
            replace_existing=True
        )
        
        self.scheduler.add_job(
            func=sync_order_data,
            trigger=IntervalTrigger(hours=1),
            id='sync_orders',
            name='Sync order data',
            replace_existing=True
        )
        
        # Database maintenance
        self.scheduler.add_job(
            func=vacuum_database,
            trigger=CronTrigger(day_of_week='sun', hour=3, minute=0),  # Weekly on Sunday
            id='vacuum_db',
            name='Vacuum database',
            replace_existing=True
        )
        
        # Load user-defined scheduled jobs from database
        self._load_user_scheduled_jobs()
    
    def _load_user_scheduled_jobs(self):
        """Load user-defined scheduled jobs from database"""
        with get_db_session() as db:
            scheduled_jobs = db.query(ScheduledAnalytics).filter(
                ScheduledAnalytics.is_active == True
            ).all()
            
            for job in scheduled_jobs:
                self.add_analytics_job(job)
    
    def add_analytics_job(self, scheduled_analytics: ScheduledAnalytics):
        """Add a scheduled analytics job"""
        job_id = f"analytics_{scheduled_analytics.id}"
        
        # Parse schedule configuration
        schedule_config = scheduled_analytics.schedule_config
        
        if schedule_config.get("type") == "cron":
            trigger = CronTrigger(**schedule_config.get("cron", {}))
        elif schedule_config.get("type") == "interval":
            trigger = IntervalTrigger(**schedule_config.get("interval", {}))
        else:
            logger.error(f"Invalid schedule type for job {job_id}")
            return
        
        # Add job
        self.scheduler.add_job(
            func=run_scheduled_analytics,
            trigger=trigger,
            args=[scheduled_analytics.id],
            id=job_id,
            name=scheduled_analytics.name,
            replace_existing=True
        )
        
        logger.info(f"Added scheduled analytics job: {scheduled_analytics.name}")
    
    def remove_job(self, job_id: str):
        """Remove a scheduled job"""
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
        except Exception as e:
            logger.error(f"Error removing job {job_id}: {str(e)}")
    
    def pause_job(self, job_id: str):
        """Pause a scheduled job"""
        self.scheduler.pause_job(job_id)
    
    def resume_job(self, job_id: str):
        """Resume a paused job"""
        self.scheduler.resume_job(job_id)
    
    def get_jobs(self) -> list:
        """Get all scheduled jobs"""
        return self.scheduler.get_jobs()
    
    def _job_executed_listener(self, event):
        """Handle job execution events"""
        logger.info(f"Job {event.job_id} executed successfully")
    
    def _job_error_listener(self, event):
        """Handle job error events"""
        logger.error(f"Job {event.job_id} failed: {event.exception}")
        
        # Log to database
        with get_db_session() as db:
            from app.models import ErrorLog
            error_log = ErrorLog(
                error_type="job_error",
                error_message=str(event.exception),
                error_details={
                    "job_id": event.job_id,
                    "scheduled_run_time": event.scheduled_run_time.isoformat()
                }
            )
            db.add(error_log)
            db.commit()


# Job execution functions
async def run_scheduled_analytics(analytics_id: str):
    """Run a scheduled analytics job"""
    from .analytics_update import run_analytics_by_id
    await run_analytics_by_id(analytics_id)


async def update_all_analytics():
    """Update all analytics metrics"""
    from .analytics_update import run_all_analytics
    await run_all_analytics()


async def cleanup_expired_sessions():
    """Cleanup expired user sessions"""
    from .cleanup import cleanup_expired_sessions as cleanup_func
    await cleanup_func()


async def cleanup_old_logs():
    """Cleanup old system logs"""
    from .cleanup import cleanup_old_logs as cleanup_func
    await cleanup_func()


async def sync_inventory_data():
    """Sync inventory data from external sources"""
    from .data_sync import sync_inventory_data as sync_func
    await sync_func()


async def sync_order_data():
    """Sync order data from external sources"""
    from .data_sync import sync_order_data as sync_func
    await sync_func()


async def vacuum_database():
    """Run database vacuum"""
    from .cleanup import vacuum_database as vacuum_func
    await vacuum_func()


# Decorator for job functions
def scheduled_job(name: str, description: str = ""):
    """Decorator for scheduled job functions"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            job_name = name or func.__name__
            
            logger.info(f"Starting job: {job_name}")
            
            try:
                result = await func(*args, **kwargs)
                
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"Job {job_name} completed in {duration:.2f} seconds")
                
                # Log successful execution
                with get_db_session() as db:
                    from app.models import SystemSettings
                    setting = db.query(SystemSettings).filter(
                        SystemSettings.key == f"last_job_run_{job_name}"
                    ).first()
                    
                    if setting:
                        setting.value = {"timestamp": datetime.utcnow().isoformat(), "duration": duration}
                    else:
                        setting = SystemSettings(
                            key=f"last_job_run_{job_name}",
                            value={"timestamp": datetime.utcnow().isoformat(), "duration": duration},
                            description=description
                        )
                        db.add(setting)
                    
                    db.commit()
                
                return result
                
            except Exception as e:
                logger.error(f"Job {job_name} failed: {str(e)}")
                
                # Log error
                with get_db_session() as db:
                    from app.models import ErrorLog
                    error_log = ErrorLog(
                        error_type="job_error",
                        error_message=str(e),
                        error_details={
                            "job_name": job_name,
                            "start_time": start_time.isoformat()
                        }
                    )
                    db.add(error_log)
                    db.commit()
                
                raise
        
        wrapper._job_name = name
        wrapper._job_description = description
        return wrapper
    
    return decorator


# Global scheduler instance
scheduler = JobScheduler(use_async=True)


# Convenience functions
def start_scheduler():
    """Start the global scheduler"""
    scheduler.start()


def stop_scheduler():
    """Stop the global scheduler"""
    scheduler.shutdown()


def schedule_analytics_update(
    analytics_type: str,
    schedule: Dict[str, Any],
    name: str,
    description: str = ""
) -> str:
    """
    Schedule an analytics update job
    
    Args:
        analytics_type: Type of analytics to run
        schedule: Schedule configuration
        name: Job name
        description: Job description
        
    Returns:
        Job ID
    """
    job_id = f"analytics_{analytics_type}_{datetime.utcnow().timestamp()}"
    
    # Create scheduled analytics record
    with get_db_session() as db:
        scheduled = ScheduledAnalytics(
            name=name,
            analytics_type=analytics_type,
            schedule_config=schedule,
            is_active=True,
            created_by_id=None  # System created
        )
        db.add(scheduled)
        db.commit()
        
        # Add to scheduler
        scheduler.add_analytics_job(scheduled)
        
        return str(scheduled.id)


def schedule_report_generation(
    report_type: str,
    schedule: Dict[str, Any],
    recipients: list[str],
    name: str
) -> str:
    """Schedule a report generation job"""
    from .analytics_update import generate_scheduled_report
    
    job_id = f"report_{report_type}_{datetime.utcnow().timestamp()}"
    
    if schedule.get("type") == "cron":
        trigger = CronTrigger(**schedule.get("cron", {}))
    else:
        trigger = IntervalTrigger(**schedule.get("interval", {}))
    
    scheduler.scheduler.add_job(
        func=generate_scheduled_report,
        trigger=trigger,
        args=[report_type, recipients],
        id=job_id,
        name=name,
        replace_existing=True
    )
    
    return job_id


def schedule_data_sync(
    sync_type: str,
    schedule: Dict[str, Any],
    source_config: Dict[str, Any],
    name: str
) -> str:
    """Schedule a data synchronization job"""
    from .data_sync import run_data_sync
    
    job_id = f"sync_{sync_type}_{datetime.utcnow().timestamp()}"
    
    if schedule.get("type") == "cron":
        trigger = CronTrigger(**schedule.get("cron", {}))
    else:
        trigger = IntervalTrigger(**schedule.get("interval", {}))
    
    scheduler.scheduler.add_job(
        func=run_data_sync,
        trigger=trigger,
        args=[sync_type, source_config],
        id=job_id,
        name=name,
        replace_existing=True
    )
    
    return job_id


def schedule_cleanup(
    cleanup_type: str,
    schedule: Dict[str, Any],
    retention_days: int,
    name: str
) -> str:
    """Schedule a cleanup job"""
    from .cleanup import run_cleanup
    
    job_id = f"cleanup_{cleanup_type}_{datetime.utcnow().timestamp()}"
    
    if schedule.get("type") == "cron":
        trigger = CronTrigger(**schedule.get("cron", {}))
    else:
        trigger = IntervalTrigger(**schedule.get("interval", {}))
    
    scheduler.scheduler.add_job(
        func=run_cleanup,
        trigger=trigger,
        args=[cleanup_type, retention_days],
        id=job_id,
        name=name,
        replace_existing=True
    )
    
    return job_id