"""
Audit logging system.

This module provides functions for logging auditable actions in the system
for compliance and security purposes.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import uuid
import asyncio
from pydantic import BaseModel

from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Audit log model
class AuditLog(BaseModel):
    """Model for audit log entries"""
    id: str
    timestamp: datetime
    user_id: str
    username: Optional[str] = None
    action: str
    resource: str
    resource_id: Optional[str] = None
    client_id: Optional[str] = None
    ip_address: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# In-memory audit log store (for development environments)
# In production, these would be written to a database
_audit_logs: List[AuditLog] = []

# Async queue for background processing
_audit_queue = asyncio.Queue()
_background_task = None

async def log_audit(
    user_id: str,
    action: str,
    resource: str,
    resource_id: Optional[str] = None,
    client_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    username: Optional[str] = None
) -> str:
    """
    Log an auditable action.
    
    Args:
        user_id: ID of the user performing the action
        action: Action being performed (e.g., "create", "update", "delete")
        resource: Resource type (e.g., "user", "dashboard", "query")
        resource_id: Optional ID of the resource being acted upon
        client_id: Optional client ID
        ip_address: Optional IP address of the user
        details: Optional additional details about the action
        username: Optional username (will be looked up if not provided)
        
    Returns:
        ID of the created audit log entry
    """
    try:
        # Generate a unique ID for this audit log
        log_id = f"audit-{uuid.uuid4().hex}"
        
        # Create the audit log entry
        audit_log = AuditLog(
            id=log_id,
            timestamp=datetime.now(),
            user_id=user_id,
            username=username,  # Will be populated by background processor if None
            action=action,
            resource=resource,
            resource_id=resource_id,
            client_id=client_id,
            ip_address=ip_address,
            details=details
        )
        
        # Add to queue for background processing
        await _audit_queue.put(audit_log)
        
        # Start background processing if not already running
        ensure_background_task()
        
        return log_id
    
    except Exception as e:
        # Log error but don't fail the main operation
        logger.error(f"Error creating audit log: {str(e)}")
        return ""

async def get_audit_logs(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    resource: Optional[str] = None,
    client_id: Optional[str] = None,
    limit: int = 100
) -> List[AuditLog]:
    """
    Get audit logs with optional filtering.
    
    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter
        user_id: Optional user ID filter
        action: Optional action filter
        resource: Optional resource filter
        client_id: Optional client ID filter
        limit: Maximum number of logs to return
        
    Returns:
        List of matching audit logs
    """
    # In a real implementation, this would query a database
    # For now, we filter the in-memory logs
    
    filtered_logs = _audit_logs.copy()
    
    # Apply filters
    if start_date:
        filtered_logs = [log for log in filtered_logs if log.timestamp >= start_date]
    
    if end_date:
        filtered_logs = [log for log in filtered_logs if log.timestamp <= end_date]
    
    if user_id:
        filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
    
    if action:
        filtered_logs = [log for log in filtered_logs if log.action == action]
    
    if resource:
        filtered_logs = [log for log in filtered_logs if log.resource == resource]
    
    if client_id:
        filtered_logs = [log for log in filtered_logs if log.client_id == client_id]
    
    # Sort by timestamp (newest first)
    filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
    
    # Apply limit
    return filtered_logs[:limit]

def ensure_background_task() -> None:
    """
    Ensure the background processing task is running.
    """
    global _background_task
    
    if _background_task is None or _background_task.done():
        _background_task = asyncio.create_task(_process_audit_logs())
        logger.debug("Started audit log background processor")

async def _process_audit_logs() -> None:
    """
    Background task to process audit logs.
    """
    global _audit_logs
    
    try:
        while True:
            # Get the next audit log from the queue
            audit_log = await _audit_queue.get()
            
            try:
                # Look up username if not provided
                if audit_log.username is None:
                    audit_log.username = await _lookup_username(audit_log.user_id)
                
                # In a real implementation, this would save to a database
                # For development, we just keep in memory
                if settings.environment == "development":
                    _audit_logs.append(audit_log)
                    
                    # Limit the size of the in-memory store
                    if len(_audit_logs) > 10000:
                        _audit_logs = _audit_logs[-10000:]
                else:
                    # In production, save to database
                    await _save_to_database(audit_log)
                
                # Log the audit action
                audit_details = audit_log.details or {}
                logger.info(
                    f"AUDIT: {audit_log.action} {audit_log.resource} "
                    f"by {audit_log.username or audit_log.user_id} "
                    f"({', '.join(f'{k}={v}' for k, v in audit_details.items() if k != 'password')})"
                )
            
            except Exception as e:
                logger.error(f"Error processing audit log: {str(e)}")
            
            finally:
                # Mark task as done
                _audit_queue.task_done()
    
    except asyncio.CancelledError:
        logger.info("Audit log processor cancelled")
    
    except Exception as e:
        logger.error(f"Error in audit log processor: {str(e)}")
        # Restart the task after a delay
        await asyncio.sleep(5)
        ensure_background_task()

async def _lookup_username(user_id: str) -> Optional[str]:
    """
    Look up a username from a user ID.
    
    Args:
        user_id: User ID to look up
        
    Returns:
        Username or None if not found
    """
    # In a real implementation, this would query a database
    # For now, return None to indicate "unknown user"
    
    try:
        # Mock implementation - in reality would query user database
        # Import here to avoid circular imports
        from app.db.interfaces.user_interface import UserInterface
        
        user_interface = UserInterface()
        user = await user_interface.get_user(user_id)
        
        if user:
            return user.username
    
    except Exception as e:
        logger.error(f"Error looking up username: {str(e)}")
    
    return None

async def _save_to_database(audit_log: AuditLog) -> None:
    """
    Save an audit log to the database.
    
    Args:
        audit_log: Audit log to save
    """
    # This would be implemented with your database connector
    # For now, it's a placeholder
    
    try:
        # Mock implementation - in reality would save to database
        # Example with PostgreSQL:
        # from app.db.connectors.postgres import PostgresConnector
        # db = PostgresConnector()
        # await db.execute(
        #     "INSERT INTO audit_logs (id, timestamp, user_id, username, action, resource, "
        #     "resource_id, client_id, ip_address, details) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
        #     audit_log.id,
        #     audit_log.timestamp,
        #     audit_log.user_id,
        #     audit_log.username,
        #     audit_log.action,
        #     audit_log.resource,
        #     audit_log.resource_id,
        #     audit_log.client_id,
        #     audit_log.ip_address,
        #     json.dumps(audit_log.details) if audit_log.details else None
        # )
        
        pass  # Placeholder
    
    except Exception as e:
        logger.error(f"Error saving audit log to database: {str(e)}")
        # In case of database failure, fall back to in-memory storage
        _audit_logs.append(audit_log)

async def export_audit_logs(
    format: str = "json",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    client_id: Optional[str] = None
) -> str:
    """
    Export audit logs for compliance reporting.
    
    Args:
        format: Export format ("json" or "csv")
        start_date: Optional start date filter
        end_date: Optional end date filter
        client_id: Optional client ID filter
        
    Returns:
        Exported audit logs as a string
    """
    # Get filtered logs
    logs = await get_audit_logs(
        start_date=start_date,
        end_date=end_date,
        client_id=client_id,
        limit=10000  # Higher limit for exports
    )
    
    if format == "json":
        # Convert to JSON
        return json.dumps([log.dict() for log in logs], default=str, indent=2)
    
    elif format == "csv":
        # Convert to CSV
        import csv
        import io
        
        output = io.StringIO()
        fieldnames = [
            "id", "timestamp", "user_id", "username", "action", 
            "resource", "resource_id", "client_id", "ip_address", "details"
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for log in logs:
            row = log.dict()
            # Convert details to string
            if row["details"]:
                row["details"] = json.dumps(row["details"])
            writer.writerow(row)
        
        return output.getvalue()
    
    else:
        raise ValueError(f"Unsupported export format: {format}")

# Cleanup function for application shutdown
async def cleanup_audit_logger() -> None:
    """
    Clean up audit logger resources.
    """
    global _background_task
    
    if _background_task:
        _background_task.cancel()
        try:
            await _background_task
        except asyncio.CancelledError:
            pass
        
        _background_task = None
        
    logger.info("Audit logger shutdown complete")