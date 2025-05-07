from typing import Dict, List, Any, Optional, Set, Union
import asyncio
from datetime import datetime, timedelta
import json
import hashlib
import uuid

from app.db.schema.schema_discovery import get_connector_for_client, discover_client_schema
from app.db.mirroring.data_syncer import sync_data, get_sync_status
from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Store schema hashes for change detection
_schema_hashes: Dict[str, Dict[str, str]] = {}

# Store active monitors
_active_monitors: Dict[str, Any] = {}

class ChangeMonitor:
    """
    Monitor for database schema and data changes.
    """
    def __init__(
        self,
        client_id: str,
        connection_id: Optional[str] = None,
        target_client_id: Optional[str] = None,
        target_connection_id: Optional[str] = None,
        interval_seconds: int = 300,  # 5 minutes
        monitor_id: Optional[str] = None
    ):
        self.client_id = client_id
        self.connection_id = connection_id
        self.target_client_id = target_client_id or client_id
        self.target_connection_id = target_connection_id
        self.interval_seconds = interval_seconds
        self.monitor_id = monitor_id or f"monitor-{uuid.uuid4().hex}"
        self.is_running = False
        self.last_check_time = None
        self.last_sync_time = None
        self.last_sync_id = None
        self.changes_detected = False
        self.task = None
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "monitor_id": self.monitor_id,
            "client_id": self.client_id,
            "connection_id": self.connection_id,
            "target_client_id": self.target_client_id,
            "target_connection_id": self.target_connection_id,
            "interval_seconds": self.interval_seconds,
            "is_running": self.is_running,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "last_sync_id": self.last_sync_id,
            "changes_detected": self.changes_detected,
            "error": self.error
        }
    
    async def start(self) -> None:
        """Start the monitor"""
        if self.is_running:
            return
        
        self.is_running = True
        self.error = None
        
        # Store initial schema hash
        try:
            await self._update_schema_hash()
        except Exception as e:
            self.error = f"Error initializing schema hash: {str(e)}"
            self.is_running = False
            logger.error(self.error)
            return
        
        # Start monitoring task
        self.task = asyncio.create_task(self._monitor_loop())
        
        # Register in active monitors
        _active_monitors[self.monitor_id] = self
        
        logger.info(f"Started change monitor {self.monitor_id} for client {self.client_id}")
    
    async def stop(self) -> None:
        """Stop the monitor"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel the task if it's running
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        # Remove from active monitors
        if self.monitor_id in _active_monitors:
            del _active_monitors[self.monitor_id]
        
        logger.info(f"Stopped change monitor {self.monitor_id}")
    
    async def check_now(self) -> bool:
        """
        Perform an immediate check for changes.
        
        Returns:
            True if changes detected, False otherwise
        """
        changes = await self._check_for_changes()
        return changes
    
    async def sync_now(self) -> Optional[str]:
        """
        Force a sync operation.
        
        Returns:
            Sync ID if started, None if error
        """
        try:
            # Start a sync
            sync_id = await sync_data(
                client_id=self.client_id,
                connection_id=self.connection_id,
                target_client_id=self.target_client_id,
                target_connection_id=self.target_connection_id,
                is_full_sync=False
            )
            
            self.last_sync_id = sync_id
            self.last_sync_time = datetime.now()
            
            return sync_id
        except Exception as e:
            self.error = f"Error starting sync: {str(e)}"
            logger.error(self.error)
            return None
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Check for changes
                changes = await self._check_for_changes()
                
                if changes:
                    # Start a sync
                    await self.sync_now()
                
                # Update last check time
                self.last_check_time = datetime.now()
                
                # Wait for next check
                await asyncio.sleep(self.interval_seconds)
                
            except asyncio.CancelledError:
                # Monitor was stopped
                break
            except Exception as e:
                self.error = f"Error in monitor loop: {str(e)}"
                logger.error(self.error)
                
                # Wait a bit before retrying
                await asyncio.sleep(60)
    
    async def _check_for_changes(self) -> bool:
        """
        Check for schema or data changes.
        
        Returns:
            True if changes detected, False otherwise
        """
        try:
            self.changes_detected = False
            
            # Check for schema changes
            schema_changes = await self._check_schema_changes()
            
            # Check for data changes
            data_changes = await self._check_data_changes()
            
            # Return True if either changed
            self.changes_detected = schema_changes or data_changes
            return self.changes_detected
            
        except Exception as e:
            error_msg = f"Error checking for changes: {str(e)}"
            logger.error(error_msg)
            self.error = error_msg
            return False
    
    async def _check_schema_changes(self) -> bool:
        """
        Check for schema changes.
        
        Returns:
            True if schema changed, False otherwise
        """
        # Get previous hash
        prev_hash = _schema_hashes.get(self.client_id, {}).get(self.connection_id or "default")
        if not prev_hash:
            # No previous hash, store current and return False
            await self._update_schema_hash()
            return False
        
        # Get current schema hash
        current_hash = await self._calculate_schema_hash()
        
        # Compare hashes
        if current_hash != prev_hash:
            logger.info(f"Schema change detected for client {self.client_id}")
            
            # Update stored hash
            if self.client_id not in _schema_hashes:
                _schema_hashes[self.client_id] = {}
            _schema_hashes[self.client_id][self.connection_id or "default"] = current_hash
            
            return True
        
        return False
    
    async def _check_data_changes(self) -> bool:
        """
        Check for data changes.
        
        Returns:
            True if data changed since last sync, False otherwise
        """
        # This would be database-specific and depend on your change tracking strategy
        # For demonstration, we'll use a simple approach that checks for updates
        # in timestamp columns in key tables
        
        # Get a connector
        connector = await get_connector_for_client(self.client_id, self.connection_id)
        
        try:
            # Get schema to identify tables with update tracking
            schema = await discover_client_schema(
                client_id=self.client_id,
                connection_id=self.connection_id
            )
            
            # Get last sync time
            last_sync_time = self.last_sync_time
            if not last_sync_time:
                # No previous sync, consider as changed
                return True
            
            # Check key tables for changes
            for table in schema.tables:
                # Look for updated_at or similar columns
                update_column = None
                for column in table.get("columns", []):
                    col_name = column["name"].lower()
                    if "updated_at" in col_name or "modified" in col_name or "last_update" in col_name:
                        update_column = column["name"]
                        break
                
                # If we found an update column, check for changes
                if update_column:
                    query = f"""
                    SELECT COUNT(*) as count
                    FROM {table['name']}
                    WHERE {update_column} > :threshold
                    """
                    
                    result = await connector.execute_query(
                        query=query,
                        params={"threshold": last_sync_time.isoformat()}
                    )
                    
                    if result["data"] and result["data"][0].get("count", 0) > 0:
                        logger.info(f"Data changes detected in table {table['name']}")
                        return True
            
            # No changes detected
            return False
            
        finally:
            # Close the connector
            await connector.close()
    
    async def _calculate_schema_hash(self) -> str:
        """
        Calculate a hash of the database schema for change detection.
        
        Returns:
            Schema hash string
        """
        # Discover schema
        schema = await discover_client_schema(
            client_id=self.client_id,
            connection_id=self.connection_id,
            force_refresh=True
        )
        
        # Convert to JSON and hash
        schema_json = json.dumps(schema.to_dict(), sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()
    
    async def _update_schema_hash(self) -> None:
        """Update the stored schema hash"""
        hash_value = await self._calculate_schema_hash()
        
        if self.client_id not in _schema_hashes:
            _schema_hashes[self.client_id] = {}
        
        _schema_hashes[self.client_id][self.connection_id or "default"] = hash_value

async def start_change_monitor(
    client_id: str,
    connection_id: Optional[str] = None,
    target_client_id: Optional[str] = None,
    target_connection_id: Optional[str] = None,
    interval_seconds: int = 300
) -> str:
    """
    Start monitoring for database changes.
    
    Args:
        client_id: Client ID
        connection_id: Optional connection ID
        target_client_id: Target client ID
        target_connection_id: Target connection ID
        interval_seconds: Check interval in seconds
        
    Returns:
        Monitor ID
    """
    # Create a new monitor
    monitor = ChangeMonitor(
        client_id=client_id,
        connection_id=connection_id,
        target_client_id=target_client_id,
        target_connection_id=target_connection_id,
        interval_seconds=interval_seconds
    )
    
    # Start monitoring
    await monitor.start()
    
    return monitor.monitor_id

async def stop_change_monitor(monitor_id: str) -> bool:
    """
    Stop a change monitor.
    
    Args:
        monitor_id: Monitor ID
        
    Returns:
        True if stopped, False if not found
    """
    if monitor_id in _active_monitors:
        monitor = _active_monitors[monitor_id]
        await monitor.stop()
        return True
    
    return False

async def get_monitors(
    client_id: Optional[str] = None,
    connection_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all active monitors.
    
    Args:
        client_id: Optional client ID filter
        connection_id: Optional connection ID filter
        
    Returns:
        List of monitor information
    """
    monitors = []
    
    for monitor in _active_monitors.values():
        # Apply filters
        if client_id and monitor.client_id != client_id:
            continue
            
        if connection_id and monitor.connection_id != connection_id:
            continue
        
        monitors.append(monitor.to_dict())
    
    return monitors

async def start_all_monitors() -> None:
    """Start all configured monitors at application startup"""
    # In a production system, you'd load monitor configurations from database
    # For demonstration, we'll just log
    logger.info("Starting all configured change monitors")

async def stop_all_monitors() -> None:
    """Stop all active monitors at application shutdown"""
    # Get a copy of the monitor IDs (to avoid dict changing during iteration)
    monitor_ids = list(_active_monitors.keys())
    
    for monitor_id in monitor_ids:
        await stop_change_monitor(monitor_id)
    
    logger.info("Stopped all change monitors")

async def trigger_sync(
    client_id: str,
    connection_id: Optional[str] = None,
    force_full_sync: bool = False
) -> str:
    """
    Trigger a manual data sync.
    
    Args:
        client_id: Client ID
        connection_id: Optional connection ID
        force_full_sync: Whether to force a full sync
        
    Returns:
        Sync ID
    """
    # Start a sync
    sync_id = await sync_data(
        client_id=client_id,
        connection_id=connection_id,
        is_full_sync=force_full_sync
    )
    
    return sync_id