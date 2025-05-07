from typing import Dict, List, Any, Optional, Set, Tuple, Union
import asyncio
from datetime import datetime
import json
import time
import uuid

from app.db.schema.schema_discovery import get_connector_for_client, get_table_schema
from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Track currently running sync operations
_active_syncs = {}

class SyncStatus:
    """Object representing the status of a sync operation"""
    def __init__(
        self,
        sync_id: str,
        client_id: str,
        connection_id: Optional[str],
        target_client_id: str,
        target_connection_id: Optional[str],
        tables: List[str],
        start_time: datetime,
        is_full_sync: bool = False,
        is_initial_sync: bool = False
    ):
        self.sync_id = sync_id
        self.client_id = client_id
        self.connection_id = connection_id
        self.target_client_id = target_client_id
        self.target_connection_id = target_connection_id
        self.tables = tables
        self.start_time = start_time
        self.end_time = None
        self.current_table = None
        self.current_table_start_time = None
        self.tables_completed = 0
        self.rows_processed = 0
        self.rows_total = 0
        self.errors = []
        self.is_running = True
        self.is_full_sync = is_full_sync
        self.is_initial_sync = is_initial_sync
        self.success = None  # None = in progress, True/False = completed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "sync_id": self.sync_id,
            "client_id": self.client_id,
            "connection_id": self.connection_id,
            "target_client_id": self.target_client_id,
            "target_connection_id": self.target_connection_id,
            "tables": self.tables,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_table": self.current_table,
            "current_table_start_time": self.current_table_start_time.isoformat() if self.current_table_start_time else None,
            "tables_completed": self.tables_completed,
            "total_tables": len(self.tables),
            "rows_processed": self.rows_processed,
            "rows_total": self.rows_total,
            "percent_complete": self.get_percent_complete(),
            "errors": self.errors,
            "is_running": self.is_running,
            "is_full_sync": self.is_full_sync,
            "is_initial_sync": self.is_initial_sync,
            "estimated_completion": self.estimate_completion().isoformat() if self.estimate_completion() else None,
            "success": self.success,
            "duration_seconds": self.get_duration()
        }
    
    def get_percent_complete(self) -> float:
        """Calculate percent complete"""
        if len(self.tables) == 0:
            return 100.0
        
        table_weight = 100.0 / len(self.tables)
        complete_tables_percent = self.tables_completed * table_weight
        
        # If working on a table and we know the total rows, calculate progress
        if self.current_table and self.rows_total > 0:
            current_table_percent = (self.rows_processed / self.rows_total) * table_weight
        else:
            current_table_percent = 0
        
        return min(complete_tables_percent + current_table_percent, 100.0)
    
    def get_duration(self) -> float:
        """Calculate duration in seconds"""
        if not self.start_time:
            return 0
        
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def estimate_completion(self) -> Optional[datetime]:
        """Estimate completion time"""
        if not self.is_running or self.get_percent_complete() >= 100:
            return self.end_time or datetime.now()
        
        if self.tables_completed == 0 or self.get_percent_complete() == 0:
            # Not enough data to estimate
            return None
        
        # Calculate time per percent
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        seconds_per_percent = elapsed_time / self.get_percent_complete()
        
        # Estimate remaining time
        remaining_percent = 100 - self.get_percent_complete()
        remaining_seconds = remaining_percent * seconds_per_percent
        
        # Calculate estimated completion time
        return datetime.now() + timedelta(seconds=remaining_seconds)
    
    def start_table(self, table_name: str) -> None:
        """Mark starting work on a table"""
        self.current_table = table_name
        self.current_table_start_time = datetime.now()
        self.rows_processed = 0
        self.rows_total = 0
    
    def complete_table(self) -> None:
        """Mark a table as completed"""
        self.tables_completed += 1
        self.current_table = None
        self.current_table_start_time = None
    
    def add_error(self, error_message: str) -> None:
        """Add an error message"""
        self.errors.append({
            "message": error_message,
            "time": datetime.now().isoformat(),
            "table": self.current_table
        })
    
    def complete(self, success: bool) -> None:
        """Mark the sync operation as complete"""
        self.is_running = False
        self.end_time = datetime.now()
        self.success = success

async def sync_data(
    client_id: str,
    connection_id: Optional[str] = None,
    target_client_id: Optional[str] = None,
    target_connection_id: Optional[str] = None,
    tables: Optional[List[str]] = None,
    batch_size: int = 1000,
    is_full_sync: bool = False,
    is_initial_sync: bool = False
) -> str:
    """
    Synchronize data from source to target database.
    
    Args:
        client_id: Source client ID
        connection_id: Optional source connection ID
        target_client_id: Target client ID (if None, uses a new connection for same client)
        target_connection_id: Optional target connection ID
        tables: Optional list of specific tables to sync
        batch_size: Number of rows to process in each batch
        is_full_sync: Whether this is a full sync (vs. incremental)
        is_initial_sync: Whether this is the initial sync
        
    Returns:
        Sync ID that can be used to check status
    """
    # If target_client_id not specified, use the same as source
    if not target_client_id:
        target_client_id = client_id
    
    # If no tables specified, get all tables from the schema
    if not tables:
        from app.db.schema.schema_discovery import discover_client_schema
        schema = await discover_client_schema(
            client_id=client_id,
            connection_id=connection_id
        )
        tables = [table["name"] for table in schema.tables]
    
    # Generate a sync ID
    sync_id = f"sync-{uuid.uuid4().hex}"
    
    # Create a sync status object
    status = SyncStatus(
        sync_id=sync_id,
        client_id=client_id,
        connection_id=connection_id,
        target_client_id=target_client_id,
        target_connection_id=target_connection_id,
        tables=tables,
        start_time=datetime.now(),
        is_full_sync=is_full_sync,
        is_initial_sync=is_initial_sync
    )
    
    # Store the status
    _active_syncs[sync_id] = status
    
    # Start the sync in a background task
    asyncio.create_task(
        _sync_data_task(
            sync_id=sync_id,
            client_id=client_id,
            connection_id=connection_id,
            target_client_id=target_client_id,
            target_connection_id=target_connection_id,
            tables=tables,
            batch_size=batch_size,
            is_full_sync=is_full_sync,
            is_initial_sync=is_initial_sync
        )
    )
    
    return sync_id

async def _sync_data_task(
    sync_id: str,
    client_id: str,
    connection_id: Optional[str],
    target_client_id: str,
    target_connection_id: Optional[str],
    tables: List[str],
    batch_size: int,
    is_full_sync: bool,
    is_initial_sync: bool
) -> None:
    """
    Background task to synchronize data from source to target database.
    """
    status = _active_syncs.get(sync_id)
    if not status:
        logger.error(f"Sync {sync_id} not found")
        return
    
    overall_success = True
    
    try:
        # Process each table
        for table_name in tables:
            try:
                status.start_table(table_name)
                logger.info(f"Starting sync for table {table_name}")
                
                # Sync the table
                success = await sync_table_data(
                    source_client_id=client_id,
                    source_connection_id=connection_id,
                    target_client_id=target_client_id,
                    target_connection_id=target_connection_id,
                    table_name=table_name,
                    batch_size=batch_size,
                    is_full_sync=is_full_sync,
                    status=status
                )
                
                if not success:
                    overall_success = False
                    status.add_error(f"Failed to sync table {table_name}")
                
                status.complete_table()
                logger.info(f"Completed sync for table {table_name}")
                
            except Exception as e:
                overall_success = False
                error_msg = f"Error syncing table {table_name}: {str(e)}"
                logger.error(error_msg)
                status.add_error(error_msg)
    
    except Exception as e:
        overall_success = False
        error_msg = f"Error during sync: {str(e)}"
        logger.error(error_msg)
        status.add_error(error_msg)
    
    finally:
        # Mark the sync as complete
        status.complete(overall_success)
        logger.info(f"Sync {sync_id} completed with status: {overall_success}")
        
        # Store sync completion in database
        await store_sync_history(status)

from datetime import timedelta

async def sync_table_data(
    source_client_id: str,
    source_connection_id: Optional[str],
    target_client_id: str,
    target_connection_id: Optional[str],
    table_name: str,
    batch_size: int = 1000,
    is_full_sync: bool = False,
    status: Optional[SyncStatus] = None
) -> bool:
    """
    Synchronize data for a specific table.
    
    Args:
        source_client_id: Source client ID
        source_connection_id: Optional source connection ID
        target_client_id: Target client ID
        target_connection_id: Optional target connection ID
        table_name: Table to synchronize
        batch_size: Number of rows to process in each batch
        is_full_sync: Whether this is a full sync (vs. incremental)
        status: Optional sync status object for progress tracking
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get source and target connectors
        source_connector = await get_connector_for_client(source_client_id, source_connection_id)
        target_connector = await get_connector_for_client(target_client_id, target_connection_id)
        
        # Get table schema from source
        table_schema = await get_table_schema(source_client_id, table_name, connection_id=source_connection_id)
        if not table_schema:
            if status:
                status.add_error(f"Table schema not found for {table_name}")
            return False
        
        # Get primary key for table
        primary_key = table_schema.get("primary_key", [])
        if not primary_key:
            if status:
                status.add_error(f"No primary key found for table {table_name}")
            return False
        
        # Get tracking column for incremental sync
        tracking_column = await _get_tracking_column(table_schema)
        
        # Determine sync strategy
        if is_full_sync or not tracking_column:
            # Full sync - clear target table first
            await _clear_target_table(target_connector, table_name, target_client_id)
            where_clause = ""
            params = {}
        else:
            # Incremental sync - get last sync time
            last_sync_time = await _get_last_sync_time(source_client_id, target_client_id, table_name)
            if last_sync_time:
                where_clause = f"{tracking_column} >= :{tracking_column}_threshold"
                params = {f"{tracking_column}_threshold": last_sync_time}
            else:
                # No previous sync, do full sync
                await _clear_target_table(target_connector, table_name, target_client_id)
                where_clause = ""
                params = {}
        
        # Get total row count for progress tracking
        if status:
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            if where_clause:
                count_query += f" WHERE {where_clause}"
            
            count_result = await source_connector.execute_query(count_query, params)
            status.rows_total = count_result["data"][0].get("count", 0)
        
        # Prepare column list
        columns = [col["name"] for col in table_schema.get("columns", [])]
        columns_str = ", ".join([f'"{col}"' for col in columns])
        
        # Process data in batches using offset pagination
        offset = 0
        while True:
            # Construct query with pagination
            query = f"SELECT {columns_str} FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            query += f" ORDER BY {primary_key[0]} LIMIT {batch_size} OFFSET {offset}"
            
            # Execute query
            result = await source_connector.execute_query(query, params)
            
            # If no more rows, we're done
            if not result["data"]:
                break
            
            # Insert batch into target
            await _insert_batch(target_connector, table_name, columns, result["data"])
            
            # Update progress
            if status:
                status.rows_processed += len(result["data"])
            
            # Move to next batch
            offset += batch_size
            
            # Small delay to prevent overwhelming the database
            await asyncio.sleep(0.1)
        
        # Store last sync time
        await _store_last_sync_time(source_client_id, target_client_id, table_name)
        
        return True
        
    except Exception as e:
        logger.error(f"Error syncing table {table_name}: {str(e)}")
        if status:
            status.add_error(f"Error syncing table {table_name}: {str(e)}")
        return False
    finally:
        # Ensure connections are closed
        if 'source_connector' in locals():
            await source_connector.close()
        if 'target_connector' in locals():
            await target_connector.close()

async def _get_tracking_column(table_schema: Dict[str, Any]) -> Optional[str]:
    """
    Determine which column to use for tracking changes (for incremental sync).
    
    Looks for columns like updated_at, modified_date, etc.
    
    Args:
        table_schema: Table schema information
        
    Returns:
        Column name to use for tracking, or None if not found
    """
    # Common column names for tracking changes
    tracking_column_patterns = [
        "updated_at",
        "modified_at",
        "date_modified",
        "last_updated",
        "modified_date",
        "update_time",
        "last_modified"
    ]
    
    # Check if any of these exist in the table
    columns = table_schema.get("columns", [])
    for pattern in tracking_column_patterns:
        for col in columns:
            col_name = col["name"].lower()
            if pattern in col_name:
                return col["name"]
    
    # If no tracking column found, return None
    return None

async def _clear_target_table(
    target_connector: Any,
    table_name: str,
    target_client_id: str
) -> None:
    """
    Clear the target table before a full sync.
    
    Args:
        target_connector: Target database connector
        table_name: Table name
        target_client_id: Target client ID
    """
    try:
        # For safety, only truncate tables in the mirror database
        # In a production system, you'd verify this is a mirror table first
        query = f"TRUNCATE TABLE {table_name}"
        await target_connector.execute_query(query)
    except Exception as e:
        logger.error(f"Error clearing target table {table_name}: {str(e)}")
        # Continue with the sync even if truncate fails

async def _insert_batch(
    target_connector: Any,
    table_name: str,
    columns: List[str],
    data: List[Dict[str, Any]]
) -> None:
    """
    Insert a batch of data into the target table.
    
    Args:
        target_connector: Target database connector
        table_name: Table name
        columns: List of column names
        data: List of row data dictionaries
    """
    if not data:
        return
    
    # Prepare column lists
    column_str = ", ".join([f'"{col}"' for col in columns])
    
    # For each row, create a VALUES clause
    values_clauses = []
    params = {}
    
    for i, row in enumerate(data):
        param_names = []
        for col in columns:
            param_name = f"p{i}_{col}"
            param_names.append(f":{param_name}")
            params[param_name] = row.get(col)
        
        values_clause = f"({', '.join(param_names)})"
        values_clauses.append(values_clause)
    
    # Build the full query
    values_str = ", ".join(values_clauses)
    query = f"INSERT INTO {table_name} ({column_str}) VALUES {values_str}"
    
    # Execute the insert
    await target_connector.execute_query(query, params)

async def _get_last_sync_time(
    source_client_id: str,
    target_client_id: str,
    table_name: str
) -> Optional[datetime]:
    """
    Get the last sync time for a table.
    
    Args:
        source_client_id: Source client ID
        target_client_id: Target client ID
        table_name: Table name
        
    Returns:
        Last sync time or None if no previous sync
    """
    # In a real implementation, you'd query a sync_history table
    # For demonstration, we'll return None (forcing full sync)
    return None

async def _store_last_sync_time(
    source_client_id: str,
    target_client_id: str,
    table_name: str
) -> None:
    """
    Store the last sync time for a table.
    
    Args:
        source_client_id: Source client ID
        target_client_id: Target client ID
        table_name: Table name
    """
    # In a real implementation, you'd store in a sync_history table
    # For demonstration, we'll just log
    logger.info(f"Storing sync time for {source_client_id}->{target_client_id} table {table_name}: {datetime.now()}")

async def store_sync_history(status: SyncStatus) -> None:
    """
    Store sync history in the database.
    
    Args:
        status: Sync status object
    """
    # In a real implementation, you'd store in a sync_history table
    # For demonstration, we'll just log
    logger.info(f"Storing sync history: {status.sync_id}, success: {status.success}")

async def get_sync_status(
    sync_id: Optional[str] = None,
    client_id: Optional[str] = None,
    connection_id: Optional[str] = None
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get status of sync operations.
    
    Args:
        sync_id: Optional specific sync ID
        client_id: Optional client ID to filter by
        connection_id: Optional connection ID to filter by
        
    Returns:
        Status information for the sync(s)
    """
    # If sync_id is provided, return just that sync
    if sync_id and sync_id in _active_syncs:
        return _active_syncs[sync_id].to_dict()
    
    # Otherwise, filter by client/connection
    results = []
    for status in _active_syncs.values():
        if client_id and status.client_id != client_id:
            continue
            
        if connection_id and status.connection_id != connection_id:
            continue
            
        results.append(status.to_dict())
    
    # Sort by start time, newest first
    results.sort(key=lambda x: x["start_time"], reverse=True)
    
    return results

async def cancel_sync(sync_id: str) -> bool:
    """
    Cancel a running sync operation.
    
    Args:
        sync_id: Sync ID to cancel
        
    Returns:
        True if canceled, False if not found or already complete
    """
    if sync_id in _active_syncs:
        status = _active_syncs[sync_id]
        if status.is_running:
            status.complete(False)
            status.add_error("Sync canceled by user")
            return True
    
    return False