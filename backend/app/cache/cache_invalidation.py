# app/cache/cache_invalidation.py

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import asyncio
import time
from datetime import datetime, timedelta
import re

from app.utils.logger import get_logger
from app.config import get_settings
from app.cache.query_cache import get_query_cache
from app.cache.result_cache import get_result_cache

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

class CacheInvalidationManager:
    """
    Manager for cache invalidation logic.
    
    This class handles the coordination of cache invalidation across different cache types
    and implements strategies like table-based invalidation.
    """
    
    def __init__(self):
        """Initialize the cache invalidation manager."""
        self.query_patterns = {
            "insert": re.compile(r"INSERT\s+INTO\s+(\w+)", re.IGNORECASE),
            "update": re.compile(r"UPDATE\s+(\w+)", re.IGNORECASE),
            "delete": re.compile(r"DELETE\s+FROM\s+(\w+)", re.IGNORECASE),
            "truncate": re.compile(r"TRUNCATE\s+TABLE\s+(\w+)", re.IGNORECASE),
            "alter": re.compile(r"ALTER\s+TABLE\s+(\w+)", re.IGNORECASE),
            "create": re.compile(r"CREATE\s+TABLE\s+(\w+)", re.IGNORECASE),
            "drop": re.compile(r"DROP\s+TABLE\s+(\w+)", re.IGNORECASE)
        }
        
        # Track changed tables
        self.changed_tables = {}  # {client_id: {table_name: timestamp}}
        
        # Background task for periodic invalidation
        self.invalidation_task = None
        
    async def start(self) -> None:
        """Start the invalidation manager."""
        if not self.invalidation_task:
            self.invalidation_task = asyncio.create_task(self._periodic_invalidation())
            logger.info("Cache invalidation manager started")
            
    async def stop(self) -> None:
        """Stop the invalidation manager."""
        if self.invalidation_task:
            self.invalidation_task.cancel()
            try:
                await self.invalidation_task
            except asyncio.CancelledError:
                pass
                
            self.invalidation_task = None
            logger.info("Cache invalidation manager stopped")
            
    async def invalidate_for_sql_query(
        self,
        sql_query: str,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Invalidate caches based on an SQL query.
        
        Args:
            sql_query: SQL query that triggered invalidation
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Tuple of (query_cache_count, result_cache_count) invalidated entries
        """
        # Extract affected tables
        affected_tables = self.extract_affected_tables(sql_query)
        
        # Update changed tables tracking
        if client_id and affected_tables:
            if client_id not in self.changed_tables:
                self.changed_tables[client_id] = {}
                
            current_time = time.time()
            for table in affected_tables:
                self.changed_tables[client_id][table] = current_time
                
        # Invalidate caches
        query_count = await self.invalidate_query_cache(client_id, connection_id)
        result_count = await self.invalidate_result_cache(affected_tables, client_id, connection_id)
        
        logger.info(f"Invalidated {query_count} query cache entries and {result_count} result cache entries for tables: {affected_tables}")
        return query_count, result_count
        
    async def invalidate_for_tables(
        self,
        table_names: List[str],
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        invalidate_query_cache: bool = True
    ) -> Tuple[int, int]:
        """
        Invalidate caches based on table names.
        
        Args:
            table_names: List of affected table names
            client_id: Optional client ID
            connection_id: Optional connection ID
            invalidate_query_cache: Whether to invalidate query cache too
            
        Returns:
            Tuple of (query_cache_count, result_cache_count) invalidated entries
        """
        # Update changed tables tracking
        if client_id and table_names:
            if client_id not in self.changed_tables:
                self.changed_tables[client_id] = {}
                
            current_time = time.time()
            for table in table_names:
                self.changed_tables[client_id][table] = current_time
                
        # Invalidate caches
        query_count = 0
        if invalidate_query_cache:
            query_count = await self.invalidate_query_cache(client_id, connection_id)
            
        result_count = await self.invalidate_result_cache(table_names, client_id, connection_id)
        
        logger.info(f"Invalidated {query_count} query cache entries and {result_count} result cache entries for tables: {table_names}")
        return query_count, result_count
        
    async def invalidate_all(
        self,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Invalidate all caches.
        
        Args:
            client_id: Optional client ID to limit invalidation
            connection_id: Optional connection ID to limit invalidation
            
        Returns:
            Tuple of (query_cache_count, result_cache_count) invalidated entries
        """
        # Invalidate query cache
        query_count = await self.invalidate_query_cache(client_id, connection_id)
        
        # Invalidate result cache
        result_count = await self.invalidate_result_cache(None, client_id, connection_id)
        
        # Clear changed tables tracking
        if client_id is None:
            self.changed_tables.clear()
        elif client_id in self.changed_tables:
            del self.changed_tables[client_id]
            
        logger.info(f"Invalidated all cache entries: {query_count} query cache entries and {result_count} result cache entries")
        return query_count, result_count
        
    async def invalidate_query_cache(
        self,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> int:
        """
        Invalidate the query cache.
        
        Args:
            client_id: Optional client ID to limit invalidation
            connection_id: Optional connection ID to limit invalidation
            
        Returns:
            Number of invalidated entries
        """
        try:
            query_cache = await get_query_cache()
            
            # Create context for invalidation
            context = {}
            if client_id:
                context["client_id"] = client_id
            if connection_id:
                context["connection_id"] = connection_id
                
            # Invalidate entries
            if context:
                count = query_cache.invalidate(schema_hash=None, context=context)
            else:
                count = query_cache.invalidate()
                
            return count
            
        except Exception as e:
            logger.error(f"Error invalidating query cache: {str(e)}")
            return 0
            
    async def invalidate_result_cache(
        self,
        table_names: Optional[List[str]] = None,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> int:
        """
        Invalidate the result cache.
        
        Args:
            table_names: Optional list of table names to invalidate
            client_id: Optional client ID to limit invalidation
            connection_id: Optional connection ID to limit invalidation
            
        Returns:
            Number of invalidated entries
        """
        try:
            result_cache = await get_result_cache()
            
            # Invalidate entries
            count = await result_cache.invalidate(
                client_id=client_id,
                connection_id=connection_id,
                table_names=table_names
            )
            
            return count
            
        except Exception as e:
            logger.error(f"Error invalidating result cache: {str(e)}")
            return 0
            
    def extract_affected_tables(self, sql_query: str) -> List[str]:
        """
        Extract affected table names from an SQL query.
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            List of affected table names
        """
        affected_tables = []
        
        # Check each pattern
        for pattern_name, pattern in self.query_patterns.items():
            matches = pattern.findall(sql_query)
            affected_tables.extend(matches)
            
        # Also check for SELECT statements
        # This is a simplified approach - in a real implementation, you'd use a proper SQL parser
        if "SELECT" in sql_query.upper():
            # Extract table names from FROM clause
            from_match = re.search(r"FROM\s+(\w+)", sql_query, re.IGNORECASE)
            if from_match:
                affected_tables.append(from_match.group(1))
                
            # Extract table names from JOIN clauses
            join_matches = re.findall(r"JOIN\s+(\w+)", sql_query, re.IGNORECASE)
            affected_tables.extend(join_matches)
            
        # Remove duplicates
        return list(set(affected_tables))
        
    async def _periodic_invalidation(self) -> None:
        """
        Background task for periodic cache invalidation.
        """
        # Get invalidation interval from settings
        interval = settings.cache_invalidation_interval if hasattr(settings, 'cache_invalidation_interval') else 300  # Default: 5 minutes
        
        while True:
            try:
                # Sleep until next invalidation
                await asyncio.sleep(interval)
                
                # Check for tables that need invalidation
                current_time = time.time()
                tables_to_invalidate = {}  # {client_id: [table_names]}
                
                for client_id, tables in self.changed_tables.items():
                    for table, timestamp in list(tables.items()):
                        # Check if enough time has passed since last change
                        if current_time - timestamp > 60:  # Wait at least 1 minute after last change
                            if client_id not in tables_to_invalidate:
                                tables_to_invalidate[client_id] = []
                                
                            tables_to_invalidate[client_id].append(table)
                            # Remove from changed tables
                            del tables[table]
                            
                # Invalidate caches for each client
                for client_id, tables in tables_to_invalidate.items():
                    if tables:
                        await self.invalidate_for_tables(tables, client_id=client_id)
                        
            except asyncio.CancelledError:
                # Task cancelled, exit
                break
            except Exception as e:
                logger.error(f"Error in periodic cache invalidation: {str(e)}")
                # Wait a bit before retrying
                await asyncio.sleep(60)

# Singleton instance
_invalidation_manager = None

async def get_invalidation_manager() -> CacheInvalidationManager:
    """
    Get the global cache invalidation manager instance.
    
    Returns:
        CacheInvalidationManager instance
    """
    global _invalidation_manager
    
    if _invalidation_manager is None:
        _invalidation_manager = CacheInvalidationManager()
        await _invalidation_manager.start()
        
    return _invalidation_manager

async def shutdown_invalidation_manager() -> None:
    """
    Shutdown the cache invalidation manager.
    
    This should be called during application shutdown.
    """
    global _invalidation_manager
    
    if _invalidation_manager:
        await _invalidation_manager.stop()
        _invalidation_manager = None

# Utility functions for cache invalidation
async def invalidate_on_data_change(
    sql_query: str,
    client_id: Optional[str] = None,
    connection_id: Optional[str] = None
) -> None:
    """
    Invalidate caches when data changes.
    
    This should be called when executing data-modifying SQL queries.
    
    Args:
        sql_query: SQL query that modified data
        client_id: Optional client ID
        connection_id: Optional connection ID
    """
    manager = await get_invalidation_manager()
    await manager.invalidate_for_sql_query(sql_query, client_id, connection_id)

async def invalidate_on_schema_change(
    client_id: str,
    connection_id: Optional[str] = None
) -> None:
    """
    Invalidate caches when schema changes.
    
    This should be called when detecting schema changes.
    
    Args:
        client_id: Client ID
        connection_id: Optional connection ID
    """
    manager = await get_invalidation_manager()
    await manager.invalidate_all(client_id, connection_id)