# app/cache/result_cache.py

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import hashlib
import time
from datetime import datetime, timedelta
import asyncio
import os
import io
import gzip
import pickle

from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

class ResultCache:
    """
    Cache for query results.
    
    This class caches the results of database queries to improve performance
    for repeated queries.
    """
    
    def __init__(
        self,
        max_cache_size: int = 200,
        ttl_seconds: int = 300,  # 5 minutes default TTL
        max_result_size: int = 10 * 1024 * 1024,  # 10MB default max size
        cache_dir: Optional[str] = None,
        compress: bool = True
    ):
        """
        Initialize the result cache.
        
        Args:
            max_cache_size: Maximum number of cache entries
            ttl_seconds: Time-to-live in seconds for cache entries
            max_result_size: Maximum size of a result to cache (in bytes)
            cache_dir: Optional directory for persistent cache
            compress: Whether to compress cached results
        """
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.max_result_size = max_result_size
        self.cache_dir = cache_dir or settings.result_cache_dir
        self.compress = compress
        
        # In-memory cache
        self.cache = {}  # {key: (timestamp, metadata, value)}
        self.size_map = {}  # {key: size_in_bytes}
        self.current_size = 0  # Total size of cache in bytes
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.inserts = 0
        self.evictions = 0
        
        # Create cache directory if it doesn't exist
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Load cache from disk if available
        self._load_cache()
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def get(
        self,
        sql_query: str,
        params: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a cached query result.
        
        Args:
            sql_query: SQL query
            params: Query parameters
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(sql_query, params, client_id, connection_id)
        
        # Check if key exists in cache and not expired
        if key in self.cache:
            timestamp, metadata, value = self.cache[key]
            if time.time() - timestamp <= self.ttl_seconds:
                # Update stats
                self.hits += 1
                logger.debug(f"Result cache hit for SQL query: {sql_query[:50]}...")
                return value
            else:
                # Entry expired, remove it
                logger.debug(f"Result cache expired for SQL query: {sql_query[:50]}...")
                await self._remove_entry(key)
                
        # Update stats
        self.misses += 1
        logger.debug(f"Result cache miss for SQL query: {sql_query[:50]}...")
        return None
        
    async def set(
        self,
        sql_query: str,
        result: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Set a cache entry.
        
        Args:
            sql_query: SQL query
            result: Query result to cache
            params: Query parameters
            client_id: Optional client ID
            connection_id: Optional connection ID
            metadata: Optional metadata about the result
            
        Returns:
            True if cached successfully, False otherwise
        """
        # Check if result is cacheable
        if not self._is_cacheable(sql_query, result):
            logger.debug(f"Result not cacheable for SQL query: {sql_query[:50]}...")
            return False
            
        # Generate key and estimate size
        key = self._generate_key(sql_query, params, client_id, connection_id)
        size = self._estimate_size(result)
        
        # Check if result is too large
        if size > self.max_result_size:
            logger.debug(f"Result too large to cache ({size} bytes): {sql_query[:50]}...")
            return False
            
        # Check if cache would exceed size limit
        if key not in self.cache and self.current_size + size > self.max_cache_size * 1024 * 1024:
            # Evict entries until we have enough space
            evicted = await self._make_space(size)
            if not evicted:
                logger.warning(f"Could not make enough space in cache for result ({size} bytes)")
                return False
                
        # Prepare metadata
        meta = metadata or {}
        meta.update({
            "sql": sql_query,
            "params": params,
            "client_id": client_id,
            "connection_id": connection_id,
            "size": size,
            "timestamp": time.time()
        })
        
        # Add to cache
        self.cache[key] = (time.time(), meta, result)
        
        # Update size tracking
        old_size = self.size_map.get(key, 0)
        self.size_map[key] = size
        self.current_size = self.current_size - old_size + size
        
        # Update stats
        self.inserts += 1
        
        # Save to disk if persistent cache is enabled
        if self.cache_dir:
            await self._save_entry(key, meta, result)
            
        logger.debug(f"Cached result ({size} bytes) for SQL query: {sql_query[:50]}...")
        return True
        
    async def invalidate(
        self,
        sql_query: Optional[str] = None,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        table_names: Optional[List[str]] = None
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            sql_query: Optional specific SQL query to invalidate
            client_id: Optional client ID to invalidate
            connection_id: Optional connection ID to invalidate
            table_names: Optional list of table names to invalidate
            
        Returns:
            Number of invalidated entries
        """
        count = 0
        keys_to_remove = []
        
        if sql_query:
            # Invalidate specific query
            key = self._generate_key(sql_query, None, client_id, connection_id)
            if key in self.cache:
                keys_to_remove.append(key)
        elif table_names:
            # Invalidate entries referencing specific tables
            for key, (_, metadata, _) in self.cache.items():
                sql = metadata.get("sql", "").lower()
                if any(table.lower() in sql for table in table_names):
                    keys_to_remove.append(key)
        elif client_id:
            # Invalidate all entries for a specific client
            for key, (_, metadata, _) in self.cache.items():
                if metadata.get("client_id") == client_id:
                    if not connection_id or metadata.get("connection_id") == connection_id:
                        keys_to_remove.append(key)
        else:
            # Invalidate all entries
            keys_to_remove = list(self.cache.keys())
            
        # Remove entries
        for key in keys_to_remove:
            await self._remove_entry(key)
            count += 1
            
        logger.info(f"Invalidated {count} result cache entries")
        return count
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        total_requests = self.hits + self.misses
        hit_ratio = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "inserts": self.inserts,
            "evictions": self.evictions,
            "hit_ratio": hit_ratio,
            "entry_count": len(self.cache),
            "size_bytes": self.current_size,
            "max_size_bytes": self.max_cache_size * 1024 * 1024
        }
        
    def _generate_key(
        self,
        sql_query: str,
        params: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ) -> str:
        """
        Generate a cache key from a query and context.
        
        Args:
            sql_query: SQL query
            params: Query parameters
            client_id: Optional client ID
            connection_id: Optional connection ID
            
        Returns:
            Cache key string
        """
        # Normalize query (remove extra whitespace)
        normalized_query = " ".join(sql_query.split())
        
        # Build key components
        key_parts = [normalized_query]
        
        if params:
            # Sort parameters for consistency
            sorted_params = json.dumps(params, sort_keys=True)
            key_parts.append(f"params:{sorted_params}")
            
        if client_id:
            key_parts.append(f"client:{client_id}")
            
        if connection_id:
            key_parts.append(f"conn:{connection_id}")
            
        # Generate hash of combined key
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
        
    async def _remove_entry(self, key: str) -> None:
        """
        Remove an entry from the cache.
        
        Args:
            key: Cache key to remove
        """
        if key in self.cache:
            # Update size tracking
            size = self.size_map.get(key, 0)
            self.current_size -= size
            
            # Remove from cache
            del self.cache[key]
            
            if key in self.size_map:
                del self.size_map[key]
                
            # Remove from disk if persistent cache is enabled
            if self.cache_dir:
                cache_file = os.path.join(self.cache_dir, f"{key}.result")
                if os.path.exists(cache_file):
                    try:
                        os.remove(cache_file)
                    except Exception as e:
                        logger.warning(f"Error removing cache file {cache_file}: {str(e)}")
                        
            # Update stats
            self.evictions += 1
    
    async def _make_space(self, required_size: int) -> bool:
        """
        Make space in the cache for a new entry.
        
        Args:
            required_size: Size in bytes needed
            
        Returns:
            True if space was made, False if couldn't make enough space
        """
        # If cache is empty, we can't make any space
        if not self.cache:
            return self.max_cache_size * 1024 * 1024 >= required_size
            
        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(
            [(key, ts) for key, (ts, _, _) in self.cache.items()],
            key=lambda x: x[1]
        )
        
        # Remove entries until we have enough space
        space_freed = 0
        for key, _ in sorted_entries:
            if self.current_size + required_size - space_freed <= self.max_cache_size * 1024 * 1024:
                # We have enough space now
                return True
                
            # Remove this entry
            size = self.size_map.get(key, 0)
            space_freed += size
            await self._remove_entry(key)
            
        # Check if we've made enough space
        return self.current_size + required_size <= self.max_cache_size * 1024 * 1024
        
    def _is_cacheable(self, sql_query: str, result: Dict[str, Any]) -> bool:
        """
        Check if a result should be cached.
        
        Args:
            sql_query: SQL query
            result: Query result
            
        Returns:
            True if cacheable, False otherwise
        """
        # Don't cache empty results
        if not result or not result.get("data"):
            return False
            
        # Don't cache very large results
        if self._estimate_size(result) > self.max_result_size:
            return False
            
        # Don't cache queries with certain statements (INSERT, UPDATE, DELETE, etc.)
        non_cacheable = ["INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT"]
        sql_upper = sql_query.upper()
        if any(stmt in sql_upper for stmt in non_cacheable):
            return False
            
        # Only cache SELECT statements (typically)
        if not sql_upper.strip().startswith("SELECT"):
            return False
            
        return True
        
    def _estimate_size(self, obj: Any) -> int:
        """
        Estimate the size of an object in bytes.
        
        Args:
            obj: Object to estimate size for
            
        Returns:
            Estimated size in bytes
        """
        try:
            # Use pickle to get a more accurate size
            buffer = io.BytesIO()
            if self.compress:
                with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump(obj, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            return buffer.getbuffer().nbytes
        except Exception as e:
            # Fallback: convert to JSON and get string length
            try:
                json_str = json.dumps(obj)
                return len(json_str.encode())
            except Exception:
                # If all else fails, use a rough estimate
                return 1024  # 1KB default assumption
                
    async def _save_entry(
        self,
        key: str,
        metadata: Dict[str, Any],
        value: Dict[str, Any]
    ) -> None:
        """
        Save a cache entry to disk.
        
        Args:
            key: Cache key
            metadata: Entry metadata
            value: Value to save
        """
        if not self.cache_dir:
            return
            
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.result")
            
            # Combine metadata and value for storage
            cache_data = {
                "metadata": metadata,
                "value": value
            }
            
            # Save with compression if enabled
            async with asyncio.Lock():
                if self.compress:
                    with gzip.open(cache_file, "wb") as f:
                        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(cache_file, "wb") as f:
                        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                        
        except Exception as e:
            logger.error(f"Error saving result cache entry to disk: {str(e)}")
            
    def _load_cache(self) -> None:
        """
        Load cache from disk.
        """
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return
            
        try:
            # Load cache files
            count = 0
            total_size = 0
            current_time = time.time()
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".result"):
                    try:
                        cache_file = os.path.join(self.cache_dir, filename)
                        key = filename[:-7]  # Remove .result extension
                        
                        # Load with decompression if enabled
                        if self.compress:
                            with gzip.open(cache_file, "rb") as f:
                                cache_data = pickle.load(f)
                        else:
                            with open(cache_file, "rb") as f:
                                cache_data = pickle.load(f)
                                
                        metadata = cache_data.get("metadata", {})
                        value = cache_data.get("value")
                        timestamp = metadata.get("timestamp", 0)
                        size = metadata.get("size", 0) or os.path.getsize(cache_file)
                        
                        # Only load non-expired entries
                        if current_time - timestamp <= self.ttl_seconds and value:
                            self.cache[key] = (timestamp, metadata, value)
                            self.size_map[key] = size
                            total_size += size
                            count += 1
                            
                    except Exception as e:
                        logger.warning(f"Error loading cache file {filename}: {str(e)}")
                        
            # Update current size
            self.current_size = total_size
            logger.info(f"Loaded {count} result cache entries from disk ({total_size} bytes)")
            
        except Exception as e:
            logger.error(f"Error loading result cache from disk: {str(e)}")
            
    async def _cleanup_loop(self) -> None:
        """
        Background task to clean up expired cache entries.
        """
        while True:
            try:
                # Sleep for a while (check every minute)
                await asyncio.sleep(60)
                
                # Find expired entries
                current_time = time.time()
                expired_keys = []
                
                for key, (timestamp, _, _) in self.cache.items():
                    if current_time - timestamp > self.ttl_seconds:
                        expired_keys.append(key)
                        
                # Remove expired entries
                for key in expired_keys:
                    await self._remove_entry(key)
                    
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired result cache entries")
                    
            except asyncio.CancelledError:
                # Task cancelled, exit
                break
            except Exception as e:
                logger.error(f"Error in result cache cleanup loop: {str(e)}")
                # Wait a bit before retrying
                await asyncio.sleep(60)
                
    async def shutdown(self) -> None:
        """
        Shutdown the cache, cancel background tasks and flush to disk.
        """
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
                
        # Save all entries to disk
        if self.cache_dir:
            for key, (timestamp, metadata, value) in self.cache.items():
                # Update timestamp in metadata
                metadata["timestamp"] = timestamp
                await self._save_entry(key, metadata, value)
                
        logger.info("Result cache shutdown complete")

# Singleton instance
_result_cache = None

async def get_result_cache() -> ResultCache:
    """
    Get the global result cache instance.
    
    Returns:
        ResultCache instance
    """
    global _result_cache
    
    if _result_cache is None:
        cache_size = settings.result_cache_size or 200
        ttl_seconds = settings.result_cache_ttl or 300
        max_result_size = settings.max_result_size or 10 * 1024 * 1024
        cache_dir = settings.result_cache_dir
        compress = settings.compress_result_cache if hasattr(settings, 'compress_result_cache') else True
        
        _result_cache = ResultCache(
            max_cache_size=cache_size,
            ttl_seconds=ttl_seconds,
            max_result_size=max_result_size,
            cache_dir=cache_dir,
            compress=compress
        )
        
    return _result_cache

async def shutdown_result_cache() -> None:
    """
    Shutdown the result cache.
    
    This should be called during application shutdown.
    """
    global _result_cache
    
    if _result_cache:
        await _result_cache.shutdown()
        _result_cache = None