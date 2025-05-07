# app/cache/query_cache.py

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import hashlib
import time
from datetime import datetime, timedelta
import asyncio
import os

from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

class QueryCache:
    """
    Cache for natural language queries to SQL translations.
    
    This class caches the results of translating natural language queries
    to SQL to avoid unnecessary LLM calls for repeated queries.
    """
    
    def __init__(
        self,
        max_cache_size: int = 1000,
        ttl_seconds: int = 3600,  # 1 hour default TTL
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the query cache.
        
        Args:
            max_cache_size: Maximum number of cache entries
            ttl_seconds: Time-to-live in seconds for cache entries
            cache_dir: Optional directory for persistent cache
        """
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.cache_dir = cache_dir or settings.cache_dir
        
        # In-memory cache
        self.cache = {}  # {key: (timestamp, value)}
        self.access_count = {}  # {key: access_count}
        
        # Create cache directory if it doesn't exist
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Load cache from disk if available
        self._load_cache()
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def get(
        self,
        query: str,
        schema_hash: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a cached query result.
        
        Args:
            query: Natural language query
            schema_hash: Optional hash of database schema
            context: Optional additional context
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(query, schema_hash, context)
        
        # Check if key exists in cache and not expired
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp <= self.ttl_seconds:
                # Update access count
                self.access_count[key] = self.access_count.get(key, 0) + 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return value
            else:
                # Entry expired, remove it
                logger.debug(f"Cache expired for query: {query[:50]}...")
                self._remove_entry(key)
                
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
        
    async def set(
        self,
        query: str,
        result: Dict[str, Any],
        schema_hash: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set a cache entry.
        
        Args:
            query: Natural language query
            result: Query result to cache
            schema_hash: Optional hash of database schema
            context: Optional additional context
        """
        key = self._generate_key(query, schema_hash, context)
        
        # Check if cache is full
        if len(self.cache) >= self.max_cache_size and key not in self.cache:
            # Evict least recently used item
            self._evict_lru()
            
        # Add to cache
        self.cache[key] = (time.time(), result)
        self.access_count[key] = 1
        
        # Save to disk if persistent cache is enabled
        if self.cache_dir:
            await self._save_entry(key, result)
            
        logger.debug(f"Cached query: {query[:50]}...")
        
    def invalidate(
        self,
        query: Optional[str] = None,
        schema_hash: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            query: Optional specific query to invalidate
            schema_hash: Optional schema hash to invalidate
            context: Optional context to invalidate
            
        Returns:
            Number of invalidated entries
        """
        count = 0
        
        if query:
            # Invalidate specific query
            key = self._generate_key(query, schema_hash, context)
            if key in self.cache:
                self._remove_entry(key)
                count += 1
        elif schema_hash:
            # Invalidate all entries for a specific schema
            keys_to_remove = []
            for key in self.cache.keys():
                if schema_hash in key:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                self._remove_entry(key)
                count += 1
        else:
            # Invalidate all entries
            count = len(self.cache)
            self.cache.clear()
            self.access_count.clear()
            
            # Remove all persistent cache files
            if self.cache_dir and os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith(".cache"):
                        os.remove(os.path.join(self.cache_dir, filename))
        
        logger.info(f"Invalidated {count} cache entries")
        return count
    
    def _generate_key(
        self,
        query: str,
        schema_hash: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a cache key from a query and context.
        
        Args:
            query: Natural language query
            schema_hash: Optional hash of database schema
            context: Optional additional context
            
        Returns:
            Cache key string
        """
        # Normalize query (lowercase, remove extra whitespace)
        normalized_query = " ".join(query.lower().split())
        
        # Build key components
        key_parts = [normalized_query]
        
        if schema_hash:
            key_parts.append(f"schema:{schema_hash}")
            
        if context:
            # Extract relevant parts of context to include in key
            relevant_context = {}
            for key in ["client_id", "connection_id", "user_id"]:
                if key in context:
                    relevant_context[key] = context[key]
                    
            if relevant_context:
                context_str = json.dumps(relevant_context, sort_keys=True)
                key_parts.append(f"context:{context_str}")
                
        # Generate hash of combined key
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def _remove_entry(self, key: str) -> None:
        """
        Remove an entry from the cache.
        
        Args:
            key: Cache key to remove
        """
        if key in self.cache:
            del self.cache[key]
            
        if key in self.access_count:
            del self.access_count[key]
            
        # Remove from disk if persistent cache is enabled
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{key}.cache")
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except Exception as e:
                    logger.warning(f"Error removing cache file {cache_file}: {str(e)}")
    
    def _evict_lru(self) -> None:
        """
        Evict the least recently used cache entry.
        """
        if not self.cache:
            return
            
        # Find entry with lowest access count
        min_access = float('inf')
        lru_key = None
        
        for key, count in self.access_count.items():
            if count < min_access:
                min_access = count
                lru_key = key
                
        # If multiple entries have the same access count, use the oldest
        if min_access == 1:  # Multiple entries with count=1
            oldest_time = float('inf')
            for key in [k for k, c in self.access_count.items() if c == 1]:
                timestamp, _ = self.cache[key]
                if timestamp < oldest_time:
                    oldest_time = timestamp
                    lru_key = key
                    
        # Remove the entry
        if lru_key:
            self._remove_entry(lru_key)
            logger.debug(f"Evicted LRU cache entry with key: {lru_key}")
            
    async def _cleanup_loop(self) -> None:
        """
        Background task to clean up expired cache entries.
        """
        while True:
            try:
                # Sleep for a while (check hourly)
                await asyncio.sleep(3600)
                
                # Find expired entries
                current_time = time.time()
                expired_keys = []
                
                for key, (timestamp, _) in self.cache.items():
                    if current_time - timestamp > self.ttl_seconds:
                        expired_keys.append(key)
                        
                # Remove expired entries
                for key in expired_keys:
                    self._remove_entry(key)
                    
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except asyncio.CancelledError:
                # Task cancelled, exit
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {str(e)}")
                # Wait a bit before retrying
                await asyncio.sleep(60)
                
    async def _save_entry(self, key: str, value: Dict[str, Any]) -> None:
        """
        Save a cache entry to disk.
        
        Args:
            key: Cache key
            value: Value to save
        """
        if not self.cache_dir:
            return
            
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.cache")
            cache_data = {
                "timestamp": time.time(),
                "value": value
            }
            
            async with asyncio.Lock():
                with open(cache_file, "w") as f:
                    json.dump(cache_data, f)
                    
        except Exception as e:
            logger.error(f"Error saving cache entry to disk: {str(e)}")
            
    def _load_cache(self) -> None:
        """
        Load cache from disk.
        """
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return
            
        try:
            # Load cache files
            count = 0
            current_time = time.time()
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".cache"):
                    try:
                        cache_file = os.path.join(self.cache_dir, filename)
                        key = filename[:-6]  # Remove .cache extension
                        
                        with open(cache_file, "r") as f:
                            cache_data = json.load(f)
                            
                        timestamp = cache_data.get("timestamp", 0)
                        value = cache_data.get("value")
                        
                        # Only load non-expired entries
                        if current_time - timestamp <= self.ttl_seconds and value:
                            self.cache[key] = (timestamp, value)
                            self.access_count[key] = 1
                            count += 1
                            
                    except Exception as e:
                        logger.warning(f"Error loading cache file {filename}: {str(e)}")
                        
            logger.info(f"Loaded {count} cache entries from disk")
            
        except Exception as e:
            logger.error(f"Error loading cache from disk: {str(e)}")
            
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
            for key, (_, value) in self.cache.items():
                await self._save_entry(key, value)
                
        logger.info("Query cache shutdown complete")

# Singleton instance
_query_cache = None

async def get_query_cache() -> QueryCache:
    """
    Get the global query cache instance.
    
    Returns:
        QueryCache instance
    """
    global _query_cache
    
    if _query_cache is None:
        cache_size = settings.query_cache_size or 1000
        ttl_seconds = settings.query_cache_ttl or 3600
        cache_dir = settings.query_cache_dir
        
        _query_cache = QueryCache(
            max_cache_size=cache_size,
            ttl_seconds=ttl_seconds,
            cache_dir=cache_dir
        )
        
    return _query_cache

async def shutdown_query_cache() -> None:
    """
    Shutdown the query cache.
    
    This should be called during application shutdown.
    """
    global _query_cache
    
    if _query_cache:
        await _query_cache.shutdown()
        _query_cache = None