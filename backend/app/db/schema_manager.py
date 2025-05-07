from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime
import asyncio
import json
import hashlib
import os
import re


from app.db.schema.schema_discovery import discover_client_schema, get_connector_for_client
from app.db.schema.schema_mapper import get_domain_mappings, update_domain_mappings
from app.utils.logger import get_logger
from app.config import get_settings


# Initialize logger
logger = get_logger(__name__)


# Get settings
settings = get_settings()


# Schema cache
_schema_cache: Dict[str, Dict[str, Any]] = {}
_schema_hash_cache: Dict[str, Dict[str, str]] = {}
_schema_timestamp_cache: Dict[str, Dict[str, datetime]] = {}


class SchemaManager:
    """
    Manager for database schema information.
    
    This class provides methods to manage database schema information,
    including caching, refreshing, and analyzing schemas.
    """
    
    def __init__(self, client_id: Optional[str] = None, connection_id: Optional[str] = None):
        """
        Initialize the schema manager.
        
        Args:
            client_id: Optional client ID
            connection_id: Optional connection ID
        """
        self.client_id = client_id
        self.connection_id = connection_id
        self.cache_ttl = settings.schema_cache_ttl or 3600  # 1 hour default
    
    async def get_schema(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get the database schema for a client.
        
        Args:
            force_refresh: Whether to force a refresh of the cached schema
            
        Returns:
            The database schema
        """
        if not self.client_id:
            raise ValueError("Client ID is required")
        
        cache_key = self.client_id
        conn_key = self.connection_id or "default"
        
        # Check if we need to refresh the cache
        should_refresh = force_refresh or await self._should_refresh_cache(cache_key, conn_key)
        
        if should_refresh:
            # Discover the schema
            schema = await discover_client_schema(
                client_id=self.client_id,
                connection_id=self.connection_id,
                force_refresh=True
            )
            
            # Convert schema to dictionary if it's not already
            schema_dict = schema.to_dict() if hasattr(schema, "to_dict") else schema
            
            # Cache the schema
            await self._cache_schema(cache_key, conn_key, schema_dict)
            
            return schema_dict
        
        # Return from cache
        if cache_key in _schema_cache and conn_key in _schema_cache[cache_key]:
            return _schema_cache[cache_key][conn_key]
        
        # If not in cache but shouldn't refresh, discover again
        schema = await discover_client_schema(
            client_id=self.client_id,
            connection_id=self.connection_id
        )
        
        schema_dict = schema.to_dict() if hasattr(schema, "to_dict") else schema
        
        # Cache the schema
        await self._cache_schema(cache_key, conn_key, schema_dict)
        
        return schema_dict
    
    async def get_tables(self) -> List[str]:
        """
        Get a list of tables in the database.
        
        Returns:
            List of table names
        """
        schema = await self.get_schema()
        return [table["name"] for table in schema.get("tables", [])]
    
    async def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the schema for a specific table.
        
        Args:
            table_name: Table name
            
        Returns:
            Table schema or None if not found
        """
        schema = await self.get_schema()
        for table in schema.get("tables", []):
            if table["name"] == table_name:
                return table
        return None
    
    async def get_relationships(self) -> List[Dict[str, str]]:
        """
        Get all relationships between tables.
        
        Returns:
            List of relationships
        """
        schema = await self.get_schema()
        return schema.get("relationships", [])
    
    async def get_domain_mappings(self) -> List[Dict[str, Any]]:
        """
        Get domain mappings for the client schema.
        
        Returns:
            List of domain mappings
        """
        return await get_domain_mappings(self.client_id)
    
    async def update_domain_mappings(self, mappings: List[Dict[str, Any]]) -> int:
        """
        Update domain mappings for the client schema.
        
        Args:
            mappings: List of domain mappings to update
            
        Returns:
            Number of mappings updated
        """
        return await update_domain_mappings(mappings)
    
    async def analyze_schema(self) -> Dict[str, Any]:
        """
        Analyze the database schema to extract statistics and insights.
        
        Returns:
            Schema analysis results
        """
        schema = await self.get_schema()
        tables = schema.get("tables", [])
        
        # Calculate basic statistics
        table_count = len(tables)
        total_columns = sum(len(table.get("columns", [])) for table in tables)
        avg_columns_per_table = total_columns / table_count if table_count > 0 else 0
        
        # Analyze table sizes
        table_sizes = []
        connector = await get_connector_for_client(self.client_id, self.connection_id)
        
        try:
            for table in tables:
                table_name = table["name"]
                try:
                    query = f"SELECT COUNT(*) as row_count FROM {table_name}"
                    result = await connector.execute_query(query)
                    row_count = result["data"][0].get("row_count", 0) if result["data"] else 0
                    
                    table_sizes.append({
                        "table_name": table_name,
                        "row_count": row_count,
                        "column_count": len(table.get("columns", []))
                    })
                except Exception as e:
                    logger.warning(f"Error getting row count for table {table_name}: {str(e)}")
                    table_sizes.append({
                        "table_name": table_name,
                        "row_count": None,
                        "column_count": len(table.get("columns", []))
                    })
        finally:
            await connector.close()
        
        # Analyze data types
        data_types = {}
        for table in tables:
            for column in table.get("columns", []):
                data_type = column.get("data_type", "unknown").lower()
                if data_type in data_types:
                    data_types[data_type] += 1
                else:
                    data_types[data_type] = 1
        
        # Analyze primary and foreign keys
        tables_with_pk = sum(1 for table in tables if table.get("primary_key"))
        total_foreign_keys = sum(len(table.get("foreign_keys", [])) for table in tables)
        
        # Build analysis result
        analysis = {
            "table_count": table_count,
            "total_columns": total_columns,
            "avg_columns_per_table": avg_columns_per_table,
            "tables_with_primary_key": tables_with_pk,
            "total_foreign_keys": total_foreign_keys,
            "data_type_distribution": data_types,
            "table_sizes": table_sizes
        }
        
        return analysis
    
    async def get_recommended_mappings(self) -> List[Dict[str, Any]]:
        """
        Generate recommended domain mappings based on schema analysis.
        
        Returns:
            List of recommended mappings
        """
        schema = await self.get_schema()
        
        # Common patterns to look for in table and column names
        domain_patterns = {
            "product": ["product", "item", "merchandise", "goods", "sku"],
            "inventory": ["inventory", "stock", "on_hand"],
            "supplier": ["supplier", "vendor", "provider"],
            "order": ["order", "purchase_order", "sales_order", "po"],
            "warehouse": ["warehouse", "location", "facility", "storage"],
            "customer": ["customer", "client", "buyer"],
            "shipment": ["shipment", "delivery", "transport"],
            "transaction": ["transaction", "ledger", "entry"]
        }
        
        # Recommended mappings
        recommended_mappings = []
        
        # Analyze tables
        for table in schema.get("tables", []):
            table_name = table["name"].lower()
            table_concept = None
            table_confidence = 0.0
            
            # Check if table name matches a domain concept
            for concept, patterns in domain_patterns.items():
                for pattern in patterns:
                    if pattern in table_name:
                        table_concept = concept
                        table_confidence = 0.7  # Good match but not exact
                        if table_name == pattern or table_name == f"{pattern}s":
                            table_confidence = 0.9  # Exact match
                        break
                
                if table_concept:
                    break
            
            # If we found a match, map the table
            if table_concept:
                # Basic mapping for the table
                mapping = {
                    "client_id": self.client_id,
                    "custom_table": table["name"],
                    "custom_column": None,
                    "domain_concept": table_concept,
                    "domain_attribute": "table",
                    "confidence": table_confidence,
                    "manual_override": False,
                    "last_updated": datetime.now().isoformat()
                }
                
                recommended_mappings.append(mapping)
                
                # Now map key columns
                for column in table.get("columns", []):
                    column_name = column["name"].lower()
                    
                    # ID columns
                    if column_name in ["id", f"{table_concept}_id", "code"]:
                        recommended_mappings.append({
                            "client_id": self.client_id,
                            "custom_table": table["name"],
                            "custom_column": column["name"],
                            "domain_concept": table_concept,
                            "domain_attribute": "id",
                            "confidence": 0.9,
                            "manual_override": False,
                            "last_updated": datetime.now().isoformat()
                        })
                    
                    # Name columns
                    elif column_name in ["name", "description", f"{table_concept}_name"]:
                        recommended_mappings.append({
                            "client_id": self.client_id,
                            "custom_table": table["name"],
                            "custom_column": column["name"],
                            "domain_concept": table_concept,
                            "domain_attribute": "name",
                            "confidence": 0.8,
                            "manual_override": False,
                            "last_updated": datetime.now().isoformat()
                        })
                    
                    # Standard tracking columns
                    elif column_name in ["created_at", "date_created"]:
                        recommended_mappings.append({
                            "client_id": self.client_id,
                            "custom_table": table["name"],
                            "custom_column": column["name"],
                            "domain_concept": table_concept,
                            "domain_attribute": "created_at",
                            "confidence": 0.9,
                            "manual_override": False,
                            "last_updated": datetime.now().isoformat()
                        })
                    
                    elif column_name in ["updated_at", "date_updated", "last_modified"]:
                        recommended_mappings.append({
                            "client_id": self.client_id,
                            "custom_table": table["name"],
                            "custom_column": column["name"],
                            "domain_concept": table_concept,
                            "domain_attribute": "updated_at",
                            "confidence": 0.9,
                            "manual_override": False,
                            "last_updated": datetime.now().isoformat()
                        })
                    
                    # Status column
                    elif column_name in ["status", "state"]:
                        recommended_mappings.append({
                            "client_id": self.client_id,
                            "custom_table": table["name"],
                            "custom_column": column["name"],
                            "domain_concept": table_concept,
                            "domain_attribute": "status",
                            "confidence": 0.8,
                            "manual_override": False,
                            "last_updated": datetime.now().isoformat()
                        })
                    
                    # Common specific columns
                    if table_concept == "product":
                        if column_name in ["price", "cost", "msrp"]:
                            attr = "price" if "price" in column_name else "cost"
                            recommended_mappings.append({
                                "client_id": self.client_id,
                                "custom_table": table["name"],
                                "custom_column": column["name"],
                                "domain_concept": table_concept,
                                "domain_attribute": attr,
                                "confidence": 0.8,
                                "manual_override": False,
                                "last_updated": datetime.now().isoformat()
                            })
                    
                    elif table_concept == "inventory":
                        if column_name in ["quantity", "qty", "on_hand", "stock_level"]:
                            recommended_mappings.append({
                                "client_id": self.client_id,
                                "custom_table": table["name"],
                                "custom_column": column["name"],
                                "domain_concept": table_concept,
                                "domain_attribute": "quantity",
                                "confidence": 0.8,
                                "manual_override": False,
                                "last_updated": datetime.now().isoformat()
                            })
        
        return recommended_mappings
    
    async def export_schema(self, format: str = "json") -> str:
        """
        Export the schema in various formats.
        
        Args:
            format: Export format (json, sql, yaml)
            
        Returns:
            Schema in the requested format
        """
        schema = await self.get_schema()
        
        if format.lower() == "json":
            return json.dumps(schema, indent=2)
        
        elif format.lower() == "sql":
            # Generate CREATE TABLE statements
            sql = []
            
            for table in schema.get("tables", []):
                table_name = table["name"]
                columns = table.get("columns", [])
                primary_key = table.get("primary_key", [])
                
                # Start the CREATE TABLE statement
                sql.append(f"CREATE TABLE {table_name} (")
                
                # Add columns
                column_defs = []
                for column in columns:
                    column_name = column["name"]
                    data_type = column.get("data_type", "")
                    nullable = column.get("nullable", True)
                    default = column.get("default")
                    
                    # Build column definition
                    col_def = f"    {column_name} {data_type}"
                    
                    if not nullable:
                        col_def += " NOT NULL"
                    
                    if default is not None:
                        col_def += f" DEFAULT {default}"
                    
                    column_defs.append(col_def)
                
                # Add primary key
                if primary_key:
                    pk_columns = ", ".join(primary_key)
                    column_defs.append(f"    PRIMARY KEY ({pk_columns})")
                
                # Join column definitions
                sql.append(",\n".join(column_defs))
                sql.append(");\n")
            
            # Add foreign key constraints
            for table in schema.get("tables", []):
                table_name = table["name"]
                foreign_keys = table.get("foreign_keys", [])
                
                for fk in foreign_keys:
                    if "column" not in fk or "references" not in fk:
                        continue
                    
                    column = fk["column"]
                    reference = fk["references"]
                    
                    # Extract reference parts
                    ref_parts = reference.split(".")
                    if len(ref_parts) < 3:
                        continue
                    
                    ref_schema = ref_parts[0]
                    ref_table = ref_parts[1]
                    ref_column = ref_parts[2]
                    
                    # Generate constraint name
                    constraint_name = f"fk_{table_name}_{column}_{ref_table}_{ref_column}"
                    constraint_name = constraint_name[:63]  # Limit length
                    
                    # Generate ALTER TABLE statement
                    sql.append(f"ALTER TABLE {table_name}")
                    sql.append(f"    ADD CONSTRAINT {constraint_name}")
                    sql.append(f"    FOREIGN KEY ({column})")
                    sql.append(f"    REFERENCES {ref_table} ({ref_column});\n")
            
            return "\n".join(sql)
        
        elif format.lower() == "yaml":
            # Simple YAML representation
            yaml = []
            
            yaml.append("tables:")
            for table in schema.get("tables", []):
                table_name = table["name"]
                yaml.append(f"  - name: {table_name}")
                
                # Add columns
                yaml.append("    columns:")
                for column in table.get("columns", []):
                    column_name = column["name"]
                    data_type = column.get("data_type", "")
                    nullable = column.get("nullable", True)
                    
                    yaml.append(f"      - name: {column_name}")
                    yaml.append(f"        type: {data_type}")
                    yaml.append(f"        nullable: {str(nullable).lower()}")
                
                # Add primary key
                if table.get("primary_key"):
                    pk_columns = ", ".join(table["primary_key"])
                    yaml.append(f"    primary_key: [{pk_columns}]")
                
                # Add foreign keys
                if table.get("foreign_keys"):
                    yaml.append("    foreign_keys:")
                    for fk in table["foreign_keys"]:
                        yaml.append(f"      - column: {fk.get('column', '')}")
                        yaml.append(f"        references: {fk.get('references', '')}")
            
            return "\n".join(yaml)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _should_refresh_cache(self, client_id: str, connection_id: str) -> bool:
        """
        Determine if the cache should be refreshed.
        
        Args:
            client_id: Client ID
            connection_id: Connection ID
            
        Returns:
            True if the cache should be refreshed, False otherwise
        """
        # If not in cache, should refresh
        if client_id not in _schema_timestamp_cache or connection_id not in _schema_timestamp_cache[client_id]:
            return True
        
        # Check cache TTL
        cache_time = _schema_timestamp_cache[client_id][connection_id]
        elapsed_seconds = (datetime.now() - cache_time).total_seconds()
        
        if elapsed_seconds > self.cache_ttl:
            return True
        
        # Check for schema changes if within TTL
        return await self._check_schema_changed(client_id, connection_id)
    
    async def _check_schema_changed(self, client_id: str, connection_id: str) -> bool:
        """
        Check if the schema has changed since it was cached.
        
        Args:
            client_id: Client ID
            connection_id: Connection ID
            
        Returns:
            True if the schema has changed, False otherwise
        """
        if client_id not in _schema_hash_cache or connection_id not in _schema_hash_cache[client_id]:
            return True
        
        try:
            # Get current schema and calculate hash
            schema = await discover_client_schema(
                client_id=client_id,
                connection_id=connection_id
            )
            
            schema_dict = schema.to_dict() if hasattr(schema, "to_dict") else schema
            current_hash = self._calculate_schema_hash(schema_dict)
            
            # Compare with cached hash
            cached_hash = _schema_hash_cache[client_id][connection_id]
            
            return current_hash != cached_hash
            
        except Exception as e:
            logger.error(f"Error checking schema changes: {str(e)}")
            # On error, assume schema has changed to force refresh
            return True
    
    async def _cache_schema(self, client_id: str, connection_id: str, schema: Dict[str, Any]) -> None:
        """
        Cache a schema.
        
        Args:
            client_id: Client ID
            connection_id: Connection ID
            schema: Schema to cache
        """
        # Initialize cache dictionaries if needed
        if client_id not in _schema_cache:
            _schema_cache[client_id] = {}
        
        if client_id not in _schema_hash_cache:
            _schema_hash_cache[client_id] = {}
        
        if client_id not in _schema_timestamp_cache:
            _schema_timestamp_cache[client_id] = {}
        
        # Cache the schema
        _schema_cache[client_id][connection_id] = schema
        
        # Calculate and cache hash
        schema_hash = self._calculate_schema_hash(schema)
        _schema_hash_cache[client_id][connection_id] = schema_hash
        
        # Cache timestamp
        _schema_timestamp_cache[client_id][connection_id] = datetime.now()
    
    def _calculate_schema_hash(self, schema: Dict[str, Any]) -> str:
        """
        Calculate a hash of the schema for change detection.
        
        Args:
            schema: Schema to hash
            
        Returns:
            Hash of the schema
        """
        # Convert to JSON and hash
        schema_json = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()


async def get_schema_manager(client_id: Optional[str] = None, connection_id: Optional[str] = None) -> SchemaManager:
    """
    Get a schema manager instance.
    
    Args:
        client_id: Optional client ID
        connection_id: Optional connection ID
        
    Returns:
        SchemaManager instance
    """
    return SchemaManager(client_id=client_id, connection_id=connection_id)


async def clear_schema_cache(client_id: Optional[str] = None, connection_id: Optional[str] = None) -> None:
    """
    Clear the schema cache.
    
    Args:
        client_id: Optional client ID (if None, clears all)
        connection_id: Optional connection ID (if None, clears all for client)
    """
    global _schema_cache, _schema_hash_cache, _schema_timestamp_cache
    
    if client_id is None:
        # Clear all cache
        _schema_cache = {}
        _schema_hash_cache = {}
        _schema_timestamp_cache = {}
        logger.info("Cleared all schema cache")
        return
    
    if connection_id is None:
        # Clear all for client
        if client_id in _schema_cache:
            del _schema_cache[client_id]
        
        if client_id in _schema_hash_cache:
            del _schema_hash_cache[client_id]
        
        if client_id in _schema_timestamp_cache:
            del _schema_timestamp_cache[client_id]
        
        logger.info(f"Cleared schema cache for client {client_id}")
        return
    
    # Clear specific client/connection
    if client_id in _schema_cache and connection_id in _schema_cache[client_id]:
        del _schema_cache[client_id][connection_id]
    
    if client_id in _schema_hash_cache and connection_id in _schema_hash_cache[client_id]:
        del _schema_hash_cache[client_id][connection_id]
    
    if client_id in _schema_timestamp_cache and connection_id in _schema_timestamp_cache[client_id]:
        del _schema_timestamp_cache[client_id][connection_id]
    
    logger.info(f"Cleared schema cache for client {client_id}, connection {connection_id}")


async def get_available_concepts() -> Dict[str, List[str]]:
    """
    Get a list of available domain concepts and attributes.
    
    Returns:
        Dictionary mapping domain concepts to their attributes
    """
    # This would typically be loaded from a configuration file or database
    # For now, we'll define a static map of common supply chain concepts
    return {
        "product": ["id", "name", "description", "sku", "category", "price", "cost", "weight", "dimensions", "status", "created_at", "updated_at"],
        "inventory": ["id", "product", "location", "quantity", "safety_stock", "reorder_point", "max_stock", "status", "created_at", "updated_at"],
        "supplier": ["id", "name", "contact", "email", "phone", "address", "category", "tier", "status", "created_at", "updated_at"],
        "order": ["id", "number", "date", "supplier", "customer", "status", "total_amount", "tax", "shipping", "created_at", "updated_at"],
        "purchase_order": ["id", "number", "date", "supplier", "status", "total_amount", "expected_delivery", "actual_delivery", "created_at", "updated_at"],
        "sales_order": ["id", "number", "date", "customer", "status", "total_amount", "shipping_date", "delivery_date", "created_at", "updated_at"],
        "warehouse": ["id", "name", "code", "address", "capacity", "status", "created_at", "updated_at"],
        "shipment": ["id", "order", "carrier", "tracking_number", "status", "ship_date", "delivery_date", "created_at", "updated_at"],
        "customer": ["id", "name", "contact", "email", "phone", "address", "segment", "status", "created_at", "updated_at"]
    }