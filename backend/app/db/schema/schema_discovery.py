from typing import Dict, List, Any, Optional, Tuple, Union
import asyncio
import re
from datetime import datetime
import json

from app.db.connectors.postgres import PostgresConnector
from app.db.connectors.mysql import MySQLConnector
from app.db.connectors.sqlserver import SQLServerConnector
from app.db.connectors.oracle import OracleConnector
from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Database schema model
class DatabaseSchema:
    """Database schema information"""
    def __init__(self, 
                 tables: List[Dict[str, Any]] = None,
                 views: List[Dict[str, Any]] = None,
                 relationships: List[Dict[str, Any]] = None,
                 client_id: Optional[str] = None,
                 connection_id: Optional[str] = None,
                 discovery_time: Optional[datetime] = None,
                 database_type: Optional[str] = None):
        self.tables = tables or []
        self.views = views or []
        self.relationships = relationships or []
        self.client_id = client_id
        self.connection_id = connection_id
        self.discovery_time = discovery_time or datetime.now()
        self.database_type = database_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "tables": self.tables,
            "views": self.views,
            "relationships": self.relationships,
            "client_id": self.client_id,
            "connection_id": self.connection_id,
            "discovery_time": self.discovery_time.isoformat() if self.discovery_time else None,
            "database_type": self.database_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatabaseSchema':
        """Create from dictionary"""
        # Parse datetime string if present
        discovery_time = None
        if data.get("discovery_time"):
            try:
                discovery_time = datetime.fromisoformat(data["discovery_time"])
            except ValueError:
                discovery_time = datetime.now()
        
        return cls(
            tables=data.get("tables", []),
            views=data.get("views", []),
            relationships=data.get("relationships", []),
            client_id=data.get("client_id"),
            connection_id=data.get("connection_id"),
            discovery_time=discovery_time,
            database_type=data.get("database_type")
        )

# Schema cache to avoid repeated discovery
_schema_cache: Dict[str, Dict[str, DatabaseSchema]] = {}

async def get_connector_for_client(
    client_id: str,
    connection_id: Optional[str] = None
) -> Union[PostgresConnector, MySQLConnector, SQLServerConnector, OracleConnector]:
    """
    Get the appropriate database connector for a client.
    
    Args:
        client_id: Client ID
        connection_id: Optional connection ID
        
    Returns:
        Database connector instance
    """
    # In a real implementation, you would look up the connection type from a database or settings
    # For demonstration, we'll use hardcoded values
    
    # Get connection metadata
    # In production, this would query a database to get connection metadata
    connection_type = "postgres"  # Default to PostgreSQL
    
    # Check specific client/connection combinations
    if client_id == "client-1":
        if connection_id == "mysql-1":
            connection_type = "mysql"
        elif connection_id == "sqlserver-1":
            connection_type = "sqlserver"
        elif connection_id == "oracle-1":
            connection_type = "oracle"
    elif client_id == "client-2":
        if connection_id == "mysql-2":
            connection_type = "mysql"
        elif connection_id == "sqlserver-2":
            connection_type = "sqlserver"
        elif connection_id == "oracle-2":
            connection_type = "oracle"
    
    # Create and return the appropriate connector
    if connection_type == "mysql":
        return MySQLConnector(client_id=client_id, connection_id=connection_id)
    elif connection_type == "sqlserver":
        return SQLServerConnector(client_id=client_id, connection_id=connection_id)
    elif connection_type == "oracle":
        return OracleConnector(client_id=client_id, connection_id=connection_id)
    else:
        # Default to PostgreSQL
        return PostgresConnector(client_id=client_id, connection_id=connection_id)

async def get_table_schema(
    client_id: str,
    table_name: str,
    schema: Optional[str] = None,
    connection_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get the schema for a specific table.
    
    Args:
        client_id: Client ID
        table_name: Table name
        schema: Optional schema name
        connection_id: Optional connection ID
        
    Returns:
        Table schema information
    """
    # Get the connector for this client
    connector = await get_connector_for_client(client_id, connection_id)
    
    try:
        # Get the table schema from the connector
        table_schema = await connector.get_table_schema(table_name, schema)
        return table_schema
    except Exception as e:
        logger.error(f"Error getting table schema: {str(e)}")
        return None
    finally:
        # Close the connector
        await connector.close()

async def discover_client_schema(
    client_id: str,
    connection_id: Optional[str] = None,
    force_refresh: bool = False,
    include_views: bool = True,
    include_relationships: bool = True,
    specific_tables: Optional[List[str]] = None,
    specific_schemas: Optional[List[str]] = None
) -> DatabaseSchema:
    """
    Discover and analyze the database schema for a client.
    
    Args:
        client_id: Client ID
        connection_id: Optional connection ID
        force_refresh: Whether to force refresh the schema
        include_views: Whether to include views
        include_relationships: Whether to include table relationships
        specific_tables: Optional list of specific tables to discover
        specific_schemas: Optional list of schemas to limit discovery
        
    Returns:
        Discovered database schema
    """
    # Check cache first
    cache_key = f"{client_id}:{connection_id or 'default'}"
    if not force_refresh and client_id in _schema_cache and cache_key in _schema_cache[client_id]:
        cached_schema = _schema_cache[client_id][cache_key]
        
        # Check if cache is still valid (less than 1 hour old)
        cache_age = (datetime.now() - cached_schema.discovery_time).total_seconds()
        if cache_age < 3600:  # 1 hour
            logger.info(f"Using cached schema for {cache_key}")
            return cached_schema
    
    # Get the connector for this client
    connector = await get_connector_for_client(client_id, connection_id)
    database_type = connector.__class__.__name__.replace("Connector", "").lower()
    
    try:
        logger.info(f"Discovering schema for client {client_id} using {database_type}")
        
        # Get list of schemas to scan
        schemas_to_scan = specific_schemas
        if not schemas_to_scan:
            # Use default schema for database type
            if database_type == "postgres":
                schemas_to_scan = ["public"]
            elif database_type == "sqlserver":
                schemas_to_scan = ["dbo"]
            elif database_type == "oracle":
                # For Oracle, get connection parameters to extract the username
                if hasattr(connector, "_get_connection_params"):
                    conn_params = await connector._get_connection_params()
                    schemas_to_scan = [conn_params.get("user", "").upper()]
            else:
                # For MySQL, schema is the database name
                schemas_to_scan = [None]  # None means use the current database
        
        # Get all tables for each schema
        all_tables = []
        all_views = []
        
        for schema in schemas_to_scan:
            # Get tables in this schema
            tables = await connector.get_tables(schema)
            
            # Filter to specific tables if requested
            if specific_tables:
                tables = [t for t in tables if t in specific_tables]
            
            # Process each table
            for table_name in tables:
                try:
                    # Get table schema
                    table_schema = await connector.get_table_schema(table_name, schema)
                    if table_schema:
                        all_tables.append(table_schema)
                except Exception as e:
                    logger.error(f"Error processing table {table_name}: {str(e)}")
            
            # Get views if requested
            if include_views:
                # This would be implemented based on database type
                # For now, we'll leave it empty
                pass
        
        # Discover relationships if requested
        relationships = []
        if include_relationships and all_tables:
            from app.db.schema.relationship_detector import detect_relationships
            relationships = await detect_relationships(all_tables, database_type)
        
        # Create schema object
        schema = DatabaseSchema(
            tables=all_tables,
            views=all_views,
            relationships=relationships,
            client_id=client_id,
            connection_id=connection_id,
            discovery_time=datetime.now(),
            database_type=database_type
        )
        
        # Cache the result
        if client_id not in _schema_cache:
            _schema_cache[client_id] = {}
        _schema_cache[client_id][cache_key] = schema
        
        logger.info(f"Schema discovery complete: {len(all_tables)} tables, {len(all_views)} views, {len(relationships)} relationships")
        return schema
        
    except Exception as e:
        logger.error(f"Error discovering schema: {str(e)}")
        raise
    finally:
        # Close the connector
        await connector.close()

async def refresh_schema_cache(client_id: str, connection_id: Optional[str] = None) -> bool:
    """
    Refresh the schema cache for a specific client/connection.
    
    Args:
        client_id: Client ID
        connection_id: Optional connection ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Force discovery to refresh cache
        await discover_client_schema(
            client_id=client_id,
            connection_id=connection_id,
            force_refresh=True
        )
        return True
    except Exception as e:
        logger.error(f"Error refreshing schema cache: {str(e)}")
        return False

async def clear_schema_cache(client_id: Optional[str] = None) -> None:
    """
    Clear the schema cache.
    
    Args:
        client_id: Optional client ID to clear only that client's cache
    """
    global _schema_cache
    
    if client_id:
        if client_id in _schema_cache:
            del _schema_cache[client_id]
            logger.info(f"Cleared schema cache for client {client_id}")
    else:
        _schema_cache = {}
        logger.info("Cleared all schema caches")

async def export_schema(client_id: str, connection_id: Optional[str] = None, format: str = "json") -> str:
    """
    Export the schema to a specific format.
    
    Args:
        client_id: Client ID
        connection_id: Optional connection ID
        format: Export format (json, yaml, sql)
        
    Returns:
        Schema in the requested format
    """
    # Get the schema
    schema = await discover_client_schema(
        client_id=client_id,
        connection_id=connection_id
    )
    
    # Convert to the requested format
    if format.lower() == "json":
        return json.dumps(schema.to_dict(), indent=2)
    elif format.lower() == "yaml":
        try:
            import yaml
            return yaml.dump(schema.to_dict())
        except ImportError:
            return json.dumps(schema.to_dict(), indent=2)
    elif format.lower() == "sql":
        # Generate SQL for creating the schema
        # This would be a more complex implementation
        return "-- SQL schema export not implemented yet"
    else:
        raise ValueError(f"Unsupported format: {format}")