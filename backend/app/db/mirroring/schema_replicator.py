from typing import Dict, List, Any, Optional, Set, Tuple
import asyncio
from datetime import datetime
import re

from app.db.connectors.postgres import PostgresConnector
from app.db.connectors.mysql import MySQLConnector
from app.db.connectors.sqlserver import SQLServerConnector
from app.db.connectors.oracle import OracleConnector
from app.db.schema.schema_discovery import get_connector_for_client, discover_client_schema
from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Type mappings between different database systems
# These are simplified - in a production system, you'd need more nuanced mappings
TYPE_MAPPINGS = {
    # From PostgreSQL types to other database types
    "postgres": {
        "mysql": {
            "integer": "INT",
            "bigint": "BIGINT",
            "smallint": "SMALLINT",
            "text": "TEXT",
            "character varying": "VARCHAR",
            "varchar": "VARCHAR",
            "char": "CHAR",
            "boolean": "TINYINT(1)",
            "date": "DATE",
            "timestamp": "DATETIME",
            "numeric": "DECIMAL",
            "decimal": "DECIMAL",
            "real": "FLOAT",
            "double precision": "DOUBLE",
            "json": "JSON",
            "jsonb": "JSON",
            "uuid": "CHAR(36)",
            "bytea": "BLOB",
            "interval": "VARCHAR(255)"
        },
        "sqlserver": {
            "integer": "INT",
            "bigint": "BIGINT",
            "smallint": "SMALLINT",
            "text": "NVARCHAR(MAX)",
            "character varying": "NVARCHAR",
            "varchar": "NVARCHAR",
            "char": "NCHAR",
            "boolean": "BIT",
            "date": "DATE",
            "timestamp": "DATETIME2",
            "numeric": "DECIMAL",
            "decimal": "DECIMAL",
            "real": "REAL",
            "double precision": "FLOAT",
            "json": "NVARCHAR(MAX)",
            "jsonb": "NVARCHAR(MAX)",
            "uuid": "UNIQUEIDENTIFIER",
            "bytea": "VARBINARY(MAX)",
            "interval": "VARCHAR(255)"
        },
        "oracle": {
            "integer": "NUMBER(10)",
            "bigint": "NUMBER(19)",
            "smallint": "NUMBER(5)",
            "text": "CLOB",
            "character varying": "VARCHAR2",
            "varchar": "VARCHAR2",
            "char": "CHAR",
            "boolean": "NUMBER(1)",
            "date": "DATE",
            "timestamp": "TIMESTAMP",
            "numeric": "NUMBER",
            "decimal": "NUMBER",
            "real": "BINARY_FLOAT",
            "double precision": "BINARY_DOUBLE",
            "json": "CLOB",
            "jsonb": "CLOB",
            "uuid": "VARCHAR2(36)",
            "bytea": "BLOB",
            "interval": "INTERVAL DAY TO SECOND"
        }
    },
    # Add mappings from other systems as needed
    # For brevity, not including all combinations
}

async def replicate_schema(
    source_client_id: str,
    source_connection_id: Optional[str] = None,
    target_client_id: str = None,
    target_connection_id: Optional[str] = None,
    target_db_type: str = "postgres",
    specific_tables: Optional[List[str]] = None,
    exclude_tables: Optional[List[str]] = None,
    include_data: bool = False
) -> Dict[str, Any]:
    """
    Replicate schema from a source database to a target database.
    
    Args:
        source_client_id: Source client ID
        source_connection_id: Optional source connection ID
        target_client_id: Target client ID (if None, uses a new connection for same client)
        target_connection_id: Optional target connection ID
        target_db_type: Target database type (postgres, mysql, sqlserver, oracle)
        specific_tables: Optional list of specific tables to replicate
        exclude_tables: Optional list of tables to exclude
        include_data: Whether to include data in replication
        
    Returns:
        Replication result information
    """
    # If target_client_id not specified, use the same as source
    if not target_client_id:
        target_client_id = source_client_id
    
    # Get source database schema
    schema = await discover_client_schema(
        client_id=source_client_id,
        connection_id=source_connection_id,
        force_refresh=True,
        specific_tables=specific_tables
    )
    
    # Get source database type
    source_db_type = schema.database_type or "postgres"
    
    # Create connectors for source and target
    source_connector = await get_connector_for_client(source_client_id, source_connection_id)
    
    # For target, we need to create a connector of the specified type
    if target_db_type == "mysql":
        target_connector = MySQLConnector(client_id=target_client_id, connection_id=target_connection_id)
    elif target_db_type == "sqlserver":
        target_connector = SQLServerConnector(client_id=target_client_id, connection_id=target_connection_id)
    elif target_db_type == "oracle":
        target_connector = OracleConnector(client_id=target_client_id, connection_id=target_connection_id)
    else:
        # Default to PostgreSQL
        target_connector = PostgresConnector(client_id=target_client_id, connection_id=target_connection_id)
    
    try:
        # Test connections
        if not await source_connector.test_connection():
            raise ValueError("Could not connect to source database")
        
        if not await target_connector.test_connection():
            raise ValueError("Could not connect to target database")
        
        # Start replication
        start_time = datetime.now()
        tables_replicated = 0
        errors = []
        
        # Process each table in the schema
        for table in schema.tables:
            table_name = table["name"]
            
            # Skip if in exclude list
            if exclude_tables and table_name in exclude_tables:
                logger.info(f"Skipping excluded table {table_name}")
                continue
            
            # Only include specific tables if specified
            if specific_tables and table_name not in specific_tables:
                continue
            
            try:
                # Generate DDL for target database
                ddl = await generate_create_table_ddl(
                    table=table,
                    source_db_type=source_db_type,
                    target_db_type=target_db_type
                )
                
                # Execute DDL on target database
                pool = await target_connector._get_connection_pool()
                async with pool.acquire() as conn:
                    # In a real implementation, you'd use the appropriate method based on the target DB type
                    # This is a simplified example for PostgreSQL
                    if hasattr(conn, "execute"):
                        await conn.execute(ddl)
                    else:
                        async with conn.cursor() as cursor:
                            await cursor.execute(ddl)
                
                # Replicate data if requested
                if include_data:
                    from app.db.mirroring.data_syncer import sync_table_data
                    await sync_table_data(
                        source_client_id=source_client_id,
                        source_connection_id=source_connection_id,
                        target_client_id=target_client_id,
                        target_connection_id=target_connection_id,
                        table_name=table_name,
                        batch_size=1000
                    )
                
                tables_replicated += 1
                logger.info(f"Replicated table {table_name}")
                
            except Exception as e:
                error_msg = f"Error replicating table {table_name}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # After all tables are created, add foreign keys
        if tables_replicated > 0:
            await add_foreign_keys(
                schema=schema,
                target_connector=target_connector,
                source_db_type=source_db_type,
                target_db_type=target_db_type,
                specific_tables=specific_tables,
                exclude_tables=exclude_tables
            )
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = {
            "success": len(errors) == 0,
            "tables_replicated": tables_replicated,
            "errors": errors,
            "duration_seconds": duration,
            "source_client_id": source_client_id,
            "source_connection_id": source_connection_id,
            "target_client_id": target_client_id,
            "target_connection_id": target_connection_id,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error during schema replication: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "source_client_id": source_client_id,
            "target_client_id": target_client_id,
            "timestamp": datetime.now().isoformat()
        }
    finally:
        # Close connections
        await source_connector.close()
        await target_connector.close()

async def generate_create_table_ddl(
    table: Dict[str, Any],
    source_db_type: str,
    target_db_type: str
) -> str:
    """
    Generate CREATE TABLE DDL for the target database.
    
    Args:
        table: Table schema information
        source_db_type: Source database type
        target_db_type: Target database type
        
    Returns:
        CREATE TABLE DDL statement
    """
    table_name = table["name"]
    schema_name = table.get("schema", "public")
    columns = table.get("columns", [])
    primary_key = table.get("primary_key", [])
    
    # Start building the DDL
    if target_db_type == "postgres":
        ddl = f'CREATE TABLE IF NOT EXISTS "{schema_name}"."{table_name}" (\n'
    elif target_db_type == "mysql":
        ddl = f"CREATE TABLE IF NOT EXISTS `{table_name}` (\n"
    elif target_db_type == "sqlserver":
        ddl = f"IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{table_name}' AND schema_id = SCHEMA_ID('{schema_name}'))\nBEGIN\nCREATE TABLE [{schema_name}].[{table_name}] (\n"
    else:  # Oracle
        ddl = f'BEGIN\nEXECUTE IMMEDIATE \'CREATE TABLE "{schema_name}"."{table_name}" (\n'
    
    # Add columns
    column_defs = []
    for column in columns:
        column_name = column["name"]
        data_type = column.get("data_type", "").lower()
        nullable = column.get("nullable", True)
        default = column.get("default")
        
        # Map data type from source to target
        target_type = map_data_type(data_type, source_db_type, target_db_type, column)
        
        # Build column definition based on target database type
        if target_db_type == "postgres":
            col_def = f'    "{column_name}" {target_type}'
        elif target_db_type == "mysql":
            col_def = f"    `{column_name}` {target_type}"
        elif target_db_type == "sqlserver":
            col_def = f"    [{column_name}] {target_type}"
        else:  # Oracle
            col_def = f'    "{column_name}" {target_type}'
        
        # Add NULL/NOT NULL constraint
        if not nullable:
            col_def += " NOT NULL"
        
        # Add default value if present
        if default is not None:
            # Format default value based on target database type
            if target_db_type == "postgres":
                col_def += f" DEFAULT {default}"
            elif target_db_type == "mysql":
                col_def += f" DEFAULT {default}"
            elif target_db_type == "sqlserver":
                col_def += f" DEFAULT {default}"
            else:  # Oracle
                col_def += f" DEFAULT {default}"
        
        column_defs.append(col_def)
    
    # Add primary key constraint if present
    if primary_key:
        if target_db_type == "postgres":
            pk_columns = ", ".join([f'"{col}"' for col in primary_key])
            column_defs.append(f'    PRIMARY KEY ({pk_columns})')
        elif target_db_type == "mysql":
            pk_columns = ", ".join([f"`{col}`" for col in primary_key])
            column_defs.append(f"    PRIMARY KEY ({pk_columns})")
        elif target_db_type == "sqlserver":
            pk_columns = ", ".join([f"[{col}]" for col in primary_key])
            column_defs.append(f"    PRIMARY KEY ({pk_columns})")
        else:  # Oracle
            pk_columns = ", ".join([f'"{col}"' for col in primary_key])
            column_defs.append(f'    PRIMARY KEY ({pk_columns})')
    
    # Combine column definitions
    ddl += ",\n".join(column_defs)
    
    # Close the CREATE TABLE statement
    if target_db_type == "postgres" or target_db_type == "mysql":
        ddl += "\n);"
    elif target_db_type == "sqlserver":
        ddl += "\n);\nEND;"
    else:  # Oracle
        ddl += "\n)\';\nEXCEPTION WHEN OTHERS THEN\n  IF SQLCODE != -955 THEN RAISE; END IF;\nEND;"
    
    return ddl

def map_data_type(
    source_type: str,
    source_db_type: str,
    target_db_type: str,
    column: Dict[str, Any]
) -> str:
    """
    Map a data type from source database type to target database type.
    
    Args:
        source_type: Source data type
        source_db_type: Source database type
        target_db_type: Target database type
        column: Column information (for additional details like length)
        
    Returns:
        Mapped data type for target database
    """
    # If source and target are the same, return as is
    if source_db_type == target_db_type:
        return source_type
    
    # For character types, include length if specified
    max_length = column.get("max_length")
    precision = column.get("precision")
    scale = column.get("scale")
    
    # Clean up source type (remove length/precision)
    base_type = re.sub(r'\(.*\)', '', source_type).strip().lower()
    
    # If we have a mapping for this source type
    if source_db_type in TYPE_MAPPINGS and target_db_type in TYPE_MAPPINGS[source_db_type]:
        type_map = TYPE_MAPPINGS[source_db_type][target_db_type]
        if base_type in type_map:
            target_type = type_map[base_type]
            
            # Add length/precision if needed
            if "varchar" in target_type.lower() or "char" in target_type.lower():
                if max_length:
                    if "MAX" not in target_type:
                        target_type = re.sub(r'\(.*\)', '', target_type)
                        target_type += f"({max_length})"
            elif "decimal" in target_type.lower() or "numeric" in target_type.lower():
                if precision:
                    target_type = re.sub(r'\(.*\)', '', target_type)
                    if scale is not None:
                        target_type += f"({precision},{scale})"
                    else:
                        target_type += f"({precision})"
            
            return target_type
    
    # If no mapping found, try to use as is (may not work)
    logger.warning(f"No mapping found for type {source_type} from {source_db_type} to {target_db_type}")
    return source_type

async def add_foreign_keys(
    schema: Any,
    target_connector: Any,
    source_db_type: str,
    target_db_type: str,
    specific_tables: Optional[List[str]] = None,
    exclude_tables: Optional[List[str]] = None
) -> None:
    """
    Add foreign key constraints after all tables are created.
    
    Args:
        schema: Database schema
        target_connector: Target database connector
        source_db_type: Source database type
        target_db_type: Target database type
        specific_tables: Optional list of specific tables
        exclude_tables: Optional list of tables to exclude
    """
    # For each table
    for table in schema.tables:
        table_name = table["name"]
        schema_name = table.get("schema", "public")
        
        # Skip if in exclude list
        if exclude_tables and table_name in exclude_tables:
            continue
        
        # Only include specific tables if specified
        if specific_tables and table_name not in specific_tables:
            continue
        
        # Get foreign keys
        foreign_keys = table.get("foreign_keys", [])
        
        for fk in foreign_keys:
            try:
                # Extract source column and reference
                if "column" not in fk or "references" not in fk:
                    continue
                    
                source_column = fk["column"]
                reference = fk["references"]
                
                # Parse the reference (format: schema.table.column)
                ref_parts = reference.split(".")
                if len(ref_parts) < 3:
                    continue
                    
                ref_schema = ref_parts[0]
                ref_table = ref_parts[1]
                ref_column = ref_parts[2]
                
                # Skip if referenced table is excluded
                if exclude_tables and ref_table in exclude_tables:
                    continue
                
                # Skip if specific tables specified and referenced table not included
                if specific_tables and ref_table not in specific_tables:
                    continue
                
                # Generate FK constraint name
                constraint_name = f"fk_{table_name}_{source_column}_{ref_table}_{ref_column}"
                constraint_name = constraint_name[:63]  # Limit length for PostgreSQL
                
                # Generate ALTER TABLE statement for the target database
                if target_db_type == "postgres":
                    sql = f"""
                    ALTER TABLE "{schema_name}"."{table_name}" 
                    ADD CONSTRAINT "{constraint_name}" 
                    FOREIGN KEY ("{source_column}") 
                    REFERENCES "{ref_schema}"."{ref_table}" ("{ref_column}");
                    """
                elif target_db_type == "mysql":
                    sql = f"""
                    ALTER TABLE `{table_name}` 
                    ADD CONSTRAINT `{constraint_name}` 
                    FOREIGN KEY (`{source_column}`) 
                    REFERENCES `{ref_table}` (`{ref_column}`);
                    """
                elif target_db_type == "sqlserver":
                    sql = f"""
                    ALTER TABLE [{schema_name}].[{table_name}] 
                    ADD CONSTRAINT [{constraint_name}] 
                    FOREIGN KEY ([{source_column}]) 
                    REFERENCES [{ref_schema}].[{ref_table}] ([{ref_column}]);
                    """
                else:  # Oracle
                    sql = f"""
                    ALTER TABLE "{schema_name}"."{table_name}" 
                    ADD CONSTRAINT "{constraint_name}" 
                    FOREIGN KEY ("{source_column}") 
                    REFERENCES "{ref_schema}"."{ref_table}" ("{ref_column}")
                    """
                
                # Execute the statement
                pool = await target_connector._get_connection_pool()
                async with pool.acquire() as conn:
                    # In a real implementation, you'd use the appropriate method based on the target DB type
                    # This is a simplified example for PostgreSQL
                    if hasattr(conn, "execute"):
                        await conn.execute(sql)
                    else:
                        async with conn.cursor() as cursor:
                            await cursor.execute(sql)
                
                logger.info(f"Added foreign key constraint {constraint_name}")
                
            except Exception as e:
                logger.error(f"Error adding foreign key constraint: {str(e)}")
    
async def create_mirror_databases(
    client_ids: List[str],
    target_db_type: str = "postgres"
) -> Dict[str, Any]:
    """
    Create mirror databases for multiple clients.
    
    Args:
        client_ids: List of client IDs to mirror
        target_db_type: Target database type
        
    Returns:
        Dictionary with results for each client
    """
    results = {}
    
    for client_id in client_ids:
        try:
            result = await replicate_schema(
                source_client_id=client_id,
                target_db_type=target_db_type,
                include_data=True
            )
            results[client_id] = result
        except Exception as e:
            results[client_id] = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    return results