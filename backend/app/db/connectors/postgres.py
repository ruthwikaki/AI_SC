import asyncpg
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import re
from datetime import datetime, date
import uuid

from app.utils.logger import get_logger
from app.config import get_settings

# Get settings
settings = get_settings()

# Initialize logger
logger = get_logger(__name__)

class PostgresConnector:
    """
    PostgreSQL database connector.
    
    This connector provides methods to interact with PostgreSQL databases,
    with support for dynamic client connections and schema discovery.
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        schema: Optional[str] = None,
        ssl_enabled: bool = False
    ):
        """
        Initialize the PostgreSQL connector.
        
        Args:
            client_id: Optional client ID to load connection from settings
            connection_id: Optional connection ID to load specific connection
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
            schema: Database schema
            ssl_enabled: Whether to use SSL for the connection
        """
        self.client_id = client_id
        self.connection_id = connection_id
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.schema = schema
        self.ssl_enabled = ssl_enabled
        self.pool = None
    
    async def _get_connection_params(self) -> Dict[str, Any]:
        """
        Get connection parameters based on client and connection IDs.
        
        If direct connection parameters are provided (host, port, etc.),
        those will be used. Otherwise, connection details will be loaded
        from settings/database based on client_id and connection_id.
        """
        # If direct parameters are provided, use those
        if self.host and self.database and self.username:
            return {
                "host": self.host,
                "port": self.port or 5432,
                "database": self.database,
                "user": self.username,
                "password": self.password,
                "ssl": "require" if self.ssl_enabled else None
            }
        
        # Otherwise, load connection details based on client_id and connection_id
        if not self.client_id:
            raise ValueError("Either direct connection parameters or client_id must be provided")
        
        # In a real implementation, you would look up connection details from a database
        # For demonstration, we'll use hardcoded values
        
        if self.client_id == "client-1":
            if self.connection_id == "conn-1" or not self.connection_id:
                return {
                    "host": "db.example.com",
                    "port": 5432,
                    "database": "client1_db",
                    "user": "client1_user",
                    "password": "client1_password",
                    "ssl": "require"
                }
        elif self.client_id == "client-2":
            if self.connection_id == "conn-2" or not self.connection_id:
                return {
                    "host": "db.example.com",
                    "port": 5432,
                    "database": "client2_db",
                    "user": "client2_user",
                    "password": "client2_password",
                    "ssl": "require"
                }
        
        # For development, use local database if no matches
        if settings.environment == "development":
            return {
                "host": "localhost",
                "port": 5432,
                "database": "supply_chain_dev",
                "user": "postgres",
                "password": "postgres",
                "ssl": None
            }
        
        raise ValueError(f"No connection found for client_id={self.client_id}, connection_id={self.connection_id}")
    
    async def _get_connection_pool(self) -> asyncpg.Pool:
        """Get a connection pool for the database."""
        if self.pool is None:
            conn_params = await self._get_connection_params()
            
            # Create a connection pool
            try:
                self.pool = await asyncpg.create_pool(
                    host=conn_params["host"],
                    port=conn_params["port"],
                    database=conn_params["database"],
                    user=conn_params["user"],
                    password=conn_params["password"],
                    ssl=conn_params["ssl"],
                    min_size=2,
                    max_size=10
                )
                
                logger.info(f"Created connection pool for {conn_params['host']}:{conn_params['port']}/{conn_params['database']}")
                
            except Exception as e:
                logger.error(f"Error creating connection pool: {str(e)}")
                raise
        
        return self.pool
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a read-only SQL query and return the results.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            schema: Optional schema to use
            
        Returns:
            Dictionary with columns and data
        """
        # Validate that this is a read-only query for safety
        if not self._is_read_only_query(query):
            raise ValueError("Only SELECT queries are allowed")
        
        # Convert params dict to list for asyncpg
        param_values = []
        if params:
            # Replace named parameters with positional parameters
            # e.g. :param1 -> $1, :param2 -> $2
            param_names = []
            for name, value in params.items():
                param_names.append(name)
                param_values.append(value)
            
            for i, name in enumerate(param_names):
                query = query.replace(f":{name}", f"${i+1}")
        
        # Set schema if provided
        schema_prefix = ""
        if schema:
            schema_prefix = f"SET search_path TO {schema}; "
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute query
            async with pool.acquire() as conn:
                if schema:
                    # Set schema for this connection
                    await conn.execute(f"SET search_path TO {schema}")
                
                # Execute the query
                rows = await conn.fetch(query, *param_values)
                
                # Convert to list of dictionaries
                result = []
                for row in rows:
                    row_dict = dict(row)
                    # Convert any non-serializable types
                    for key, value in row_dict.items():
                        if isinstance(value, (datetime, date, uuid.UUID)):
                            row_dict[key] = str(value)
                    result.append(row_dict)
                
                # Get column names from the first row
                columns = []
                if rows:
                    columns = [key for key in dict(rows[0]).keys()]
                
                return {
                    "columns": columns,
                    "data": result,
                    "row_count": len(result)
                }
        
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    async def get_tables(
        self,
        schema: Optional[str] = None
    ) -> List[str]:
        """
        Get a list of tables in the database.
        
        Args:
            schema: Optional schema to limit the tables
            
        Returns:
            List of table names
        """
        # Set schema condition
        schema_condition = "AND table_schema = $1" if schema else "AND table_schema NOT IN ('pg_catalog', 'information_schema')"
        params = [schema] if schema else []
        
        # Query to get tables
        query = f"""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_type = 'BASE TABLE' 
        {schema_condition}
        ORDER BY table_name
        """
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute query
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [row['table_name'] for row in rows]
        
        except Exception as e:
            logger.error(f"Error getting tables: {str(e)}")
            raise
    
    async def get_table_schema(
        self,
        table_name: str,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the schema for a specific table.
        
        Args:
            table_name: Table name
            schema: Optional schema name
            
        Returns:
            Dictionary with table schema information
        """
        # Set schema for the query
        schema_param = schema or 'public'
        
        # Query to get columns
        columns_query = """
        SELECT 
            column_name, 
            data_type, 
            is_nullable = 'YES' as is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale
        FROM 
            information_schema.columns
        WHERE 
            table_name = $1
            AND table_schema = $2
        ORDER BY 
            ordinal_position
        """
        
        # Query to get primary key
        pk_query = """
        SELECT 
            kcu.column_name
        FROM 
            information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
        WHERE 
            tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_name = $1
            AND tc.table_schema = $2
        ORDER BY 
            kcu.ordinal_position
        """
        
        # Query to get foreign keys
        fk_query = """
        SELECT 
            kcu.column_name,
            ccu.table_schema as foreign_table_schema,
            ccu.table_name as foreign_table_name,
            ccu.column_name as foreign_column_name
        FROM 
            information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
                AND tc.table_schema = ccu.constraint_schema
        WHERE 
            tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_name = $1
            AND tc.table_schema = $2
        """
        
        # Query to get indexes
        index_query = """
        SELECT
            i.relname as index_name,
            a.attname as column_name,
            ix.indisunique as is_unique
        FROM
            pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
            JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE
            t.relname = $1
            AND n.nspname = $2
            AND NOT ix.indisprimary
        ORDER BY
            i.relname, a.attnum
        """
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute queries
            async with pool.acquire() as conn:
                # Get columns
                columns_rows = await conn.fetch(columns_query, table_name, schema_param)
                
                # Get primary key
                pk_rows = await conn.fetch(pk_query, table_name, schema_param)
                
                # Get foreign keys
                fk_rows = await conn.fetch(fk_query, table_name, schema_param)
                
                # Get indexes
                index_rows = await conn.fetch(index_query, table_name, schema_param)
                
                # Process columns
                columns = []
                for row in columns_rows:
                    column = {
                        "name": row["column_name"],
                        "data_type": row["data_type"],
                        "nullable": row["is_nullable"],
                        "default": row["column_default"]
                    }
                    
                    # Add length for character types
                    if row["character_maximum_length"]:
                        column["max_length"] = row["character_maximum_length"]
                    
                    # Add precision and scale for numeric types
                    if row["numeric_precision"]:
                        column["precision"] = row["numeric_precision"]
                        if row["numeric_scale"]:
                            column["scale"] = row["numeric_scale"]
                    
                    columns.append(column)
                
                # Process primary key
                primary_key = [row["column_name"] for row in pk_rows]
                
                # Process foreign keys
                foreign_keys = []
                for row in fk_rows:
                    foreign_keys.append({
                        "column": row["column_name"],
                        "references": f"{row['foreign_table_schema']}.{row['foreign_table_name']}.{row['foreign_column_name']}"
                    })
                
                # Process indexes
                indexes = []
                current_index = None
                for row in index_rows:
                    if current_index is None or current_index["name"] != row["index_name"]:
                        if current_index:
                            indexes.append(current_index)
                        current_index = {
                            "name": row["index_name"],
                            "columns": [row["column_name"]],
                            "unique": row["is_unique"]
                        }
                    else:
                        current_index["columns"].append(row["column_name"])
                
                if current_index:
                    indexes.append(current_index)
                
                # Build table schema
                table_schema = {
                    "name": table_name,
                    "schema": schema_param,
                    "columns": columns,
                    "primary_key": primary_key,
                    "foreign_keys": foreign_keys,
                    "indexes": indexes
                }
                
                return table_schema
        
        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            raise
    
    async def get_table_data(
        self,
        table_name: str,
        schema: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get paginated data from a table.
        
        Args:
            table_name: Table name
            schema: Optional schema name
            page: Page number (1-based)
            page_size: Number of rows per page
            sort_by: Column to sort by
            sort_order: Sort order (asc or desc)
            filter_conditions: Optional filter conditions
            
        Returns:
            Dictionary with columns, data, and pagination info
        """
        # Validate parameters
        if page < 1:
            raise ValueError("Page must be at least 1")
        if page_size < 1:
            raise ValueError("Page size must be at least 1")
        if sort_order.lower() not in ["asc", "desc"]:
            raise ValueError("Sort order must be 'asc' or 'desc'")
        
        # Build the query
        schema_prefix = f"{schema}." if schema else ""
        query = f"SELECT * FROM {schema_prefix}\"{table_name}\""
        count_query = f"SELECT COUNT(*) FROM {schema_prefix}\"{table_name}\""
        
        # Add filter conditions
        params = []
        if filter_conditions:
            where_clauses = []
            param_index = 1
            
            for column, value in filter_conditions.items():
                # Handle different types of filters
                if isinstance(value, dict):
                    # Advanced filter with operators
                    for op, val in value.items():
                        if op == "eq":
                            where_clauses.append(f"\"{column}\" = ${param_index}")
                        elif op == "neq":
                            where_clauses.append(f"\"{column}\" != ${param_index}")
                        elif op == "gt":
                            where_clauses.append(f"\"{column}\" > ${param_index}")
                        elif op == "gte":
                            where_clauses.append(f"\"{column}\" >= ${param_index}")
                        elif op == "lt":
                            where_clauses.append(f"\"{column}\" < ${param_index}")
                        elif op == "lte":
                            where_clauses.append(f"\"{column}\" <= ${param_index}")
                        elif op == "like":
                            where_clauses.append(f"\"{column}\" LIKE ${param_index}")
                        elif op == "in":
                            if isinstance(val, list):
                                placeholders = [f"${param_index + i}" for i in range(len(val))]
                                where_clauses.append(f"\"{column}\" IN ({', '.join(placeholders)})")
                                params.extend(val)
                                param_index += len(val) - 1
                            else:
                                where_clauses.append(f"\"{column}\" = ${param_index}")
                                params.append(val)
                        else:
                            raise ValueError(f"Unsupported operator: {op}")
                        
                        params.append(val)
                        param_index += 1
                else:
                    # Simple equality filter
                    where_clauses.append(f"\"{column}\" = ${param_index}")
                    params.append(value)
                    param_index += 1
            
            if where_clauses:
                where_clause = " AND ".join(where_clauses)
                query += f" WHERE {where_clause}"
                count_query += f" WHERE {where_clause}"
        
        # Add sorting
        if sort_by:
            query += f" ORDER BY \"{sort_by}\" {sort_order}"
        
        # Add pagination
        offset = (page - 1) * page_size
        query += f" LIMIT {page_size} OFFSET {offset}"
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute queries
            async with pool.acquire() as conn:
                # Get data
                rows = await conn.fetch(query, *params)
                
                # Get total count
                count_result = await conn.fetchval(count_query, *params)
                total_rows = count_result if count_result is not None else 0
                
                # Convert to list of dictionaries
                result = []
                for row in rows:
                    row_dict = dict(row)
                    # Convert any non-serializable types
                    for key, value in row_dict.items():
                        if isinstance(value, (datetime, date, uuid.UUID)):
                            row_dict[key] = str(value)
                    result.append(row_dict)
                
                # Get column names from the first row
                columns = []
                if rows:
                    columns = [key for key in dict(rows[0]).keys()]
                
                return {
                    "columns": columns,
                    "data": result,
                    "total_rows": total_rows,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": (total_rows + page_size - 1) // page_size if page_size > 0 else 0
                }
        
        except Exception as e:
            logger.error(f"Error getting table data: {str(e)}")
            raise
    
    async def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Get connection pool
            pool = await self._get_connection_pool()
            
            # Test connection
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Closed connection pool")
    
    @staticmethod
    def _is_read_only_query(query: str) -> bool:
        """
        Check if a query is read-only (SELECT only).
        
        Args:
            query: SQL query to check
            
        Returns:
            True if the query is read-only, False otherwise
        """
        # Remove comments and normalize whitespace
        clean_query = re.sub(r'--.*?(\n|$)', ' ', query)
        clean_query = re.sub(r'/\*.*?\*/', ' ', clean_query, flags=re.DOTALL)
        clean_query = ' '.join(clean_query.split()).strip().lower()
        
        # Check if the query is a SELECT query
        if clean_query.startswith('select '):
            # Check for any data modification statements
            data_modification = [
                ' insert ', ' update ', ' delete ', ' truncate ',
                ' create ', ' alter ', ' drop ', ' grant ', ' revoke ',
                ';insert ', ';update ', ';delete ', ';truncate ',
                ';create ', ';alter ', ';drop ', ';grant ', ';revoke '
            ]
            return not any(statement in f" {clean_query} " for statement in data_modification)
        
        return False