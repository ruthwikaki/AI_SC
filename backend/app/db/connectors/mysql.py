import aiomysql
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

class MySQLConnector:
    """
    MySQL database connector.
    
    This connector provides methods to interact with MySQL databases,
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
        ssl_enabled: bool = False
    ):
        """
        Initialize the MySQL connector.
        
        Args:
            client_id: Optional client ID to load connection from settings
            connection_id: Optional connection ID to load specific connection
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
            ssl_enabled: Whether to use SSL for the connection
        """
        self.client_id = client_id
        self.connection_id = connection_id
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
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
                "port": self.port or 3306,
                "db": self.database,
                "user": self.username,
                "password": self.password,
                "ssl": self.ssl_enabled
            }
        
        # Otherwise, load connection details based on client_id and connection_id
        if not self.client_id:
            raise ValueError("Either direct connection parameters or client_id must be provided")
        
        # In a real implementation, you would look up connection details from a database
        # For demonstration, we'll use hardcoded values
        
        if self.client_id == "client-1":
            if self.connection_id == "mysql-1" or not self.connection_id:
                return {
                    "host": "mysql.example.com",
                    "port": 3306,
                    "db": "client1_db",
                    "user": "client1_user",
                    "password": "client1_password",
                    "ssl": True
                }
        elif self.client_id == "client-2":
            if self.connection_id == "mysql-2" or not self.connection_id:
                return {
                    "host": "mysql.example.com",
                    "port": 3306,
                    "db": "client2_db",
                    "user": "client2_user",
                    "password": "client2_password",
                    "ssl": True
                }
        
        # For development, use local database if no matches
        if settings.environment == "development":
            return {
                "host": "localhost",
                "port": 3306,
                "db": "supply_chain_dev",
                "user": "root",
                "password": "password",
                "ssl": False
            }
        
        raise ValueError(f"No connection found for client_id={self.client_id}, connection_id={self.connection_id}")
    
    async def _get_connection_pool(self) -> aiomysql.Pool:
        """Get a connection pool for the database."""
        if self.pool is None:
            conn_params = await self._get_connection_params()
            
            # Create a connection pool
            try:
                self.pool = await aiomysql.create_pool(
                    host=conn_params["host"],
                    port=conn_params["port"],
                    db=conn_params["db"],
                    user=conn_params["user"],
                    password=conn_params["password"],
                    ssl=conn_params["ssl"],
                    minsize=2,
                    maxsize=10,
                    autocommit=True
                )
                
                logger.info(f"Created MySQL connection pool for {conn_params['host']}:{conn_params['port']}/{conn_params['db']}")
                
            except Exception as e:
                logger.error(f"Error creating MySQL connection pool: {str(e)}")
                raise
        
        return self.pool
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a read-only SQL query and return the results.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Dictionary with columns and data
        """
        # Validate that this is a read-only query for safety
        if not self._is_read_only_query(query):
            raise ValueError("Only SELECT queries are allowed")
        
        # Convert params dict to list for aiomysql
        param_values = []
        if params:
            # Replace named parameters with placeholders
            # e.g. :param1 -> %s, :param2 -> %s
            for name, value in params.items():
                query = query.replace(f":{name}", "%s")
                param_values.append(value)
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute query
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    # Execute the query
                    await cursor.execute(query, param_values)
                    
                    # Fetch all rows
                    rows = await cursor.fetchall()
                    
                    # Convert non-serializable types
                    result = []
                    for row in rows:
                        row_dict = dict(row)
                        # Convert any non-serializable types
                        for key, value in row_dict.items():
                            if isinstance(value, (datetime, date)):
                                row_dict[key] = str(value)
                            elif isinstance(value, bytes):
                                row_dict[key] = value.hex()
                        result.append(row_dict)
                    
                    # Get column names from cursor description
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    
                    return {
                        "columns": columns,
                        "data": result,
                        "row_count": len(result)
                    }
        
        except Exception as e:
            logger.error(f"Error executing MySQL query: {str(e)}")
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
        # Use provided schema or get it from connection params
        if not schema:
            conn_params = await self._get_connection_params()
            schema = conn_params.get("db")
        
        # Query to get tables
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = %s
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute query
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(query, (schema,))
                    rows = await cursor.fetchall()
                    return [row[0] for row in rows]
        
        except Exception as e:
            logger.error(f"Error getting MySQL tables: {str(e)}")
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
        # Use provided schema or get it from connection params
        if not schema:
            conn_params = await self._get_connection_params()
            schema = conn_params.get("db")
        
        # Query to get columns
        columns_query = """
        SELECT 
            column_name, 
            data_type, 
            is_nullable, 
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            column_key,
            extra
        FROM 
            information_schema.columns
        WHERE 
            table_name = %s
            AND table_schema = %s
        ORDER BY 
            ordinal_position
        """
        
        # Query to get primary key
        pk_query = """
        SELECT 
            column_name
        FROM 
            information_schema.key_column_usage
        WHERE 
            table_name = %s
            AND table_schema = %s
            AND constraint_name = 'PRIMARY'
        ORDER BY 
            ordinal_position
        """
        
        # Query to get foreign keys
        fk_query = """
        SELECT 
            k.column_name,
            k.referenced_table_schema,
            k.referenced_table_name,
            k.referenced_column_name
        FROM 
            information_schema.key_column_usage k
            JOIN information_schema.table_constraints t
                ON k.constraint_name = t.constraint_name
                AND k.table_schema = t.table_schema
                AND k.table_name = t.table_name
        WHERE 
            t.constraint_type = 'FOREIGN KEY'
            AND k.table_name = %s
            AND k.table_schema = %s
            AND k.referenced_table_name IS NOT NULL
        """
        
        # Query to get indexes
        index_query = """
        SELECT 
            index_name,
            column_name,
            non_unique
        FROM 
            information_schema.statistics
        WHERE 
            table_name = %s
            AND table_schema = %s
            AND index_name != 'PRIMARY'
        ORDER BY 
            index_name, 
            seq_in_index
        """
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute queries
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    # Get columns
                    await cursor.execute(columns_query, (table_name, schema))
                    columns_rows = await cursor.fetchall()
                    
                    # Get primary key
                    await cursor.execute(pk_query, (table_name, schema))
                    pk_rows = await cursor.fetchall()
                    
                    # Get foreign keys
                    await cursor.execute(fk_query, (table_name, schema))
                    fk_rows = await cursor.fetchall()
                    
                    # Get indexes
                    await cursor.execute(index_query, (table_name, schema))
                    index_rows = await cursor.fetchall()
                    
                    # Process columns
                    columns = []
                    for row in columns_rows:
                        column = {
                            "name": row["column_name"],
                            "data_type": row["data_type"],
                            "nullable": row["is_nullable"] == "YES",
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
                        
                        # Check if column is auto-increment
                        if row["extra"] and "auto_increment" in row["extra"].lower():
                            column["auto_increment"] = True
                        
                        columns.append(column)
                    
                    # Process primary key
                    primary_key = [row["column_name"] for row in pk_rows]
                    
                    # Process foreign keys
                    foreign_keys = []
                    for row in fk_rows:
                        foreign_keys.append({
                            "column": row["column_name"],
                            "references": f"{row['referenced_table_schema']}.{row['referenced_table_name']}.{row['referenced_column_name']}"
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
                                "unique": not row["non_unique"]
                            }
                        else:
                            current_index["columns"].append(row["column_name"])
                    
                    if current_index:
                        indexes.append(current_index)
                    
                    # Build table schema
                    table_schema = {
                        "name": table_name,
                        "schema": schema,
                        "columns": columns,
                        "primary_key": primary_key,
                        "foreign_keys": foreign_keys,
                        "indexes": indexes
                    }
                    
                    return table_schema
        
        except Exception as e:
            logger.error(f"Error getting MySQL table schema: {str(e)}")
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
        
        # Use provided schema or get it from connection params
        if not schema:
            conn_params = await self._get_connection_params()
            schema = conn_params.get("db")
        
        # Build the query
        query = f"SELECT * FROM `{schema}`.`{table_name}`"
        count_query = f"SELECT COUNT(*) FROM `{schema}`.`{table_name}`"
        
        # Add filter conditions
        params = []
        if filter_conditions:
            where_clauses = []
            
            for column, value in filter_conditions.items():
                # Handle different types of filters
                if isinstance(value, dict):
                    # Advanced filter with operators
                    for op, val in value.items():
                        if op == "eq":
                            where_clauses.append(f"`{column}` = %s")
                            params.append(val)
                        elif op == "neq":
                            where_clauses.append(f"`{column}` != %s")
                            params.append(val)
                        elif op == "gt":
                            where_clauses.append(f"`{column}` > %s")
                            params.append(val)
                        elif op == "gte":
                            where_clauses.append(f"`{column}` >= %s")
                            params.append(val)
                        elif op == "lt":
                            where_clauses.append(f"`{column}` < %s")
                            params.append(val)
                        elif op == "lte":
                            where_clauses.append(f"`{column}` <= %s")
                            params.append(val)
                        elif op == "like":
                            where_clauses.append(f"`{column}` LIKE %s")
                            params.append(val)
                        elif op == "in":
                            if isinstance(val, list):
                                placeholders = ", ".join(["%s"] * len(val))
                                where_clauses.append(f"`{column}` IN ({placeholders})")
                                params.extend(val)
                            else:
                                where_clauses.append(f"`{column}` = %s")
                                params.append(val)
                        else:
                            raise ValueError(f"Unsupported operator: {op}")
                else:
                    # Simple equality filter
                    where_clauses.append(f"`{column}` = %s")
                    params.append(value)
            
            if where_clauses:
                where_clause = " AND ".join(where_clauses)
                query += f" WHERE {where_clause}"
                count_query += f" WHERE {where_clause}"
        
        # Add sorting
        if sort_by:
            query += f" ORDER BY `{sort_by}` {sort_order}"
        
        # Add pagination
        offset = (page - 1) * page_size
        query += f" LIMIT {page_size} OFFSET {offset}"
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute queries
            async with pool.acquire() as conn:
                # Get total count
                async with conn.cursor() as count_cursor:
                    await count_cursor.execute(count_query, params)
                    count_result = await count_cursor.fetchone()
                    total_rows = count_result[0] if count_result else 0
                
                # Get data
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(query, params)
                    rows = await cursor.fetchall()
                    
                    # Get column names
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    
                    # Convert to list of dictionaries
                    result = []
                    for row in rows:
                        row_dict = dict(row)
                        # Convert any non-serializable types
                        for key, value in row_dict.items():
                            if isinstance(value, (datetime, date)):
                                row_dict[key] = str(value)
                            elif isinstance(value, bytes):
                                row_dict[key] = value.hex()
                        result.append(row_dict)
                    
                    return {
                        "columns": columns,
                        "data": result,
                        "total_rows": total_rows,
                        "page": page,
                        "page_size": page_size,
                        "total_pages": (total_rows + page_size - 1) // page_size if page_size > 0 else 0
                    }
        
        except Exception as e:
            logger.error(f"Error getting MySQL table data: {str(e)}")
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
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    result = await cursor.fetchone()
                    return result[0] == 1
        
        except Exception as e:
            logger.error(f"MySQL connection test failed: {str(e)}")
            return False
    
    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            self.pool = None
            logger.info("Closed MySQL connection pool")
    
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
        clean_query = re.sub(r'#.*?(\n|$)', ' ', clean_query)  # MySQL-specific comments
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