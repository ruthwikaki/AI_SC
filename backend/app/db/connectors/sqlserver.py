import aioodbc
import pyodbc
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

class SQLServerConnector:
    """
    SQL Server database connector.
    
    This connector provides methods to interact with SQL Server databases,
    with support for dynamic client connections and schema discovery.
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        server: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        driver: Optional[str] = None,
        trusted_connection: bool = False,
        encrypt: bool = True
    ):
        """
        Initialize the SQL Server connector.
        
        Args:
            client_id: Optional client ID to load connection from settings
            connection_id: Optional connection ID to load specific connection
            server: Server name or IP
            port: Server port
            database: Database name
            username: Database username
            password: Database password
            driver: ODBC driver name
            trusted_connection: Whether to use Windows authentication
            encrypt: Whether to encrypt the connection
        """
        self.client_id = client_id
        self.connection_id = connection_id
        self.server = server
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.driver = driver
        self.trusted_connection = trusted_connection
        self.encrypt = encrypt
        self.pool = None
    
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
        
        # Default schema if not provided
        schema = schema or "dbo"
        
        # Calculate offset for pagination
        offset = (page - 1) * page_size
        
        # Build the query with SQL Server-specific pagination (for SQL Server 2012+)
        base_query = f"SELECT * FROM [{schema}].[{table_name}]"
        count_query = f"SELECT COUNT(*) FROM [{schema}].[{table_name}]"
        
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
                            where_clauses.append(f"[{column}] = ?")
                            params.append(val)
                        elif op == "neq":
                            where_clauses.append(f"[{column}] != ?")
                            params.append(val)
                        elif op == "gt":
                            where_clauses.append(f"[{column}] > ?")
                            params.append(val)
                        elif op == "gte":
                            where_clauses.append(f"[{column}] >= ?")
                            params.append(val)
                        elif op == "lt":
                            where_clauses.append(f"[{column}] < ?")
                            params.append(val)
                        elif op == "lte":
                            where_clauses.append(f"[{column}] <= ?")
                            params.append(val)
                        elif op == "like":
                            where_clauses.append(f"[{column}] LIKE ?")
                            params.append(val)
                        elif op == "in":
                            if isinstance(val, list):
                                placeholders = ", ".join(["?"] * len(val))
                                where_clauses.append(f"[{column}] IN ({placeholders})")
                                params.extend(val)
                            else:
                                where_clauses.append(f"[{column}] = ?")
                                params.append(val)
                        else:
                            raise ValueError(f"Unsupported operator: {op}")
                else:
                    # Simple equality filter
                    where_clauses.append(f"[{column}] = ?")
                    params.append(value)
            
            if where_clauses:
                where_clause = " AND ".join(where_clauses)
                base_query += f" WHERE {where_clause}"
                count_query += f" WHERE {where_clause}"
        
        # Add ORDER BY for pagination and sorting
        order_by = f"ORDER BY [{sort_by or 'id'}] {sort_order}" if sort_by else "ORDER BY (SELECT NULL)"
        
        # Build the final query with OFFSET/FETCH for pagination (SQL Server 2012+)
        query = f"""
        {base_query}
        {order_by}
        OFFSET ? ROWS
        FETCH NEXT ? ROWS ONLY
        """
        
        # Add pagination parameters
        params.extend([offset, page_size])
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute queries
            async with pool.acquire() as conn:
                # Get total count
                async with conn.cursor() as count_cursor:
                    await count_cursor.execute(count_query, params[:-2])  # Exclude pagination params
                    count_result = await count_cursor.fetchone()
                    total_rows = count_result[0] if count_result else 0
                
                # Get data
                async with conn.cursor() as cursor:
                    await cursor.execute(query, params)
                    
                    # Get column names from cursor description
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    
                    # Fetch all rows
                    rows = await cursor.fetchall()
                    
                    # Convert to list of dictionaries
                    result = []
                    for row in rows:
                        row_dict = {}
                        for i, col in enumerate(columns):
                            value = row[i]
                            # Convert any non-serializable types
                            if isinstance(value, (datetime, date)):
                                value = str(value)
                            elif isinstance(value, uuid.UUID):
                                value = str(value)
                            elif isinstance(value, bytes):
                                value = value.hex()
                            row_dict[col] = value
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
            logger.error(f"Error getting SQL Server table data: {str(e)}")
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
            logger.error(f"SQL Server connection test failed: {str(e)}")
            return False
    
    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            self.pool = None
            logger.info("Closed SQL Server connection pool")
    
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
                ' exec ', ' execute ', ' sp_', ' xp_',  # SQL Server-specific
                ';insert ', ';update ', ';delete ', ';truncate ',
                ';create ', ';alter ', ';drop ', ';grant ', ';revoke ',
                ';exec ', ';execute ', ';sp_', ';xp_'
            ]
            return not any(statement in f" {clean_query} " for statement in data_modification)
        
        return False _get_connection_string(self) -> str:
        """
        Get connection string based on client and connection IDs.
        
        If direct connection parameters are provided (server, port, etc.),
        those will be used. Otherwise, connection details will be loaded
        from settings/database based on client_id and connection_id.
        """
        # If direct parameters are provided, use those
        if self.server and self.database:
            # Determine port string
            port_str = f",{self.port}" if self.port else ""
            
            # Determine driver
            driver = self.driver or "ODBC Driver 17 for SQL Server"
            
            # Build connection string
            if self.trusted_connection:
                # Windows authentication
                return (
                    f"DRIVER={{{driver}}};"
                    f"SERVER={self.server}{port_str};"
                    f"DATABASE={self.database};"
                    f"Trusted_Connection=yes;"
                    f"Encrypt={'yes' if self.encrypt else 'no'}"
                )
            else:
                # SQL Server authentication
                return (
                    f"DRIVER={{{driver}}};"
                    f"SERVER={self.server}{port_str};"
                    f"DATABASE={self.database};"
                    f"UID={self.username};"
                    f"PWD={self.password};"
                    f"Encrypt={'yes' if self.encrypt else 'no'}"
                )
        
        # Otherwise, load connection details based on client_id and connection_id
        if not self.client_id:
            raise ValueError("Either direct connection parameters or client_id must be provided")
        
        # In a real implementation, you would look up connection details from a database
        # For demonstration, we'll use hardcoded values
        
        if self.client_id == "client-1":
            if self.connection_id == "sqlserver-1" or not self.connection_id:
                return (
                    "DRIVER={ODBC Driver 17 for SQL Server};"
                    "SERVER=sqlserver.example.com,1433;"
                    "DATABASE=client1_db;"
                    "UID=client1_user;"
                    "PWD=client1_password;"
                    "Encrypt=yes"
                )
        elif self.client_id == "client-2":
            if self.connection_id == "sqlserver-2" or not self.connection_id:
                return (
                    "DRIVER={ODBC Driver 17 for SQL Server};"
                    "SERVER=sqlserver.example.com,1433;"
                    "DATABASE=client2_db;"
                    "UID=client2_user;"
                    "PWD=client2_password;"
                    "Encrypt=yes"
                )
        
        # For development, use local database if no matches
        if settings.environment == "development":
            return (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                "SERVER=localhost,1433;"
                "DATABASE=supply_chain_dev;"
                "UID=sa;"
                "PWD=Password123!;"
                "Encrypt=no"
            )
        
        raise ValueError(f"No connection found for client_id={self.client_id}, connection_id={self.connection_id}")
    
    async def _get_connection_pool(self) -> aioodbc.Pool:
        """Get a connection pool for the database."""
        if self.pool is None:
            connection_string = await self._get_connection_string()
            
            # Create a connection pool
            try:
                self.pool = await aioodbc.create_pool(
                    dsn=connection_string,
                    minsize=2,
                    maxsize=10,
                    autocommit=True
                )
                
                logger.info(f"Created SQL Server connection pool for {self.database or 'unknown'}")
                
            except Exception as e:
                logger.error(f"Error creating SQL Server connection pool: {str(e)}")
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
        
        # Convert params dict to list for aioodbc
        param_values = []
        if params:
            # Replace named parameters with question marks
            # e.g. :param1 -> ?, :param2 -> ?
            for name, value in params.items():
                query = query.replace(f":{name}", "?")
                param_values.append(value)
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute query
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # Execute the query
                    await cursor.execute(query, param_values)
                    
                    # Fetch all rows
                    rows = await cursor.fetchall()
                    
                    # Get column names from cursor description
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    
                    # Convert to list of dictionaries
                    result = []
                    for row in rows:
                        row_dict = {}
                        for i, col in enumerate(columns):
                            value = row[i]
                            # Convert any non-serializable types
                            if isinstance(value, (datetime, date)):
                                value = str(value)
                            elif isinstance(value, uuid.UUID):
                                value = str(value)
                            elif isinstance(value, bytes):
                                value = value.hex()
                            row_dict[col] = value
                        result.append(row_dict)
                    
                    return {
                        "columns": columns,
                        "data": result,
                        "row_count": len(result)
                    }
        
        except Exception as e:
            logger.error(f"Error executing SQL Server query: {str(e)}")
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
        schema_condition = "AND s.name = ?" if schema else ""
        params = [schema] if schema else []
        
        # Query to get tables
        query = f"""
        SELECT t.name
        FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE 1=1 
        {schema_condition}
        ORDER BY t.name
        """
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute query
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(query, params)
                    rows = await cursor.fetchall()
                    return [row[0] for row in rows]
        
        except Exception as e:
            logger.error(f"Error getting SQL Server tables: {str(e)}")
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
        # Default schema if not provided
        schema = schema or "dbo"
        
        # Query to get columns
        columns_query = """
        SELECT 
            c.name AS column_name,
            t.name AS data_type,
            c.is_nullable,
            c.column_default,
            c.max_length,
            c.precision,
            c.scale,
            c.is_identity,
            c.is_computed
        FROM 
            sys.columns c
        JOIN 
            sys.types t ON c.user_type_id = t.user_type_id
        JOIN 
            sys.tables tbl ON c.object_id = tbl.object_id
        JOIN 
            sys.schemas s ON tbl.schema_id = s.schema_id
        WHERE 
            tbl.name = ?
            AND s.name = ?
        ORDER BY 
            c.column_id
        """
        
        # Query to get primary key
        pk_query = """
        SELECT 
            c.name AS column_name
        FROM 
            sys.indexes i
        JOIN 
            sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
        JOIN 
            sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
        JOIN 
            sys.tables t ON i.object_id = t.object_id
        JOIN 
            sys.schemas s ON t.schema_id = s.schema_id
        WHERE 
            i.is_primary_key = 1
            AND t.name = ?
            AND s.name = ?
        ORDER BY 
            ic.key_ordinal
        """
        
        # Query to get foreign keys
        fk_query = """
        SELECT 
            c.name AS column_name,
            rs.name AS referenced_schema,
            rt.name AS referenced_table,
            rc.name AS referenced_column
        FROM 
            sys.foreign_key_columns fkc
        JOIN 
            sys.tables t ON fkc.parent_object_id = t.object_id
        JOIN 
            sys.schemas s ON t.schema_id = s.schema_id
        JOIN 
            sys.columns c ON fkc.parent_object_id = c.object_id AND fkc.parent_column_id = c.column_id
        JOIN 
            sys.tables rt ON fkc.referenced_object_id = rt.object_id
        JOIN 
            sys.schemas rs ON rt.schema_id = rs.schema_id
        JOIN 
            sys.columns rc ON fkc.referenced_object_id = rc.object_id AND fkc.referenced_column_id = rc.column_id
        WHERE 
            t.name = ?
            AND s.name = ?
        """
        
        # Query to get indexes
        index_query = """
        SELECT 
            i.name AS index_name,
            c.name AS column_name,
            i.is_unique
        FROM 
            sys.indexes i
        JOIN 
            sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
        JOIN 
            sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
        JOIN 
            sys.tables t ON i.object_id = t.object_id
        JOIN 
            sys.schemas s ON t.schema_id = s.schema_id
        WHERE 
            i.is_primary_key = 0
            AND t.name = ?
            AND s.name = ?
        ORDER BY 
            i.name, 
            ic.key_ordinal
        """
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute queries
            async with pool.acquire() as conn:
                # Get columns
                columns = []
                async with conn.cursor() as cursor:
                    await cursor.execute(columns_query, (table_name, schema))
                    columns_rows = await cursor.fetchall()
                    
                    for row in columns_rows:
                        column = {
                            "name": row[0],  # column_name
                            "data_type": row[1],  # data_type
                            "nullable": bool(row[2]),  # is_nullable
                            "default": row[3]  # column_default
                        }
                        
                        # Add length for character types
                        if row[4] and row[1] in ('char', 'varchar', 'nchar', 'nvarchar', 'binary', 'varbinary'):
                            column["max_length"] = row[4]  # max_length
                        
                        # Add precision and scale for numeric types
                        if row[5] and row[1] in ('decimal', 'numeric'):
                            column["precision"] = row[5]  # precision
                            if row[6]:
                                column["scale"] = row[6]  # scale
                        
                        # Add identity (auto-increment) flag
                        if row[7]:
                            column["auto_increment"] = True
                        
                        # Add computed column flag
                        if row[8]:
                            column["computed"] = True
                        
                        columns.append(column)
                
                # Get primary key
                primary_key = []
                async with conn.cursor() as cursor:
                    await cursor.execute(pk_query, (table_name, schema))
                    pk_rows = await cursor.fetchall()
                    primary_key = [row[0] for row in pk_rows]
                
                # Get foreign keys
                foreign_keys = []
                async with conn.cursor() as cursor:
                    await cursor.execute(fk_query, (table_name, schema))
                    fk_rows = await cursor.fetchall()
                    
                    for row in fk_rows:
                        foreign_keys.append({
                            "column": row[0],  # column_name
                            "references": f"{row[1]}.{row[2]}.{row[3]}"  # referenced_schema.referenced_table.referenced_column
                        })
                
                # Get indexes
                indexes = []
                current_index = None
                async with conn.cursor() as cursor:
                    await cursor.execute(index_query, (table_name, schema))
                    index_rows = await cursor.fetchall()
                    
                    for row in index_rows:
                        if current_index is None or current_index["name"] != row[0]:
                            if current_index:
                                indexes.append(current_index)
                            current_index = {
                                "name": row[0],  # index_name
                                "columns": [row[1]],  # column_name
                                "unique": bool(row[2])  # is_unique
                            }
                        else:
                            current_index["columns"].append(row[1])
                    
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
            logger.error(f"Error getting SQL Server table schema: {str(e)}")
            raise
    
    async def