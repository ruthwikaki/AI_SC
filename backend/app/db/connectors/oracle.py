import asyncio
import oracledb
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

# Enable thick mode for Oracle client
try:
    oracledb.init_oracle_client()
except Exception as e:
    logger.warning(f"Failed to initialize Oracle client in thick mode: {str(e)}")
    logger.info("Falling back to thin mode (limited functionality)")

class OracleConnector:
    """
    Oracle database connector.
    
    This connector provides methods to interact with Oracle databases,
    with support for dynamic client connections and schema discovery.
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        service_name: Optional[str] = None,
        sid: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        wallet_location: Optional[str] = None,
        wallet_password: Optional[str] = None
    ):
        """
        Initialize the Oracle connector.
        
        Args:
            client_id: Optional client ID to load connection from settings
            connection_id: Optional connection ID to load specific connection
            host: Database host
            port: Database port
            service_name: Oracle service name
            sid: Oracle SID
            username: Database username
            password: Database password
            wallet_location: Oracle wallet location for TLS
            wallet_password: Oracle wallet password
        """
        self.client_id = client_id
        self.connection_id = connection_id
        self.host = host
        self.port = port
        self.service_name = service_name
        self.sid = sid
        self.username = username
        self.password = password
        self.wallet_location = wallet_location
        self.wallet_password = wallet_password
        self.pool = None
        self._connection_params = None
    
    async def _get_connection_params(self) -> Dict[str, Any]:
        """
        Get connection parameters based on client and connection IDs.
        
        If direct connection parameters are provided (host, port, etc.),
        those will be used. Otherwise, connection details will be loaded
        from settings/database based on client_id and connection_id.
        """
        # Use cached parameters if available
        if self._connection_params:
            return self._connection_params
            
        # If direct parameters are provided, use those
        if self.host and (self.service_name or self.sid) and self.username and self.password:
            dsn = None
            if self.service_name:
                dsn = f"{self.host}:{self.port or 1521}/{self.service_name}"
            else:
                dsn = f"{self.host}:{self.port or 1521}:{self.sid}"
                
            params = {
                "dsn": dsn,
                "user": self.username,
                "password": self.password
            }
            
            # Add wallet configuration if provided
            if self.wallet_location:
                params["wallet_location"] = self.wallet_location
                if self.wallet_password:
                    params["wallet_password"] = self.wallet_password
            
            self._connection_params = params
            return params
        
        # Otherwise, load connection details based on client_id and connection_id
        if not self.client_id:
            raise ValueError("Either direct connection parameters or client_id must be provided")
        
        # In a real implementation, you would look up connection details from a database
        # For demonstration, we'll use hardcoded values
        
        if self.client_id == "client-1":
            if self.connection_id == "oracle-1" or not self.connection_id:
                params = {
                    "dsn": "oracle.example.com:1521/XEPDB1",
                    "user": "client1_user",
                    "password": "client1_password"
                }
                self._connection_params = params
                return params
        elif self.client_id == "client-2":
            if self.connection_id == "oracle-2" or not self.connection_id:
                params = {
                    "dsn": "oracle.example.com:1521/XEPDB1",
                    "user": "client2_user",
                    "password": "client2_password"
                }
                self._connection_params = params
                return params
        
        # For development, use local database if no matches
        if settings.environment == "development":
            params = {
                "dsn": "localhost:1521/XEPDB1",
                "user": "system",
                "password": "oracle"
            }
            self._connection_params = params
            return params
        
        raise ValueError(f"No connection found for client_id={self.client_id}, connection_id={self.connection_id}")
    
    async def _get_connection_pool(self) -> oracledb.AsyncConnectionPool:
        """Get a connection pool for the database."""
        if self.pool is None:
            conn_params = await self._get_connection_params()
            
            # Create a connection pool
            try:
                # Oracle's async pool needs to be created in a thread
                self.pool = await asyncio.to_thread(
                    oracledb.create_pool,
                    **conn_params,
                    min=2,
                    max=10,
                    increment=1,
                    getmode=oracledb.POOL_GETMODE_WAIT
                )
                
                logger.info(f"Created Oracle connection pool for {conn_params['dsn']}")
                
            except Exception as e:
                logger.error(f"Error creating Oracle connection pool: {str(e)}")
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
        
        # Convert params dict to positional or named parameters for oracledb
        bind_params = {}
        if params:
            # Oracle allows named parameters with :name syntax
            # Just make sure the keys in the params dict match the parameter names in the query
            bind_params = params
        
        # If schema is provided, set it in the query
        if schema:
            # In Oracle, we need to prepend the schema to table names
            # This is a simplified version - in reality, we'd need to parse the query
            # and prepend the schema to each table name
            query = f"ALTER SESSION SET CURRENT_SCHEMA = {schema};\n{query}"
        
        # Get connection pool
        pool = await self._get_connection_pool()
        
        try:
            # Execute query
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # If schema is provided, set it for this session
                    if schema:
                        await cursor.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema}")
                    
                    # Execute the query
                    await cursor.execute(query, bind_params)
                    
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
                            elif isinstance(value, oracledb.LOB):
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
            logger.error(f"Error executing Oracle query: {str(e)}")
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
        # If schema not provided, use the current user's schema
        if not schema:
            # Get connection parameters to extract the username
            conn_params = await self._get_connection_params()
            schema = conn_params["user"].upper()
        else:
            # Oracle schema names are typically uppercase
            schema = schema.upper()
        
        # Query to get tables
        query = """
        SELECT table_name 
        FROM all_tables 
        WHERE owner = :schema
        ORDER BY table_name
        """
        
        try:
            # Execute the query
            result = await self.execute_query(query, {"schema": schema})
            
            # Extract table names
            return [row["TABLE_NAME"] for row in result["data"]]
        
        except Exception as e:
            logger.error(f"Error getting Oracle tables: {str(e)}")
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
        # If schema not provided, use the current user's schema
        if not schema:
            # Get connection parameters to extract the username
            conn_params = await self._get_connection_params()
            schema = conn_params["user"].upper()
        else:
            # Oracle schema names are typically uppercase
            schema = schema.upper()
        
        # Oracle table names are typically uppercase
        table_name = table_name.upper()
        
        # Query to get columns
        columns_query = """
        SELECT 
            column_name,
            data_type,
            CASE WHEN nullable = 'Y' THEN 1 ELSE 0 END as is_nullable,
            data_default as column_default,
            data_length,
            data_precision,
            data_scale,
            identity_column
        FROM 
            all_tab_columns
        WHERE 
            table_name = :table_name
            AND owner = :schema
        ORDER BY 
            column_id
        """
        
        # Query to get primary key
        pk_query = """
        SELECT 
            cols.column_name
        FROM 
            all_constraints cons
            JOIN all_cons_columns cols
                ON cons.constraint_name = cols.constraint_name
                AND cons.owner = cols.owner
        WHERE 
            cons.constraint_type = 'P'
            AND cons.table_name = :table_name
            AND cons.owner = :schema
        ORDER BY 
            cols.position
        """
        
        # Query to get foreign keys
        fk_query = """
        SELECT 
            cols.column_name,
            r_cons.owner as referenced_schema,
            r_cons.table_name as referenced_table,
            r_cols.column_name as referenced_column
        FROM 
            all_constraints cons
            JOIN all_cons_columns cols
                ON cons.constraint_name = cols.constraint_name
                AND cons.owner = cols.owner
            JOIN all_constraints r_cons
                ON cons.r_constraint_name = r_cons.constraint_name
                AND cons.r_owner = r_cons.owner
            JOIN all_cons_columns r_cols
                ON r_cons.constraint_name = r_cols.constraint_name
                AND r_cons.owner = r_cols.owner
                AND cols.position = r_cols.position
        WHERE 
            cons.constraint_type = 'R'
            AND cons.table_name = :table_name
            AND cons.owner = :schema
        """
        
        # Query to get indexes
        index_query = """
        SELECT 
            idx.index_name,
            col.column_name,
            CASE WHEN idx.uniqueness = 'UNIQUE' THEN 1 ELSE 0 END as is_unique
        FROM 
            all_indexes idx
            JOIN all_ind_columns col
                ON idx.index_name = col.index_name
                AND idx.owner = col.index_owner
        WHERE 
            idx.table_name = :table_name
            AND idx.owner = :schema
            AND idx.index_type != 'LOB'
        ORDER BY 
            idx.index_name, 
            col.column_position
        """
        
        try:
            # Execute queries
            columns_result = await self.execute_query(
                columns_query, 
                {"table_name": table_name, "schema": schema}
            )
            
            pk_result = await self.execute_query(
                pk_query, 
                {"table_name": table_name, "schema": schema}
            )
            
            fk_result = await self.execute_query(
                fk_query, 
                {"table_name": table_name, "schema": schema}
            )
            
            index_result = await self.execute_query(
                index_query, 
                {"table_name": table_name, "schema": schema}
            )
            
            # Process columns
            columns = []
            for row in columns_result["data"]:
                column = {
                    "name": row["COLUMN_NAME"],
                    "data_type": row["DATA_TYPE"],
                    "nullable": bool(row["IS_NULLABLE"]),
                    "default": row["COLUMN_DEFAULT"]
                }
                
                # Add length for character types
                if row["DATA_LENGTH"] and row["DATA_TYPE"] in ('CHAR', 'VARCHAR', 'VARCHAR2', 'NCHAR', 'NVARCHAR2'):
                    column["max_length"] = row["DATA_LENGTH"]
                
                # Add precision and scale for numeric types
                if row["DATA_PRECISION"] and row["DATA_TYPE"] in ('NUMBER', 'FLOAT', 'DECIMAL'):
                    column["precision"] = row["DATA_PRECISION"]
                    if row["DATA_SCALE"] is not None:
                        column["scale"] = row["DATA_SCALE"]
                
                # Check if column is identity
                if row["IDENTITY_COLUMN"] == 'YES':
                    column["auto_increment"] = True
                
                columns.append(column)
            
            # Process primary key
            primary_key = [row["COLUMN_NAME"] for row in pk_result["data"]]
            
            # Process foreign keys
            foreign_keys = []
            for row in fk_result["data"]:
                foreign_keys.append({
                    "column": row["COLUMN_NAME"],
                    "references": f"{row['REFERENCED_SCHEMA']}.{row['REFERENCED_TABLE']}.{row['REFERENCED_COLUMN']}"
                })
            
            # Process indexes
            indexes = []
            current_index = None
            for row in index_result["data"]:
                if current_index is None or current_index["name"] != row["INDEX_NAME"]:
                    if current_index:
                        indexes.append(current_index)
                    current_index = {
                        "name": row["INDEX_NAME"],
                        "columns": [row["COLUMN_NAME"]],
                        "unique": bool(row["IS_UNIQUE"])
                    }
                else:
                    current_index["columns"].append(row["COLUMN_NAME"])
            
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
            logger.error(f"Error getting Oracle table schema: {str(e)}")
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
        
        # If schema not provided, use the current user's schema
        if not schema:
            # Get connection parameters to extract the username
            conn_params = await self._get_connection_params()
            schema = conn_params["user"].upper()
        else:
            # Oracle schema names are typically uppercase
            schema = schema.upper()
        
        # Oracle table names are typically uppercase
        table_name = table_name.upper()
        
        # Calculate pagination
        offset = (page - 1) * page_size
        
        # Build the base query
        # Note: Oracle requires a different pagination approach
        base_query = f'SELECT * FROM "{schema}"."{table_name}"'
        count_query = f'SELECT COUNT(*) AS total FROM "{schema}"."{table_name}"'
        
        # Add filter conditions
        bind_params = {}
        if filter_conditions:
            where_clauses = []
            param_index = 1
            
            for column, value in filter_conditions.items():
                param_name = f"p{param_index}"
                
                # Handle different types of filters
                if isinstance(value, dict):
                    # Advanced filter with operators
                    for op, val in value.items():
                        op_param_name = f"p{param_index}"
                        param_index += 1
                        
                        if op == "eq":
                            where_clauses.append(f'"{column}" = :{op_param_name}')
                            bind_params[op_param_name] = val
                        elif op == "neq":
                            where_clauses.append(f'"{column}" != :{op_param_name}')
                            bind_params[op_param_name] = val
                        elif op == "gt":
                            where_clauses.append(f'"{column}" > :{op_param_name}')
                            bind_params[op_param_name] = val
                        elif op == "gte":
                            where_clauses.append(f'"{column}" >= :{op_param_name}')
                            bind_params[op_param_name] = val
                        elif op == "lt":
                            where_clauses.append(f'"{column}" < :{op_param_name}')
                            bind_params[op_param_name] = val
                        elif op == "lte":
                            where_clauses.append(f'"{column}" <= :{op_param_name}')
                            bind_params[op_param_name] = val
                        elif op == "like":
                            where_clauses.append(f'"{column}" LIKE :{op_param_name}')
                            bind_params[op_param_name] = val
                        elif op == "in":
                            if isinstance(val, list):
                                in_params = []
                                for i, item in enumerate(val):
                                    item_param_name = f"{op_param_name}_{i}"
                                    in_params.append(f":{item_param_name}")
                                    bind_params[item_param_name] = item
                                where_clauses.append(f'"{column}" IN ({", ".join(in_params)})')
                            else:
                                where_clauses.append(f'"{column}" = :{op_param_name}')
                                bind_params[op_param_name] = val
                        else:
                            raise ValueError(f"Unsupported operator: {op}")
                else:
                    # Simple equality filter
                    where_clauses.append(f'"{column}" = :{param_name}')
                    bind_params[param_name] = value
                    param_index += 1
            
            if where_clauses:
                where_clause = " AND ".join(where_clauses)
                base_query += f" WHERE {where_clause}"
                count_query += f" WHERE {where_clause}"
        
        # Oracle pagination using ROW_NUMBER()
        # Note: For Oracle 12c and above, you can use OFFSET/FETCH syntax
        order_by = f'"{sort_by}" {sort_order}' if sort_by else 'ROWNUM'
        
        query = f"""
        SELECT * FROM (
            SELECT a.*, ROWNUM rnum FROM (
                {base_query}
                ORDER BY {order_by}
            ) a
            WHERE ROWNUM <= :end_row
        )
        WHERE rnum > :start_row
        """
        
        # Add pagination parameters
        bind_params["start_row"] = offset
        bind_params["end_row"] = offset + page_size
        
        try:
            # Get total count
            count_result = await self.execute_query(count_query, bind_params)
            total_rows = int(count_result["data"][0]["TOTAL"]) if count_result["data"] else 0
            
            # Get data
            data_result = await self.execute_query(query, bind_params)
            
            # Return paginated data
            return {
                "columns": data_result["columns"],
                "data": data_result["data"],
                "total_rows": total_rows,
                "page": page,
                "page_size": page_size,
                "total_pages": (total_rows + page_size - 1) // page_size if page_size > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting Oracle table data: {str(e)}")
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
                    await cursor.execute("SELECT 1 FROM DUAL")
                    result = await cursor.fetchone()
                    return result[0] == 1
        
        except Exception as e:
            logger.error(f"Oracle connection test failed: {str(e)}")
            return False
    
    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            try:
                # Oracle's async pool needs to be closed in a thread
                await asyncio.to_thread(self.pool.close)
                self.pool = None
                logger.info("Closed Oracle connection pool")
            except Exception as e:
                logger.error(f"Error closing Oracle connection pool: {str(e)}")
    
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
                ' merge ', ' call ', ' begin ', ' declare ',  # Oracle-specific
                ';insert ', ';update ', ';delete ', ';truncate ',
                ';create ', ';alter ', ';drop ', ';grant ', ';revoke ',
                ';merge ', ';call ', ';begin ', ';declare '
            ]
            return not any(statement in f" {clean_query} " for statement in data_modification)
        
        return False int(