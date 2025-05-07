from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import json

from app.db.schema.schema_discovery import discover_client_schema, get_table_schema
from app.db.connectors.postgres import PostgresConnector  # This would be dynamically selected based on client
from app.db.interfaces.user_interface import User
from app.security.rbac_manager import check_permission
from app.utils.logger import get_logger
from app.api.middleware.client_context import get_client_context
from app.api.routes.auth import get_current_active_user

# Initialize logger
logger = get_logger(__name__)

# Router
router = APIRouter(
    prefix="/database",
    tags=["database"],
    dependencies=[Depends(get_current_active_user)],
    responses={401: {"description": "Unauthorized"}}
)

# Models
class Column(BaseModel):
    """Database column metadata"""
    name: str
    data_type: str
    nullable: bool
    primary_key: bool = False
    foreign_key: Optional[str] = None
    description: Optional[str] = None

class Table(BaseModel):
    """Database table metadata"""
    name: str
    schema: str
    columns: List[Column]
    primary_key: Optional[List[str]] = None
    foreign_keys: Optional[List[Dict[str, str]]] = None
    description: Optional[str] = None
    row_count: Optional[int] = None

class DatabaseSchema(BaseModel):
    """Database schema information"""
    tables: List[Table]
    views: Optional[List[Dict[str, Any]]] = None
    relationships: Optional[List[Dict[str, str]]] = None

class TableData(BaseModel):
    """Table data with pagination"""
    table_name: str
    columns: List[str]
    data: List[Dict[str, Any]]
    total_rows: int
    page: int
    page_size: int
    total_pages: int

class QueryRequest(BaseModel):
    """Raw SQL query request"""
    query: str
    params: Optional[Dict[str, Any]] = None
    client_id: Optional[str] = None
    connection_id: Optional[str] = None

class QueryResult(BaseModel):
    """SQL query result"""
    columns: List[str]
    data: List[Dict[str, Any]]
    row_count: int
    execution_time: float
    query: str
    timestamp: datetime

class Connection(BaseModel):
    """Database connection information"""
    id: str
    name: str
    type: str  # postgres, mysql, sqlserver, oracle
    host: str
    port: int
    database: str
    username: str
    password: Optional[str] = None  # This would be encrypted in storage
    schema: Optional[str] = None
    ssl_enabled: bool = False
    created_by: str
    created_at: datetime
    last_used: Optional[datetime] = None
    client_id: str
    description: Optional[str] = None

# Routes
@router.get("/schema", response_model=DatabaseSchema)
async def get_database_schema(
    client_id: Optional[str] = None,
    connection_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get the database schema for a client"""
    # Check user has permission
    check_permission(current_user.role, "database:schema:view")
    
    # Get client ID
    client_id = client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Discover the schema
        schema = await discover_client_schema(client_id, connection_id)
        
        logger.info(f"Retrieved database schema for client: {client_id}")
        return schema
        
    except Exception as e:
        logger.error(f"Error retrieving database schema: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving database schema: {str(e)}"
        )

@router.get("/tables", response_model=List[str])
async def get_tables(
    client_id: Optional[str] = None,
    connection_id: Optional[str] = None,
    schema: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get list of tables in the database"""
    # Check user has permission
    check_permission(current_user.role, "database:tables:list")
    
    # Get client ID
    client_id = client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get database connector
        db_connector = PostgresConnector(client_id=client_id, connection_id=connection_id)
        
        # Get table list
        tables = await db_connector.get_tables(schema=schema)
        
        logger.info(f"Retrieved table list for client: {client_id}")
        return tables
        
    except Exception as e:
        logger.error(f"Error retrieving tables: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving tables: {str(e)}"
        )

@router.get("/tables/{table_name}", response_model=TableData)
async def get_table_data(
    table_name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    sort_by: Optional[str] = None,
    sort_order: str = Query("asc", regex="^(asc|desc)$"),
    filter: Optional[str] = None,  # JSON string of filter conditions
    client_id: Optional[str] = None,
    connection_id: Optional[str] = None,
    schema: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get paginated data from a table"""
    # Check user has permission
    check_permission(current_user.role, "database:data:view")
    
    # Get client ID
    client_id = client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Parse filter if provided
        filter_conditions = None
        if filter:
            try:
                filter_conditions = json.loads(filter)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid filter JSON"
                )
        
        # Get database connector
        db_connector = PostgresConnector(client_id=client_id, connection_id=connection_id)
        
        # Get table data
        result = await db_connector.get_table_data(
            table_name=table_name,
            schema=schema,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
            filter_conditions=filter_conditions
        )
        
        # Create response
        response = TableData(
            table_name=table_name,
            columns=result.get("columns", []),
            data=result.get("data", []),
            total_rows=result.get("total_rows", 0),
            page=page,
            page_size=page_size,
            total_pages=(result.get("total_rows", 0) + page_size - 1) // page_size
        )
        
        logger.info(f"Retrieved data from table {table_name} for client: {client_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving table data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving table data: {str(e)}"
        )

@router.post("/query", response_model=QueryResult)
async def execute_query(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Execute a raw SQL query (read-only)"""
    # Check user has elevated permission for raw SQL
    check_permission(current_user.role, "database:query:execute")
    
    # Get client ID
    client_id = query_request.client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Validate query is read-only to prevent modifications
        query = query_request.query.strip().lower()
        if not query.startswith("select"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only SELECT queries are allowed"
            )
        
        # Start timing execution
        start_time = datetime.now()
        
        # Get database connector
        db_connector = PostgresConnector(
            client_id=client_id, 
            connection_id=query_request.connection_id
        )
        
        # Execute query
        result = await db_connector.execute_query(
            query=query_request.query,
            params=query_request.params
        )
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Create response
        response = QueryResult(
            columns=result.get("columns", []),
            data=result.get("data", []),
            row_count=len(result.get("data", [])),
            execution_time=execution_time,
            query=query_request.query,
            timestamp=datetime.now()
        )
        
        logger.info(f"Executed query for client: {client_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing query: {str(e)}"
        )

@router.get("/table/{table_name}/schema", response_model=Table)
async def get_table_schema_endpoint(
    table_name: str,
    client_id: Optional[str] = None,
    connection_id: Optional[str] = None,
    schema: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get detailed schema for a specific table"""
    # Check user has permission
    check_permission(current_user.role, "database:schema:view")
    
    # Get client ID
    client_id = client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get table schema
        table_schema = await get_table_schema(
            client_id=client_id,
            connection_id=connection_id,
            table_name=table_name,
            schema=schema
        )
        
        if not table_schema:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Table not found: {table_name}"
            )
        
        logger.info(f"Retrieved schema for table {table_name}")
        return table_schema
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving table schema: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving table schema: {str(e)}"
        )

@router.get("/connections", response_model=List[Connection])
async def get_connections(
    client_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get all database connections for a client"""
    # Check user has permission
    check_permission(current_user.role, "database:connections:view")
    
    # Get client ID
    client_id = client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # In a real implementation, you would fetch from database
        # For demonstration, we'll return mock data
        connections = [
            Connection(
                id="conn-1",
                name="Production Database",
                type="postgres",
                host="db.example.com",
                port=5432,
                database="supply_chain_db",
                username="readonly_user",
                schema="public",
                ssl_enabled=True,
                created_by=current_user.id,
                created_at=datetime.now(),
                last_used=datetime.now(),
                client_id=client_id,
                description="Main production database"
            ),
            Connection(
                id="conn-2",
                name="Data Warehouse",
                type="postgres",
                host="warehouse.example.com",
                port=5432,
                database="analytics_db",
                username="readonly_user",
                schema="public",
                ssl_enabled=True,
                created_by=current_user.id,
                created_at=datetime.now(),
                last_used=datetime.now(),
                client_id=client_id,
                description="Data warehouse for analytics"
            )
        ]
        
        logger.info(f"Retrieved connections for client: {client_id}")
        return connections
        
    except Exception as e:
        logger.error(f"Error retrieving connections: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving connections: {str(e)}"
        )

@router.post("/connections", response_model=Connection)
async def create_connection(
    connection: Connection,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new database connection"""
    # Check user has permission
    check_permission(current_user.role, "database:connections:create")
    
    # Ensure client_id is set
    if not connection.client_id:
        connection.client_id = current_user.client_id
        
    if not connection.client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Set created_by and created_at
        connection.created_by = current_user.id
        connection.created_at = datetime.now()
        
        # Test connection
        db_connector = PostgresConnector(
            host=connection.host,
            port=connection.port,
            database=connection.database,
            username=connection.username,
            password=connection.password,
            ssl_enabled=connection.ssl_enabled
        )
        
        success = await db_connector.test_connection()
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to connect to database with provided credentials"
            )
        
        # In a real implementation, you would save to database
        # For demonstration, we'll just generate an ID
        connection.id = f"conn-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Remove password from response
        connection_response = connection.dict()
        connection_response.pop("password", None)
        
        logger.info(f"Created database connection: {connection.name}")
        return Connection(**connection_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating connection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating connection: {str(e)}"
        )

@router.delete("/connections/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_connection(
    connection_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a database connection"""
    # Check user has permission
    check_permission(current_user.role, "database:connections:delete")
    
    try:
        # In a real implementation, you would delete from database
        logger.info(f"Deleted connection: {connection_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error deleting connection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting connection: {str(e)}"
        )

@router.get("/relationships", response_model=List[Dict[str, str]])
async def get_relationships(
    client_id: Optional[str] = None,
    connection_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get all detected relationships between tables"""
    # Check user has permission
    check_permission(current_user.role, "database:schema:view")
    
    # Get client ID
    client_id = client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get schema
        schema = await discover_client_schema(client_id, connection_id)
        
        # Extract relationships
        relationships = schema.relationships or []
        
        logger.info(f"Retrieved table relationships for client: {client_id}")
        return relationships
        
    except Exception as e:
        logger.error(f"Error retrieving relationships: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving relationships: {str(e)}"
        )