from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from datetime import datetime
import uuid

from app.llm.controller.active_model_manager import get_active_model
from app.llm.prompt.template_manager import get_template
from app.llm.prompt.context_builder import build_query_context
from app.llm.prompt.schema_provider import get_database_schema
from app.db.schema.schema_discovery import discover_client_schema
from app.cache.query_cache import get_cached_query, cache_query
from app.security.rbac_manager import check_permission
from app.api.middleware.client_context import get_client_context
from app.utils.logger import get_logger
from app.db.interfaces.user_interface import User

from app.api.routes.auth import get_current_active_user

# Initialize logger
logger = get_logger(__name__)

# Router
router = APIRouter(
    prefix="/queries",
    tags=["natural language queries"],
    dependencies=[Depends(get_current_active_user)],
    responses={401: {"description": "Unauthorized"}},
)

# Models
class NaturalLanguageQuery(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    use_cache: bool = True
    client_id: Optional[str] = None
    db_connection_id: Optional[str] = None

class QueryResponse(BaseModel):
    query_id: str
    natural_query: str
    sql: str
    results: List[Dict[str, Any]]
    execution_time: float
    timestamp: datetime
    model_used: str
    cached: bool = False
    explanation: Optional[str] = None
    chart_suggestion: Optional[Dict[str, Any]] = None

class SavedQuery(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    query: str
    created_by: str
    created_at: datetime
    last_used: Optional[datetime] = None
    use_count: int = 0
    is_public: bool = False
    tags: List[str] = []
    client_id: str

# Routes
@router.post("/", response_model=QueryResponse)
async def process_query(
    query_request: NaturalLanguageQuery,
    request: Request,
    current_user: User = Depends(get_current_active_user),
):
    """Process a natural language query and convert it to SQL"""
    # Check user has permission to query
    check_permission(current_user.role, "queries:execute")
    
    # Get client context
    client_id = query_request.client_id or await get_client_context(request) or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client context required"
        )
    
    # Check for cached query if enabled
    if query_request.use_cache:
        cached_result = get_cached_query(query_request.query, client_id)
        if cached_result:
            logger.info(f"Returning cached query result for: {query_request.query[:50]}...")
            return {**cached_result, "cached": True}
    
    # Start timing execution
    start_time = datetime.now()
    
    # Get active LLM model
    llm_model = get_active_model()
    
    try:
        # Discover database schema for the client
        db_schema = await discover_client_schema(
            client_id=client_id, 
            connection_id=query_request.db_connection_id
        )
        
        # Prepare schema context for LLM
        schema_context = get_database_schema(db_schema)
        
        # Get prompt template
        template = get_template("query_translation")
        
        # Build context for the query
        context = build_query_context(
            query=query_request.query,
            schema=schema_context,
            additional_context=query_request.context
        )
        
        # Execute LLM to translate natural language to SQL
        llm_response = await llm_model.generate(
            prompt_template=template,
            context=context
        )
        
        # Extract information from LLM response
        sql = llm_response.get("sql", "")
        explanation = llm_response.get("explanation", "")
        
        # Execute the SQL against the client's database
        # This would be handled by a database connector that's configured for the client
        from app.db.connectors.postgres import PostgresConnector  # This would be dynamically selected
        db_connector = PostgresConnector(client_id=client_id, connection_id=query_request.db_connection_id)
        results = await db_connector.execute_query(sql)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Get chart suggestion based on the query and results
        from app.visualization.recommendation_engine import recommend_chart_type
        chart_suggestion = recommend_chart_type(query_request.query, results)
        
        # Create response
        response = QueryResponse(
            query_id=f"q-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
            natural_query=query_request.query,
            sql=sql,
            results=results,
            execution_time=execution_time,
            timestamp=datetime.now(),
            model_used=llm_model.name,
            explanation=explanation,
            chart_suggestion=chart_suggestion
        )
        
        # Cache the query result if caching is enabled
        if query_request.use_cache:
            cache_query(query_request.query, client_id, response.dict())
            
        logger.info(f"Processed query: {query_request.query[:50]}... in {execution_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@router.get("/saved", response_model=List[SavedQuery])
async def get_saved_queries(
    client_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
):
    """Get all saved queries for the current user"""
    # Check user has permission
    check_permission(current_user.role, "queries:view")
    
    # Use provided client_id or user's client_id
    client_id = client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    # Get saved queries from database
    from app.db.interfaces.user_interface import UserInterface
    user_interface = UserInterface()
    saved_queries = await user_interface.get_saved_queries(
        user_id=current_user.id,
        client_id=client_id
    )
    
    logger.info(f"Retrieved {len(saved_queries)} saved queries for user: {current_user.username}")
    return saved_queries

@router.post("/save", response_model=SavedQuery)
async def save_query(
    query: SavedQuery,
    current_user: User = Depends(get_current_active_user),
):
    """Save a query for future use"""
    # Check user has permission
    check_permission(current_user.role, "queries:save")
    
    # Ensure client_id is set
    if not query.client_id:
        query.client_id = current_user.client_id
        
    if not query.client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    # Generate ID if not provided
    if not query.id:
        query.id = f"sq-{uuid.uuid4().hex}"
    
    # Set created_by and created_at
    query.created_by = current_user.id
    query.created_at = datetime.now()
    
    # Save query to database
    from app.db.interfaces.user_interface import UserInterface
    user_interface = UserInterface()
    saved_query = await user_interface.save_query(query.dict())
    
    logger.info(f"Query saved: {query.name}")
    return saved_query

@router.get("/history", response_model=List[QueryResponse])
async def get_query_history(
    limit: int = 10,
    client_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
):
    """Get query execution history for the current user"""
    # Check user has permission
    check_permission(current_user.role, "queries:history")
    
    # Use provided client_id or user's client_id
    client_id = client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    # Get query history from database
    from app.db.interfaces.user_interface import UserInterface
    user_interface = UserInterface()
    history = await user_interface.get_query_history(
        user_id=current_user.id,
        client_id=client_id,
        limit=limit
    )
    
    logger.info(f"Retrieved query history for user: {current_user.username}")
    return history

@router.delete("/saved/{query_id}")
async def delete_saved_query(
    query_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Delete a saved query"""
    # Check user has permission
    check_permission(current_user.role, "queries:delete")
    
    # Delete query from database
    from app.db.interfaces.user_interface import UserInterface
    user_interface = UserInterface()
    deleted = await user_interface.delete_saved_query(
        query_id=query_id,
        user_id=current_user.id
    )
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved query not found"
        )
    
    logger.info(f"Saved query deleted: {query_id}")
    return {"detail": "Query deleted successfully"}

@router.post("/suggest", response_model=List[str])
async def suggest_queries(
    query_prefix: str,
    client_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
):
    """Suggest queries based on a prefix and user history"""
    # Check user has permission
    check_permission(current_user.role, "queries:view")
    
    # Use provided client_id or user's client_id
    client_id = client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    # Get active LLM model
    llm_model = get_active_model()
    
    try:
        # Discover database schema for the client
        db_schema = await discover_client_schema(client_id=client_id)
        
        # Prepare schema context for LLM
        schema_context = get_database_schema(db_schema)
        
        # Get prompt template
        template = get_template("query_suggestions")
        
        # Build context for the suggestions
        context = {
            "query_prefix": query_prefix,
            "schema": schema_context,
            "domain": "supply_chain"
        }
        
        # Execute LLM to generate suggestions
        llm_response = await llm_model.generate(
            prompt_template=template,
            context=context
        )
        
        # Extract suggestions from LLM response
        suggestions = llm_response.get("suggestions", [])
        
        logger.info(f"Generated {len(suggestions)} query suggestions for prefix: {query_prefix}")
        return suggestions
        
    except Exception as e:
        logger.error(f"Error generating query suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating query suggestions: {str(e)}"
        )