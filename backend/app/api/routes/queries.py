from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import uuid
import json
import time

from app.db.database import get_db
from app.models.query import NaturalLanguageQuery as QueryModel
from app.models.user import User
from app.services.llm_service import LLMService

# Router
router = APIRouter(
    prefix="/queries",
    tags=["queries"]
)

# Request/Response models
class NaturalLanguageQueryRequest(BaseModel):
    query: str
    use_cache: bool = True
    include_explanation: bool = True

class QueryResponse(BaseModel):
    query_id: str
    query: str
    sql: str
    results: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float
    timestamp: datetime
    model: str
    success: bool
    error: Optional[str] = None
    explanation: Optional[str] = None
    cached: bool = False

class SavedQuery(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    query: str
    sql: Optional[str] = None
    created_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    use_count: int = 0
    is_public: bool = False
    tags: List[str] = []

# Initialize LLM service
llm_service = LLMService()

# Helper function to get current user (simplified for now)
async def get_current_user():
    # For now, return a dummy user - in production, this would validate JWT token
    return {"id": "user-001", "username": "admin", "role": "admin"}

# Helper function to execute SQL
def execute_sql_query(db: Session, sql: str) -> tuple[List[Dict[str, Any]], float]:
    """Execute SQL query and return results with execution time"""
    start_time = time.time()
    
    try:
        result = db.execute(text(sql))
        
        # Handle SELECT queries
        if sql.strip().upper().startswith('SELECT'):
            columns = result.keys()
            rows = result.fetchall()
            
            # Convert rows to list of dicts
            results = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    # Handle special types
                    if hasattr(value, 'isoformat'):  # datetime
                        row_dict[col] = value.isoformat()
                    elif isinstance(value, uuid.UUID):
                        row_dict[col] = str(value)
                    elif isinstance(value, (int, float, str, bool)) or value is None:
                        row_dict[col] = value
                    else:
                        row_dict[col] = str(value)
                results.append(row_dict)
            
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            return results, execution_time
        else:
            # For non-SELECT queries, return affected rows
            db.commit()
            affected_rows = result.rowcount
            execution_time = (time.time() - start_time) * 1000
            return [{"affected_rows": affected_rows}], execution_time
            
    except Exception as e:
        db.rollback()
        raise e

# Main endpoint to process natural language queries
@router.post("/natural-language", response_model=QueryResponse)
async def process_natural_language_query(
    request: NaturalLanguageQueryRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Process a natural language query and return results from the database"""
    
    query_id = f"q-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    start_time = datetime.now()
    
    try:
        # Check cache first if enabled
        cached_result = None
        if request.use_cache:
            # Check if we have a cached query
            cached_query = db.query(QueryModel).filter(
                QueryModel.natural_language_query == request.query,
                QueryModel.success == True
            ).order_by(QueryModel.created_at.desc()).first()
            
            if cached_query and cached_query.generated_sql:
                # Execute the cached SQL
                try:
                    results, exec_time = execute_sql_query(db, cached_query.generated_sql)
                    
                    return QueryResponse(
                        query_id=query_id,
                        query=request.query,
                        sql=cached_query.generated_sql,
                        results=results,
                        row_count=len(results),
                        execution_time_ms=exec_time,
                        timestamp=datetime.now(),
                        model=cached_query.model_used or "cached",
                        success=True,
                        explanation=cached_query.explanation,
                        cached=True
                    )
                except Exception:
                    # If cached SQL fails, continue to generate new SQL
                    pass
        
        # Get database schema information
        schema_info = await get_database_schema(db)
        
        # Generate SQL using LLM
        llm_response = await llm_service.generate_sql(
            query=request.query,
            schema=schema_info,
            include_explanation=request.include_explanation
        )
        
        generated_sql = llm_response.get("sql", "").strip()
        explanation = llm_response.get("explanation", "")
        model_used = llm_response.get("model", "tinyllama")
        
        # Validate SQL (basic check)
        if not generated_sql:
            raise ValueError("No SQL generated")
        
        # Clean up SQL (remove markdown code blocks if present)
        if generated_sql.startswith("```sql"):
            generated_sql = generated_sql[6:]
        if generated_sql.startswith("```"):
            generated_sql = generated_sql[3:]
        if generated_sql.endswith("```"):
            generated_sql = generated_sql[:-3]
        generated_sql = generated_sql.strip()
        
        # Execute the generated SQL
        results, exec_time = execute_sql_query(db, generated_sql)
        
        # Save query to history
        query_record = QueryModel(
            user_id=current_user["id"],
            query_text=request.query,
            generated_sql=generated_sql,
            is_successful=True,
            response_time_ms=int(exec_time),
            result_count=len(results),
            model_used=model_used,
            explanation=explanation,
            created_at=datetime.now()
        )
        db.add(query_record)
        db.commit()
        
        return QueryResponse(
            query_id=query_id,
            query=request.query,
            sql=generated_sql,
            results=results,
            row_count=len(results),
            execution_time_ms=exec_time,
            timestamp=datetime.now(),
            model=model_used,
            success=True,
            explanation=explanation,
            cached=False
        )
        
    except SQLAlchemyError as e:
        # Log SQL execution error
        error_msg = f"SQL execution error: {str(e)}"
        
        # Save failed query
        query_record = QueryModel(
            user_id=current_user["id"],
            query_text=request.query,
            generated_sql=generated_sql if 'generated_sql' in locals() else None,
            is_successful=False,
            error_message=error_msg,
            created_at=datetime.now()
        )
        db.add(query_record)
        db.commit()
        
        return QueryResponse(
            query_id=query_id,
            query=request.query,
            sql=generated_sql if 'generated_sql' in locals() else "",
            results=[],
            row_count=0,
            execution_time_ms=0,
            timestamp=datetime.now(),
            model=model_used if 'model_used' in locals() else "error",
            success=False,
            error=error_msg
        )
        
    except Exception as e:
        # Log general error
        error_msg = f"Error processing query: {str(e)}"
        
        return QueryResponse(
            query_id=query_id,
            query=request.query,
            sql="",
            results=[],
            row_count=0,
            execution_time_ms=0,
            timestamp=datetime.now(),
            model="error",
            success=False,
            error=error_msg
        )

# Get query history
@router.get("/history")
async def get_query_history(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get query execution history for the current user"""
    
    queries = db.query(QueryModel).filter(
        QueryModel.user_id == current_user["id"]
    ).order_by(QueryModel.created_at.desc()).limit(limit).all()
    
    history = []
    for q in queries:
        history.append({
            "id": str(q.id),
            "query": q.natural_language_query,
            "sql": q.generated_sql,
            "success": q.success,
            "timestamp": q.created_at.isoformat() if q.created_at else None,
            "execution_time_ms": q.execution_time_ms,
            "result_count": q.result_count,
            "error": q.error_message
        })
    
    return {"history": history, "count": len(history)}

# Save a query
@router.post("/save")
async def save_query(
    query: SavedQuery,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Save a query for future use"""
    
    # Generate ID if not provided
    if not query.id:
        query.id = f"sq-{uuid.uuid4().hex}"
    
    # Set timestamps
    query.created_at = datetime.now()
    
    # Save to database (you would need a SavedQuery model)
    # For now, save as a special type of NaturalLanguageQuery
    saved_query = QueryModel(
        user_id=current_user["id"],
        query_text=query.query,
        generated_sql=query.sql,
        is_successful=True,
        metadata={
            "saved_query_id": query.id,
            "name": query.name,
            "description": query.description,
            "tags": query.tags,
            "is_public": query.is_public
        },
        created_at=query.created_at
    )
    
    db.add(saved_query)
    db.commit()
    
    return {
        "id": query.id,
        "message": "Query saved successfully",
        "query": query.dict()
    }

# Get saved queries
@router.get("/saved")
async def get_saved_queries(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all saved queries for the current user"""
    
    saved_queries = db.query(QueryModel).filter(
        QueryModel.user_id == current_user["id"],
        QueryModel.query_type == "saved"
    ).order_by(QueryModel.created_at.desc()).all()
    
    queries = []
    for q in saved_queries:
        metadata = q.metadata or {}
        queries.append({
            "id": metadata.get("saved_query_id", str(q.id)),
            "name": metadata.get("name", "Untitled Query"),
            "description": metadata.get("description"),
            "query": q.natural_language_query,
            "sql": q.generated_sql,
            "created_at": q.created_at.isoformat() if q.created_at else None,
            "tags": metadata.get("tags", []),
            "is_public": metadata.get("is_public", False)
        })
    
    return {"saved_queries": queries, "count": len(queries)}

# Query suggestions endpoint
@router.post("/suggest")
async def suggest_queries(
    prefix: str,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """Suggest queries based on prefix and history"""
    
    # Get recent successful queries that match the prefix
    recent_queries = db.query(QueryModel.natural_language_query).filter(
        QueryModel.success == True,
        QueryModel.natural_language_query.ilike(f"{prefix}%")
    ).distinct().limit(limit * 2).all()
    
    suggestions = [q[0] for q in recent_queries]
    
    # Add some common supply chain queries if needed
    common_queries = [
        "Show me products with low inventory",
        "Which suppliers have the best rating?",
        "What products are below reorder point?",
        "Show all orders from last month",
        "List top performing suppliers",
        "Display inventory levels by location",
        "What is the total value of inventory?",
        "Show pending orders",
        "Which products have no inventory?",
        "Display supplier performance metrics"
    ]
    
    # Add matching common queries
    for cq in common_queries:
        if cq.lower().startswith(prefix.lower()) and cq not in suggestions:
            suggestions.append(cq)
    
    return {"suggestions": suggestions[:limit]}

# Helper function to get database schema
async def get_database_schema(db: Session) -> Dict[str, Any]:
    """Get database schema information for LLM context"""
    
    schema_query = """
    SELECT 
        t.table_name,
        array_agg(
            json_build_object(
                'column_name', c.column_name,
                'data_type', c.data_type,
                'is_nullable', c.is_nullable
            ) ORDER BY c.ordinal_position
        ) as columns
    FROM information_schema.tables t
    JOIN information_schema.columns c ON t.table_name = c.table_name
    WHERE t.table_schema = 'public'
    AND t.table_type = 'BASE TABLE'
    GROUP BY t.table_name
    ORDER BY t.table_name;
    """
    
    result = db.execute(text(schema_query))
    
    schema = {}
    for row in result:
        schema[row.table_name] = {
            "columns": row.columns
        }
    
    return schema

# Test endpoint
@router.get("/test")
async def test_connection(db: Session = Depends(get_db)):
    """Test database connection and return basic info"""
    
    try:
        # Test query
        result = db.execute(text("SELECT COUNT(*) as count FROM products"))
        product_count = result.scalar()
        
        result = db.execute(text("SELECT COUNT(*) as count FROM suppliers"))
        supplier_count = result.scalar()
        
        result = db.execute(text("SELECT COUNT(*) as count FROM inventory"))
        inventory_count = result.scalar()
        
        return {
            "status": "connected",
            "database": "Supplychain_AI",
            "tables": {
                "products": product_count,
                "suppliers": supplier_count,
                "inventory": inventory_count
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }