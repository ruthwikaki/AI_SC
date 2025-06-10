# backend/app/api/routes/suggestions.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
import json

from app.db.database import get_db
from app.api.middleware.auth import get_current_user
from app.models.user import User
from app.models.query import QueryHistory, SavedQuery
from app.db.schema.schema_discovery import SchemaDiscovery
from app.llm.prompt.context_builder import ContextBuilder
from app.utils.logger import logger

router = APIRouter(prefix="/api/suggestions", tags=["suggestions"])

@router.get("/queries")
async def get_query_suggestions(
    partial_query: str = Query(..., description="Partial query text"),
    limit: int = Query(10, le=20),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get intelligent query suggestions based on partial input"""
    try:
        suggestions = []
        
        # 1. Get suggestions from user's query history
        history_suggestions = _get_history_based_suggestions(
            partial_query, current_user.id, db, limit=5
        )
        suggestions.extend(history_suggestions)
        
        # 2. Get popular queries from all users (anonymized)
        popular_suggestions = _get_popular_query_suggestions(
            partial_query, db, limit=5
        )
        suggestions.extend(popular_suggestions)
        
        # 3. Get schema-aware suggestions
        schema_suggestions = _get_schema_based_suggestions(
            partial_query, db, limit=5
        )
        suggestions.extend(schema_suggestions)
        
        # 4. Get saved query suggestions
        saved_suggestions = _get_saved_query_suggestions(
            partial_query, current_user.id, db, limit=5
        )
        suggestions.extend(saved_suggestions)
        
        # Remove duplicates and limit
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion['text'] not in seen:
                seen.add(suggestion['text'])
                unique_suggestions.append(suggestion)
                if len(unique_suggestions) >= limit:
                    break
        
        return {
            "suggestions": unique_suggestions,
            "query": partial_query
        }
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def get_query_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get predefined query templates"""
    try:
        templates = {
            "inventory": [
                {
                    "name": "Low Stock Alert",
                    "query": "Show me all products with stock levels below their reorder point",
                    "description": "Identify items that need reordering"
                },
                {
                    "name": "Inventory Turnover",
                    "query": "Calculate inventory turnover ratio for the last quarter",
                    "description": "Measure how quickly inventory is selling"
                },
                {
                    "name": "Dead Stock Analysis",
                    "query": "Find products that haven't sold in the last 6 months",
                    "description": "Identify slow-moving inventory"
                }
            ],
            "orders": [
                {
                    "name": "Pending Orders",
                    "query": "Show all pending orders older than 3 days",
                    "description": "Track orders requiring attention"
                },
                {
                    "name": "Order Fulfillment Rate",
                    "query": "Calculate order fulfillment rate for this month",
                    "description": "Measure order processing efficiency"
                },
                {
                    "name": "Top Customers",
                    "query": "List top 10 customers by order value this year",
                    "description": "Identify most valuable customers"
                }
            ],
            "suppliers": [
                {
                    "name": "Supplier Performance",
                    "query": "Compare supplier on-time delivery rates",
                    "description": "Evaluate supplier reliability"
                },
                {
                    "name": "Cost Analysis",
                    "query": "Show average cost changes by supplier over last 6 months",
                    "description": "Track supplier pricing trends"
                },
                {
                    "name": "Risk Assessment",
                    "query": "List suppliers with high risk scores",
                    "description": "Identify supply chain risks"
                }
            ],
            "analytics": [
                {
                    "name": "Revenue Trend",
                    "query": "Show monthly revenue trend for the last year",
                    "description": "Track revenue patterns"
                },
                {
                    "name": "Product Performance",
                    "query": "Rank products by profit margin",
                    "description": "Identify most profitable products"
                },
                {
                    "name": "Seasonal Analysis",
                    "query": "Analyze seasonal demand patterns for top products",
                    "description": "Understand demand fluctuations"
                }
            ]
        }
        
        if category and category in templates:
            return {
                "templates": templates[category],
                "category": category
            }
        
        return {"templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/autocomplete")
async def get_autocomplete_suggestions(
    field: str = Query(..., description="Field or table name"),
    value: str = Query("", description="Partial value"),
    limit: int = Query(10, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get autocomplete suggestions for specific fields"""
    try:
        schema_discovery = SchemaDiscovery(db)
        
        # Parse field (could be table.column or just column)
        parts = field.split('.')
        if len(parts) == 2:
            table_name, column_name = parts
        else:
            # Try to find column in any table
            column_name = parts[0]
            table_name = None
        
        suggestions = schema_discovery.get_column_value_suggestions(
            column_name=column_name,
            table_name=table_name,
            partial_value=value,
            limit=limit
        )
        
        return {
            "field": field,
            "suggestions": suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/related")
async def get_related_queries(
    query_id: Optional[int] = Query(None, description="Base query ID"),
    query_text: Optional[str] = Query(None, description="Base query text"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get queries related to a given query"""
    try:
        if not query_id and not query_text:
            raise HTTPException(status_code=400, detail="Either query_id or query_text required")
        
        related_queries = []
        
        if query_id:
            # Get base query
            base_query = db.query(QueryHistory).filter(
                QueryHistory.id == query_id,
                QueryHistory.user_id == current_user.id
            ).first()
            
            if not base_query:
                raise HTTPException(status_code=404, detail="Query not found")
            
            query_text = base_query.natural_language_query
        
        # Find related queries based on similarity
        # This is a simplified version - in production, you might use
        # more sophisticated similarity measures
        similar_queries = db.query(QueryHistory).filter(
            QueryHistory.user_id == current_user.id,
            QueryHistory.natural_language_query.ilike(f"%{query_text.split()[0]}%")
        ).limit(10).all()
        
        for query in similar_queries:
            if query.natural_language_query != query_text:
                related_queries.append({
                    "id": query.id,
                    "query": query.natural_language_query,
                    "executed_at": query.executed_at,
                    "similarity_score": 0.8  # Placeholder
                })
        
        return {
            "base_query": query_text,
            "related_queries": related_queries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _get_history_based_suggestions(partial_query: str, user_id: int, db: Session, limit: int) -> List[Dict]:
    """Get suggestions from user's query history"""
    suggestions = []
    
    recent_queries = db.query(QueryHistory).filter(
        QueryHistory.user_id == user_id,
        QueryHistory.natural_language_query.ilike(f"{partial_query}%")
    ).order_by(desc(QueryHistory.executed_at)).limit(limit).all()
    
    for query in recent_queries:
        suggestions.append({
            "text": query.natural_language_query,
            "type": "history",
            "metadata": {
                "last_used": query.executed_at.isoformat(),
                "execution_count": 1  # Could track this separately
            }
        })
    
    return suggestions

def _get_popular_query_suggestions(partial_query: str, db: Session, limit: int) -> List[Dict]:
    """Get popular queries from all users"""
    suggestions = []
    
    # Get most used queries matching partial
    popular = db.query(
        QueryHistory.natural_language_query,
        func.count(QueryHistory.id).label('count')
    ).filter(
        QueryHistory.natural_language_query.ilike(f"{partial_query}%")
    ).group_by(
        QueryHistory.natural_language_query
    ).order_by(
        desc('count')
    ).limit(limit).all()
    
    for query_text, count in popular:
        suggestions.append({
            "text": query_text,
            "type": "popular",
            "metadata": {
                "usage_count": count
            }
        })
    
    return suggestions

def _get_schema_based_suggestions(partial_query: str, db: Session, limit: int) -> List[Dict]:
    """Get suggestions based on schema understanding"""
    suggestions = []
    
    # Common query patterns with schema elements
    patterns = [
        "Show all {table}",
        "Count {table} where",
        "Find {table} with",
        "List {table} by",
        "Get top {table}",
        "Calculate total {metric} from {table}",
        "Compare {table} between"
    ]
    
    schema_discovery = SchemaDiscovery(db)
    tables = schema_discovery.get_tables()
    
    for pattern in patterns:
        if pattern.lower().startswith(partial_query.lower()):
            for table in tables[:3]:  # Limit tables to avoid too many suggestions
                suggestion = pattern.replace("{table}", table)
                if suggestion.lower().startswith(partial_query.lower()):
                    suggestions.append({
                        "text": suggestion,
                        "type": "schema",
                        "metadata": {
                            "pattern": pattern,
                            "table": table
                        }
                    })
                    if len(suggestions) >= limit:
                        break
    
    return suggestions[:limit]

def _get_saved_query_suggestions(partial_query: str, user_id: int, db: Session, limit: int) -> List[Dict]:
    """Get suggestions from saved queries"""
    suggestions = []
    
    saved_queries = db.query(SavedQuery).filter(
        SavedQuery.user_id == user_id,
        SavedQuery.name.ilike(f"%{partial_query}%") | 
        SavedQuery.description.ilike(f"%{partial_query}%")
    ).limit(limit).all()
    
    for saved in saved_queries:
        suggestions.append({
            "text": saved.query,
            "type": "saved",
            "metadata": {
                "name": saved.name,
                "description": saved.description,
                "saved_at": saved.created_at.isoformat()
            }
        })
    
    return suggestions