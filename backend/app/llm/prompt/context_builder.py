# app/llm/prompt/context_builder.py

from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime

from app.utils.logger import get_logger
from app.config import get_settings
from app.llm.prompt.schema_provider import get_database_schema

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

async def build_query_context(
    query: str,
    schema: Dict[str, Any] = None,
    client_id: Optional[str] = None,
    connection_id: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a context dictionary for a natural language query.
    
    Args:
        query: Natural language query
        schema: Optional schema information
        client_id: Optional client ID
        connection_id: Optional connection ID
        additional_context: Optional additional context
        
    Returns:
        Context dictionary
    """
    context = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "domain": "supply_chain"
    }
    
    # Add schema if provided
    if schema:
        context["schema"] = schema
    # Fetch schema if not provided but client ID is available
    elif client_id:
        try:
            schema_data = await get_database_schema(client_id, connection_id)
            context["schema"] = schema_data
        except Exception as e:
            logger.error(f"Error fetching schema for context: {str(e)}")
    
    # Add client-specific context if available
    if client_id:
        context["client_id"] = client_id
        
        # Add client settings if available
        from app.api.middleware.client_context import get_client_settings
        try:
            # This is a placeholder - in a real implementation, you'd get the 
            # current request context from somewhere
            request = None
            if request:
                client_settings = await get_client_settings(request)
                if client_settings:
                    context["client_settings"] = client_settings
        except Exception as e:
            logger.error(f"Error getting client settings for context: {str(e)}")
    
    # Add database connection details if available
    if connection_id:
        context["connection_id"] = connection_id
    
    # Add additional context
    if additional_context:
        for key, value in additional_context.items():
            context[key] = value
    
    # Add recent queries for context if available
    try:
        recent_queries = await _get_recent_queries(client_id)
        if recent_queries:
            context["recent_queries"] = recent_queries
    except Exception as e:
        logger.error(f"Error getting recent queries for context: {str(e)}")
    
    # Add relevant tables for this query based on schema analysis
    if "schema" in context:
        try:
            relevant_tables = _identify_relevant_tables(query, context["schema"])
            if relevant_tables:
                context["relevant_tables"] = relevant_tables
        except Exception as e:
            logger.error(f"Error identifying relevant tables: {str(e)}")
    
    # Add domain-specific context based on query intent
    try:
        domain_context = _add_domain_context(query)
        if domain_context:
            context["domain_context"] = domain_context
    except Exception as e:
        logger.error(f"Error adding domain context: {str(e)}")
    
    return context

async def build_analysis_context(
    analysis_type: str,
    parameters: Dict[str, Any],
    schema: Dict[str, Any] = None,
    client_id: Optional[str] = None,
    connection_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build context for an analytics query.
    
    Args:
        analysis_type: Type of analysis (e.g., "inventory", "supplier")
        parameters: Analysis parameters
        schema: Optional schema information
        client_id: Optional client ID
        connection_id: Optional connection ID
        
    Returns:
        Context dictionary
    """
    # Start with base context
    context = await build_query_context(
        query=f"Perform {analysis_type} analysis",
        schema=schema,
        client_id=client_id,
        connection_id=connection_id,
        additional_context={"analysis_type": analysis_type}
    )
    
    # Add analysis parameters
    context["parameters"] = parameters
    
    # Add analysis-specific templates and guidance
    if analysis_type == "inventory":
        context["metrics"] = [
            "stock_levels", "inventory_turns", "days_of_supply",
            "stockout_rate", "fill_rate", "carrying_cost"
        ]
        context["dimensions"] = [
            "product", "location", "time", "category"
        ]
    elif analysis_type == "supplier":
        context["metrics"] = [
            "on_time_delivery", "quality_rating", "price_variance",
            "lead_time", "responsiveness", "compliance"
        ]
        context["dimensions"] = [
            "supplier", "category", "time", "location"
        ]
    elif analysis_type == "logistics":
        context["metrics"] = [
            "transit_time", "on_time_delivery", "cost_per_mile",
            "damage_rate", "capacity_utilization", "carbon_footprint"
        ]
        context["dimensions"] = [
            "carrier", "route", "mode", "time", "product"
        ]
    
    return context

async def _get_recent_queries(client_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get recent queries for context.
    
    Args:
        client_id: Optional client ID
        
    Returns:
        List of recent queries
    """
    # In a real implementation, you'd query a database for recent queries
    # This is a placeholder implementation
    return [
        {"query": "Show me inventory levels below safety stock", "timestamp": "2023-06-01T10:00:00"},
        {"query": "List top suppliers by on-time delivery rate", "timestamp": "2023-06-01T09:30:00"},
        {"query": "What was our fill rate last month by product category?", "timestamp": "2023-05-31T16:45:00"}
    ]

def _identify_relevant_tables(query: str, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Identify tables relevant to a query.
    
    Args:
        query: Natural language query
        schema: Database schema
        
    Returns:
        List of relevant tables
    """
    relevant_tables = []
    
    # Extract tables from schema
    tables = schema.get("tables", [])
    
    # Simple keyword matching for relevance
    # In a real implementation, this could use embeddings or more sophisticated NLP
    keywords = query.lower().split()
    
    for table in tables:
        table_name = table.get("name", "").lower()
        relevance_score = 0
        
        # Check table name relevance
        for keyword in keywords:
            if keyword in table_name:
                relevance_score += 3
        
        # Check column relevance
        columns = table.get("columns", [])
        for column in columns:
            column_name = column.get("name", "").lower()
            for keyword in keywords:
                if keyword in column_name:
                    relevance_score += 1
        
        # Add table if it seems relevant
        if relevance_score > 0:
            relevant_tables.append({
                "name": table.get("name"),
                "relevance_score": relevance_score,
                "columns": [c.get("name") for c in columns]
            })
    
    # Sort by relevance
    relevant_tables.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Return top 5 most relevant tables
    return relevant_tables[:5]

def _add_domain_context(query: str) -> Dict[str, Any]:
    """
    Add domain-specific context based on query intent.
    
    Args:
        query: Natural language query
        
    Returns:
        Domain context
    """
    # Simple rule-based intent detection
    # In a real implementation, this could use more sophisticated NLP
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["inventory", "stock", "level", "on hand"]):
        return {
            "domain": "inventory",
            "key_metrics": ["quantity", "value", "days_of_supply"],
            "common_analyses": ["stock_level", "abc_analysis", "safety_stock"]
        }
    elif any(word in query_lower for word in ["supplier", "vendor", "performance"]):
        return {
            "domain": "supplier",
            "key_metrics": ["on_time_delivery", "quality", "cost"],
            "common_analyses": ["scorecard", "risk", "performance"]
        }
    elif any(word in query_lower for word in ["order", "purchase", "po", "sales"]):
        return {
            "domain": "orders",
            "key_metrics": ["order_value", "fill_rate", "cycle_time"],
            "common_analyses": ["order_status", "fill_rate", "backorders"]
        }
    elif any(word in query_lower for word in ["ship", "deliver", "transport", "logistics"]):
        return {
            "domain": "logistics",
            "key_metrics": ["transit_time", "cost", "on_time_delivery"],
            "common_analyses": ["carrier_performance", "route_analysis", "delivery_status"]
        }
    
    # Default to general supply chain context
    return {
        "domain": "supply_chain",
        "key_metrics": ["cost", "service_level", "cycle_time"],
        "common_analyses": ["performance", "optimization", "risk"]
    }