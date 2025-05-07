# app/llm/prompt/schema_provider.py

from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime

from app.utils.logger import get_logger
from app.db.schema.schema_discovery import discover_client_schema
from app.db.schema.schema_mapper import get_domain_mappings

# Initialize logger
logger = get_logger(__name__)

async def get_database_schema(
    client_id: str,
    connection_id: Optional[str] = None,
    max_tables: int = 50,
    include_sample_data: bool = False,
    simplified: bool = False,
    domain_mapping: bool = True
) -> Dict[str, Any]:
    """
    Get database schema information for LLM context.
    
    Args:
        client_id: Client ID
        connection_id: Optional connection ID
        max_tables: Maximum number of tables to include
        include_sample_data: Whether to include sample data
        simplified: Whether to use simplified schema format
        domain_mapping: Whether to include domain mappings
        
    Returns:
        Schema dictionary
    """
    try:
        # Discover schema
        schema = await discover_client_schema(
            client_id=client_id,
            connection_id=connection_id
        )
        
        # Get domain mappings if requested
        domain_mappings = None
        if domain_mapping:
            try:
                domain_mappings = await get_domain_mappings(client_id)
            except Exception as e:
                logger.error(f"Error getting domain mappings: {str(e)}")
        
        # Convert to LLM-friendly format
        if simplified:
            schema_data = simplify_schema(schema, max_tables, domain_mappings)
        else:
            schema_data = format_schema_for_llm(schema, max_tables, domain_mappings)
        
        # Add sample data if requested
        if include_sample_data:
            schema_data = await add_sample_data(schema_data, client_id, connection_id)
        
        return schema_data
        
    except Exception as e:
        logger.error(f"Error getting database schema: {str(e)}")
        # Return minimal schema to avoid breaking the LLM
        return {
            "database_type": "unknown",
            "tables": [],
            "relationships": [],
            "error": str(e)
        }

def simplify_schema(
    schema: Any,
    max_tables: int = 50,
    domain_mappings: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Create a simplified schema representation for LLM.
    
    Args:
        schema: Schema object
        max_tables: Maximum number of tables to include
        domain_mappings: Optional domain mappings
        
    Returns:
        Simplified schema dictionary
    """
    # Extract schema data
    tables = getattr(schema, "tables", [])[:max_tables]
    database_type = getattr(schema, "database_type", "unknown")
    
    # Build simplified schema
    simplified = {
        "database_type": database_type,
        "tables": []
    }
    
    # Convert domain mappings to a more usable format
    domain_map = {}
    if domain_mappings:
        for mapping in domain_mappings:
            table = mapping.get("custom_table")
            column = mapping.get("custom_column")
            concept = mapping.get("domain_concept")
            attribute = mapping.get("domain_attribute")
            
            if table and concept:
                if table not in domain_map:
                    domain_map[table] = {
                        "concept": concept,
                        "columns": {}
                    }
                if column and attribute:
                    domain_map[table]["columns"][column] = attribute
    
    # Process tables
    for table in tables:
        table_name = table.get("name", "")
        
        # Determine domain concept
        domain_concept = None
        if table_name in domain_map:
            domain_concept = domain_map[table_name].get("concept")
        
        # Extract columns (simplified)
        columns = []
        for column in table.get("columns", []):
            col_name = column.get("name", "")
            data_type = column.get("data_type", "")
            
            # Determine domain attribute
            domain_attribute = None
            if table_name in domain_map and "columns" in domain_map[table_name]:
                domain_attribute = domain_map[table_name]["columns"].get(col_name)
            
            columns.append({
                "name": col_name,
                "type": data_type,
                "domain_attribute": domain_attribute
            })
        
        # Add table to schema
        simplified["tables"].append({
            "name": table_name,
            "columns": columns,
            "domain_concept": domain_concept
        })
    
    return simplified

def format_schema_for_llm(
    schema: Any,
    max_tables: int = 50,
    domain_mappings: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Format schema for LLM consumption with more details.
    
    Args:
        schema: Schema object
        max_tables: Maximum number of tables to include
        domain_mappings: Optional domain mappings
        
    Returns:
        Formatted schema dictionary
    """
    # Extract schema data
    schema_dict = schema.to_dict() if hasattr(schema, "to_dict") else schema
    tables = schema_dict.get("tables", [])[:max_tables]
    database_type = schema_dict.get("database_type", "unknown")
    relationships = schema_dict.get("relationships", [])
    
    # Convert domain mappings to a more usable format
    domain_map = {}
    if domain_mappings:
        for mapping in domain_mappings:
            table = mapping.get("custom_table")
            column = mapping.get("custom_column")
            concept = mapping.get("domain_concept")
            attribute = mapping.get("domain_attribute")
            confidence = mapping.get("confidence", 0.0)
            
            if table and concept:
                if table not in domain_map:
                    domain_map[table] = {
                        "concept": concept,
                        "confidence": confidence,
                        "columns": {}
                    }
                if column and attribute:
                    domain_map[table]["columns"][column] = {
                        "attribute": attribute,
                        "confidence": confidence
                    }
    
    # Format tables with more details
    formatted_tables = []
    for table in tables:
        table_name = table.get("name", "")
        
        # Add domain concept
        domain_info = {}
        if table_name in domain_map:
            domain_info = {
                "domain_concept": domain_map[table_name].get("concept"),
                "mapping_confidence": domain_map[table_name].get("confidence")
            }
        
        # Process columns with more details
        columns = []
        for column in table.get("columns", []):
            col_name = column.get("name", "")
            
            # Add domain attribute
            column_domain = {}
            if (table_name in domain_map and 
                "columns" in domain_map[table_name] and 
                col_name in domain_map[table_name]["columns"]):
                column_domain = {
                    "domain_attribute": domain_map[table_name]["columns"][col_name].get("attribute"),
                    "mapping_confidence": domain_map[table_name]["columns"][col_name].get("confidence")
                }
            
            columns.append({
                **column,
                **column_domain
            })
        
        # Add table with all details
        formatted_tables.append({
            **table,
            "columns": columns,
            **domain_info
        })
    
    # Format schema
    formatted_schema = {
        "database_type": database_type,
        "tables": formatted_tables,
        "relationships": relationships,
        "table_count": len(formatted_tables)
    }
    
    return formatted_schema

async def add_sample_data(
    schema: Dict[str, Any],
    client_id: str,
    connection_id: Optional[str] = None,
    rows_per_table: int = 3
) -> Dict[str, Any]:
    """
    Add sample data to schema for better LLM understanding.
    
    Args:
        schema: Schema dictionary
        client_id: Client ID
        connection_id: Optional connection ID
        rows_per_table: Number of sample rows per table
        
    Returns:
        Schema with sample data
    """
    # Get database connector
    from app.db.schema.schema_discovery import get_connector_for_client
    connector = await get_connector_for_client(client_id, connection_id)
    
    try:
        # Add sample data to each table
        for table in schema.get("tables", []):
            table_name = table.get("name")
            try:
                # Query for sample data
                query = f"SELECT * FROM {table_name} LIMIT {rows_per_table}"
                result = await connector.execute_query(query)
                
                # Add sample data to table
                table["sample_data"] = result.get("data", [])
                
            except Exception as e:
                logger.warning(f"Error getting sample data for table {table_name}: {str(e)}")
                table["sample_data_error"] = str(e)
        
        return schema
    
    except Exception as e:
        logger.error(f"Error adding sample data: {str(e)}")
        return schema
    
    finally:
        # Close connector
        if connector:
            await connector.close()

def format_schema_as_markdown(schema: Dict[str, Any]) -> str:
    """
    Format schema as markdown for LLM prompt.
    
    Args:
        schema: Schema dictionary
        
    Returns:
        Markdown formatted schema
    """
    markdown = f"# Database Schema ({schema.get('database_type', 'unknown')})\n\n"
    
    # Add tables
    for i, table in enumerate(schema.get("tables", [])):
        table_name = table.get("name", f"Table_{i}")
        domain_concept = table.get("domain_concept", "")
        
        if domain_concept:
            markdown += f"## Table: {table_name} (Domain: {domain_concept})\n\n"
        else:
            markdown += f"## Table: {table_name}\n\n"
        
        # Add columns
        markdown += "| Column | Type | Domain | Description |\n"
        markdown += "| ------ | ---- | ------ | ----------- |\n"
        
        for column in table.get("columns", []):
            name = column.get("name", "")
            data_type = column.get("data_type", column.get("type", ""))
            domain = column.get("domain_attribute", "")
            is_pk = "PK" if name in table.get("primary_key", []) else ""
            is_fk = ""
            
            # Check if foreign key
            for fk in table.get("foreign_keys", []):
                if fk.get("column") == name:
                    is_fk = f"FK → {fk.get('references', '')}"
                    break
            
            description = f"{is_pk} {is_fk}".strip()
            
            markdown += f"| {name} | {data_type} | {domain} | {description} |\n"
        
        markdown += "\n"
    
    # Add relationships
    if schema.get("relationships"):
        markdown += "## Relationships\n\n"
        for rel in schema.get("relationships", []):
            markdown += f"- {rel.get('from', '')} → {rel.get('to', '')}"
            if rel.get("cardinality"):
                markdown += f" ({rel.get('cardinality', '')})"
            markdown += "\n"
    
    return markdown