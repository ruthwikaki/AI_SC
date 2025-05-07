from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import re

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

async def detect_relationships(
    tables: List[Dict[str, Any]],
    database_type: str = "postgres"
) -> List[Dict[str, str]]:
    """
    Detect relationships between tables based on foreign keys and naming conventions.
    
    Args:
        tables: List of table schemas
        database_type: Type of database (postgres, mysql, sqlserver, oracle)
        
    Returns:
        List of relationships
    """
    relationships = []
    
    # First, detect explicit foreign key relationships
    explicit_relations = await detect_foreign_key_relationships(tables)
    relationships.extend(explicit_relations)
    
    # Then, detect implicit relationships based on naming conventions
    implicit_relations = await detect_implicit_relationships(tables, explicit_relations)
    relationships.extend(implicit_relations)
    
    # Remove duplicates and clean up
    clean_relations = await deduplicate_relationships(relationships)
    
    logger.info(f"Detected {len(explicit_relations)} explicit and {len(implicit_relations)} implicit relationships")
    return clean_relations

async def detect_foreign_key_relationships(
    tables: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Detect explicit foreign key relationships from table schemas.
    
    Args:
        tables: List of table schemas
        
    Returns:
        List of explicit relationships
    """
    relationships = []
    
    # Create a lookup of table names to schemas
    table_lookup = {
        f"{table.get('schema', 'public')}.{table['name']}": table 
        for table in tables
    }
    
    # Check each table for foreign keys
    for table in tables:
        table_name = table['name']
        schema_name = table.get('schema', 'public')
        
        # Look for foreign keys
        foreign_keys = table.get('foreign_keys', [])
        for fk in foreign_keys:
            # Parse the reference string
            if 'references' in fk and 'column' in fk:
                source_column = fk['column']
                reference = fk['references']
                
                # Parse the reference (format: schema.table.column)
                ref_parts = reference.split('.')
                if len(ref_parts) >= 3:
                    ref_schema = ref_parts[0]
                    ref_table = ref_parts[1]
                    ref_column = ref_parts[2]
                elif len(ref_parts) == 2:
                    ref_schema = schema_name  # Use current schema
                    ref_table = ref_parts[0]
                    ref_column = ref_parts[1]
                else:
                    continue  # Invalid reference format
                
                # Check if referenced table exists in our list
                ref_key = f"{ref_schema}.{ref_table}"
                if ref_key in table_lookup:
                    relationship = {
                        "from": f"{schema_name}.{table_name}.{source_column}",
                        "to": f"{ref_schema}.{ref_table}.{ref_column}",
                        "type": "foreign_key",
                        "cardinality": "many_to_one"
                    }
                    relationships.append(relationship)
    
    return relationships

async def detect_implicit_relationships(
    tables: List[Dict[str, Any]],
    existing_relationships: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Detect implicit relationships based on naming conventions.
    
    Args:
        tables: List of table schemas
        existing_relationships: List of already detected relationships
        
    Returns:
        List of implicit relationships
    """
    relationships = []
    
    # Create a set of existing relationships to avoid duplicates
    existing_relations = {
        (rel['from'], rel['to'])
        for rel in existing_relationships
    }
    
    # Create a lookup of table names to columns
    table_columns = {
        f"{table.get('schema', 'public')}.{table['name']}": [
            col['name'] for col in table.get('columns', [])
        ]
        for table in tables
    }
    
    # Create a lookup of primary keys
    primary_keys = {
        f"{table.get('schema', 'public')}.{table['name']}": table.get('primary_key', [])
        for table in tables
    }
    
    # Patterns to match potential foreign key column names
    fk_patterns = [
        # Common patterns like: user_id, userId, UserId
        (r'(\w+)_id$', r'\1'),
        (r'(\w+)Id$', r'\1'),
        (r'(\w+)ID$', r'\1')
    ]
    
    # Check each table for potential implicit foreign keys
    for table in tables:
        table_name = table['name']
        schema_name = table.get('schema', 'public')
        
        # Check each column for potential foreign key patterns
        for column in table.get('columns', []):
            column_name = column['name']
            
            # Skip primary key columns
            if column_name in table.get('primary_key', []):
                continue
            
            # Check patterns
            for pattern, replacement in fk_patterns:
                match = re.match(pattern, column_name, re.IGNORECASE)
                if match:
                    # Potential reference table name
                    ref_table_base = match.group(1)
                    
                    # Convert to singular if plural
                    if ref_table_base.endswith('s'):
                        ref_table_singular = ref_table_base[:-1]
                    else:
                        ref_table_singular = ref_table_base
                    
                    # Check potential reference tables
                    candidates = [ref_table_base, ref_table_singular]
                    
                    for candidate in candidates:
                        # Look for matching table name
                        for ref_key, pk_columns in primary_keys.items():
                            ref_schema, ref_table = ref_key.split('.')
                            
                            # Skip self-references
                            if ref_table == table_name and ref_schema == schema_name:
                                continue
                            
                            # Check if reference table matches pattern
                            if ref_table.lower() == candidate.lower() and pk_columns:
                                ref_column = pk_columns[0]  # Use first primary key column
                                
                                # Create relationship
                                from_path = f"{schema_name}.{table_name}.{column_name}"
                                to_path = f"{ref_schema}.{ref_table}.{ref_column}"
                                
                                # Skip if already exists
                                if (from_path, to_path) in existing_relations:
                                    continue
                                
                                relationship = {
                                    "from": from_path,
                                    "to": to_path,
                                    "type": "naming_convention",
                                    "cardinality": "many_to_one",
                                    "confidence": 0.7
                                }
                                relationships.append(relationship)
                                existing_relations.add((from_path, to_path))
    
    # Find many-to-many relationships through junction tables
    junction_relations = await detect_junction_tables(tables, existing_relations)
    relationships.extend(junction_relations)
    
    return relationships

async def detect_junction_tables(
    tables: List[Dict[str, Any]],
    existing_relations: Set[Tuple[str, str]]
) -> List[Dict[str, str]]:
    """
    Detect many-to-many relationships through junction tables.
    
    Args:
        tables: List of table schemas
        existing_relations: Set of existing relationship tuples
        
    Returns:
        List of many-to-many relationships
    """
    relationships = []
    
    # Create a dictionary of foreign keys by table
    table_fks = defaultdict(list)
    
    # Process existing relationships to find foreign keys per table
    for rel_from, rel_to in existing_relations:
        from_parts = rel_from.split('.')
        if len(from_parts) == 3:
            from_schema, from_table, from_column = from_parts
            table_fks[f"{from_schema}.{from_table}"].append((rel_from, rel_to))
    
    # Identify potential junction tables (tables with exactly 2 foreign keys)
    for table_key, fks in table_fks.items():
        if len(fks) == 2:
            schema, table = table_key.split('.')
            
            # Check if this looks like a junction table
            is_junction = True
            
            # Get table schema
            table_schema = None
            for t in tables:
                if t['name'] == table and t.get('schema', 'public') == schema:
                    table_schema = t
                    break
            
            if not table_schema:
                continue
            
            # Check if the table has additional non-FK columns beyond a possible 'id'
            columns = table_schema.get('columns', [])
            if len(columns) > 3:  # id + 2 FKs + something else
                is_junction = False
            
            # Check for additional non-id, non-FK columns
            fk_columns = set()
            for rel_from, _ in fks:
                fk_columns.add(rel_from.split('.')[2])
            
            # Skip junction detection if table has non-indexed columns
            for col in columns:
                col_name = col['name']
                if col_name not in fk_columns and col_name != 'id' and col_name not in table_schema.get('primary_key', []):
                    # But allow created_at, updated_at, etc.
                    if not any(col_name.endswith(suffix) for suffix in ['_at', '_on', '_date', '_time']):
                        is_junction = False
                        break
            
            if is_junction:
                # Create a many-to-many relationship between the two tables
                rel1_from, rel1_to = fks[0]
                rel2_from, rel2_to = fks[1]
                
                # Extract the endpoint tables
                table1 = '.'.join(rel1_to.split('.')[:2])
                table2 = '.'.join(rel2_to.split('.')[:2])
                
                # Create the many-to-many relationship
                relationship = {
                    "from": rel1_to,
                    "to": rel2_to,
                    "through": table_key,
                    "type": "many_to_many",
                    "cardinality": "many_to_many",
                    "confidence": 0.8
                }
                relationships.append(relationship)
    
    return relationships

async def deduplicate_relationships(
    relationships: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Remove duplicate relationships and clean up.
    
    Args:
        relationships: List of relationships
        
    Returns:
        Cleaned list of relationships
    """
    # Use a dictionary to deduplicate by from/to pairs
    unique_relations = {}
    
    for rel in relationships:
        key = (rel['from'], rel['to'])
        
        # If this relationship already exists, keep the one with higher confidence
        if key in unique_relations:
            existing_rel = unique_relations[key]
            
            # Keep explicit foreign key relationships over implicit ones
            if rel.get('type') == 'foreign_key' and existing_rel.get('type') != 'foreign_key':
                unique_relations[key] = rel
            # Otherwise keep the one with higher confidence
            elif rel.get('confidence', 0) > existing_rel.get('confidence', 0):
                unique_relations[key] = rel
        else:
            unique_relations[key] = rel
    
    return list(unique_relations.values())

async def suggest_missing_indexes(
    tables: List[Dict[str, Any]],
    relationships: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Suggest missing indexes based on relationships.
    
    Args:
        tables: List of table schemas
        relationships: List of detected relationships
        
    Returns:
        List of suggested indexes
    """
    suggested_indexes = []
    
    # Create a lookup of table names to schemas
    table_lookup = {}
    for table in tables:
        key = f"{table.get('schema', 'public')}.{table['name']}"
        table_lookup[key] = table
    
    # Create a set of existing indexes
    existing_indexes = set()
    for table in tables:
        schema_name = table.get('schema', 'public')
        table_name = table['name']
        
        # Add indexes
        for idx in table.get('indexes', []):
            for column in idx.get('columns', []):
                existing_indexes.add(f"{schema_name}.{table_name}.{column}")
        
        # Add primary key columns
        for column in table.get('primary_key', []):
            existing_indexes.add(f"{schema_name}.{table_name}.{column}")
    
    # Check each relationship for potential missing indexes
    for rel in relationships:
        # We mostly care about indexes on foreign key columns
        from_parts = rel['from'].split('.')
        if len(from_parts) == 3:
            from_schema, from_table, from_column = from_parts
            
            # Check if index already exists
            index_key = f"{from_schema}.{from_table}.{from_column}"
            if index_key not in existing_indexes:
                # Get table schema
                table_key = f"{from_schema}.{from_table}"
                if table_key in table_lookup:
                    table = table_lookup[table_key]
                    
                    # Create suggested index
                    suggestion = {
                        "table": from_table,
                        "schema": from_schema,
                        "column": from_column,
                        "reason": f"Foreign key to {rel['to']}",
                        "impact": "high" if rel.get('type') == 'foreign_key' else "medium"
                    }
                    suggested_indexes.append(suggestion)
    
    return suggested_indexes

async def detect_potential_relationships(
    tables: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Detect potential relationships that aren't explicitly defined.
    
    Args:
        tables: List of table schemas
        
    Returns:
        List of potential relationships with confidence scores
    """
    # This would be a more sophisticated implementation using column name matching,
    # data type compatibility, and potentially even data sampling
    
    # For now, we'll implement a simplified version similar to detect_implicit_relationships
    explicit_relations = await detect_foreign_key_relationships(tables)
    implicit_relations = await detect_implicit_relationships(tables, explicit_relations)
    
    # Filter to only include implicit relations with lower confidence
    potential_relations = [
        rel for rel in implicit_relations
        if rel.get('confidence', 1.0) < 0.8
    ]
    
    return potential_relations