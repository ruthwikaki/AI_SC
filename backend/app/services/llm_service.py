import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx
import re
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

class LLMService:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("LLM_MODEL", "deepseek-coder-v2:16b-lite-instruct-q4_0")
        self.schema_cache = {}
        self.schema_json = {}  # Store structured schema
        self.table_embeddings = {}  # Store embeddings for RAG
        
    def discover_schema(self, db: Session) -> str:
        """Discover database schema dynamically - with hybrid approach from main.py"""
        cache_key = str(db.bind.url)
        if cache_key in self.schema_cache:
            return self.schema_cache[cache_key]
            
        inspector = inspect(db.bind)
        schema_parts = ["DATABASE SCHEMA:\n"]
        self.schema_json = {"tables": {}}
        
        # Get all tables with details
        for table_name in inspector.get_table_names():
            if table_name.startswith('pg_'):
                continue
                
            # Table info
            schema_parts.append(f"\nTABLE: {table_name}")
            table_info = {"columns": {}, "foreign_keys": [], "sample_data": {}}
            
            # Columns
            columns = inspector.get_columns(table_name)
            schema_parts.append("Columns:")
            for col in columns:
                schema_parts.append(f"  - {col['name']}: {col['type']}")
                table_info["columns"][col['name']] = str(col['type'])
            
            # Foreign keys - CRUCIAL for joins
            fks = inspector.get_foreign_keys(table_name)
            if fks:
                schema_parts.append("Foreign Keys:")
                for fk in fks:
                    schema_parts.append(f"  - {fk['constrained_columns'][0]} -> {fk['referred_table']}.{fk['referred_columns'][0]}")
                    table_info["foreign_keys"].append({
                        "column": fk['constrained_columns'][0],
                        "references": f"{fk['referred_table']}.{fk['referred_columns'][0]}"
                    })
            
            # Sample data to understand content
            try:
                result = db.execute(text(f"SELECT * FROM {table_name} LIMIT 3"))
                rows = result.fetchall()
                if rows:
                    schema_parts.append("Sample rows:")
                    for idx, row in enumerate(rows[:2]):
                        for i, col in enumerate(columns[:3]):  # First 3 columns
                            schema_parts.append(f"  - {col['name']}: {row[i]}")
                            if idx == 0:
                                table_info["sample_data"][col['name']] = str(row[i])
            except:
                pass
            
            self.schema_json["tables"][table_name] = table_info
        
        # Key relationships summary
        schema_parts.append("\n\nKEY RELATIONSHIPS:")
        schema_parts.append("- Products and Suppliers are connected through: products -> order_items -> orders -> suppliers")
        schema_parts.append("- Products and Inventory: products.id = inventory.product_id")
        schema_parts.append("- Orders and Order Items: orders.id = order_items.order_id")
        
        schema_info = "\n".join(schema_parts)
        self.schema_cache[cache_key] = schema_info
        print(f"✅ Schema discovered: {len(inspector.get_table_names())} tables")
        
        # Create embeddings for RAG
        self._create_schema_embeddings()
        
        return schema_info
    
    def _create_schema_embeddings(self):
        """Create embeddings for each table (simplified version)"""
        for table_name, table_info in self.schema_json["tables"].items():
            # Create a text representation for embedding
            text_repr = f"Table {table_name} contains columns: {', '.join(table_info['columns'].keys())}"
            if table_info.get('foreign_keys'):
                text_repr += f". Joins with: {', '.join([fk['references'].split('.')[0] for fk in table_info['foreign_keys']])}"
            
            # In production, use actual embedding model
            # For now, store keywords for simple matching
            self.table_embeddings[table_name] = {
                "text": text_repr,
                "keywords": set(table_name.lower().split('_') + 
                              [col.lower() for col in table_info['columns'].keys()])
            }
    
    def _retrieve_relevant_schema(self, question: str, top_k: int = 5) -> str:
        """Retrieve most relevant schema parts for the question"""
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        # Score each table based on keyword overlap
        table_scores = []
        for table_name, embedding_info in self.table_embeddings.items():
            score = len(question_words.intersection(embedding_info["keywords"]))
            # Boost score for exact table name match
            if table_name.lower() in question_lower:
                score += 5
            table_scores.append((table_name, score))
        
        # Sort by score and get top tables
        table_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_tables = [t[0] for t in table_scores[:top_k] if t[1] > 0]
        
        # Always include core relationship tables if joining is likely needed
        join_keywords = ['supplier', 'product', 'order', 'inventory', 'with', 'by', 'per', 'each']
        if any(keyword in question_lower for keyword in join_keywords):
            core_tables = ['products', 'orders', 'order_items', 'suppliers', 'inventory']
            for table in core_tables:
                if table in self.schema_json["tables"] and table not in relevant_tables:
                    relevant_tables.append(table)
        
        # Build focused schema
        schema_parts = ["RELEVANT SCHEMA:\n"]
        relationships = []
        
        for table_name in relevant_tables:
            if table_name in self.schema_json["tables"]:
                table_info = self.schema_json["tables"][table_name]
                schema_parts.append(f"\nTABLE: {table_name}")
                schema_parts.append("Columns:")
                for col_name, col_type in table_info["columns"].items():
                    schema_parts.append(f"  - {col_name}: {col_type}")
                
                if table_info.get("foreign_keys"):
                    schema_parts.append("Foreign Keys:")
                    for fk in table_info["foreign_keys"]:
                        fk_str = f"  - {fk['column']} -> {fk['references']}"
                        schema_parts.append(fk_str)
                        relationships.append(fk_str)
        
        # Add relationship summary if multiple tables
        if len(relevant_tables) > 1 and relationships:
            schema_parts.append("\nRELEVANT RELATIONSHIPS:")
            for rel in set(relationships):
                schema_parts.append(rel)
        
        return "\n".join(schema_parts)
    
    async def generate_sql(self, query: str, schema: Dict[str, Any] = None, db_session: Session = None, include_explanation: bool = True) -> Dict[str, Any]:
        """Generate SQL from natural language query with hybrid approach"""
        
        # If db_session provided, discover schema
        if db_session:
            # Full schema discovery
            full_schema = self.discover_schema(db_session)
            # Get relevant schema parts (RAG)
            schema_text = self._retrieve_relevant_schema(query)
        elif schema:
            schema_text = self._format_schema_dict(schema)
        else:
            return {
                "sql": "",
                "model": self.model,
                "error": "No schema information provided"
            }
        
        # Check if visualization is requested
        viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'diagram', 'show me', 'display']
        needs_viz = any(word in query.lower() for word in viz_keywords)
        
        # Build enhanced prompt with both full context and focused schema
        prompt = f"""You are an expert SQL query generator. You have been fine-tuned on this specific database schema.

{schema_text}

IMPORTANT CONTEXT - Full Database Overview:
- Main entities: products, suppliers, orders, order_items, inventory, users
- Key relationships: 
  * products -> order_items -> orders -> suppliers (for product-supplier queries)
  * products.id = inventory.product_id (for stock queries)
  * orders.id = order_items.order_id (for order details)

User Question: {query}

Generate a PostgreSQL query to answer this question. STRICT RULES:
1. Use ONLY the tables and columns shown in the RELEVANT SCHEMA above
2. For joins, follow the foreign key relationships exactly
3. Table and column names are case-sensitive - use lowercase
4. For "top N" queries, use ORDER BY with LIMIT
5. For totals/counts, use appropriate aggregate functions (SUM, COUNT, AVG)
6. For grouping, always include GROUP BY for non-aggregate columns
7. Use proper JOIN syntax based on the relationships shown

Common patterns for this database:
- To link products with suppliers: JOIN through order_items and orders
- To get inventory info: JOIN products with inventory on product_id
- To get order details: JOIN orders with order_items
- For date ranges: use order_date, created_at, or updated_at columns

{f'''
Also determine if this needs a visualization. Consider these factors:
- Comparisons, distributions, or trends usually need visualization
- Top N lists benefit from bar charts
- Status distributions work well as pie charts
- Time-based data suits line charts

If visualization needed, specify:
- chart_type: bar, pie, line, scatter
- x_column: which column for x-axis (usually the grouping column)
- y_column: which column for y-axis (usually the aggregate column)
''' if needs_viz else ''}

Respond in EXACTLY this format:
SQL: <your query here>
{f'NEEDS_VIZ: true/false' if needs_viz else ''}
{f'CHART_TYPE: <type>' if needs_viz else ''}
{f'X_COLUMN: <exact column name or alias>' if needs_viz else ''}
{f'Y_COLUMN: <exact column name or alias>' if needs_viz else ''}

Example good response:
SQL: SELECT s.name, COUNT(o.id) as order_count FROM suppliers s JOIN orders o ON s.id = o.supplier_id GROUP BY s.name ORDER BY order_count DESC LIMIT 10;
NEEDS_VIZ: true
CHART_TYPE: bar
X_COLUMN: name
Y_COLUMN: order_count
"""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.1,
                        "num_predict": 1000,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                )
                
                if response.status_code == 200:
                    ai_response = response.json().get("response", "")
                    
                    # Extract SQL with better parsing
                    sql_match = re.search(r'SQL:\s*(.+?)(?:NEEDS_VIZ:|$)', ai_response, re.DOTALL | re.IGNORECASE)
                    if sql_match:
                        sql = sql_match.group(1).strip()
                        
                        # Clean SQL
                        sql = sql.replace('```sql', '').replace('```', '').strip()
                        sql = re.sub(r'\s+', ' ', sql)
                        if not sql.endswith(';'):
                            sql += ';'
                        
                        # Validate SQL has required tables
                        sql_lower = sql.lower()
                        has_valid_table = any(table in sql_lower for table in self.schema_json["tables"].keys())
                        
                        if not has_valid_table:
                            return {
                                "success": False, 
                                "error": "Generated SQL doesn't reference valid tables",
                                "sql": sql
                            }
                        
                        result = {"sql": sql, "success": True, "model": self.model}
                        
                        # Extract explanation if requested
                        if include_explanation:
                            result["explanation"] = "Query generated successfully using hybrid AI approach"
                        
                        # Check for visualization
                        if needs_viz:
                            needs_viz_response = 'NEEDS_VIZ: true' in ai_response or 'needs_viz: true' in ai_response.lower()
                            if needs_viz_response:
                                chart_match = re.search(r'CHART_TYPE:\s*(\w+)', ai_response, re.IGNORECASE)
                                x_match = re.search(r'X_COLUMN:\s*([\w_]+)', ai_response, re.IGNORECASE)
                                y_match = re.search(r'Y_COLUMN:\s*([\w_]+)', ai_response, re.IGNORECASE)
                                
                                if chart_match:
                                    result['visualization'] = {
                                        'needed': True,
                                        'chart_type': chart_match.group(1).lower(),
                                        'x_column': x_match.group(1) if x_match else None,
                                        'y_column': y_match.group(1) if y_match else None
                                    }
                        
                        return result
                    else:
                        return {
                            "success": False, 
                            "error": "Could not extract SQL from response",
                            "raw_response": ai_response[:200]
                        }
                        
        except httpx.ConnectError:
            return {
                "sql": "",
                "model": self.model,
                "error": "Failed to connect to Ollama. Make sure it's running with the model loaded.",
                "success": False
            }
        except Exception as e:
            print(f"LLM Error: {e}")
            return {
                "sql": "",
                "model": self.model,
                "error": f"Error generating SQL: {str(e)}",
                "success": False
            }
    
    def _format_schema_dict(self, schema: Dict[str, Any]) -> str:
        """Format schema dictionary to text"""
        schema_text = "Database Schema:\n"
        for table_name, table_info in schema.items():
            schema_text += f"\nTable: {table_name}\n"
            for col in table_info.get("columns", []):
                schema_text += f"  - {col['column_name']} ({col['data_type']})\n"
        return schema_text
    
    async def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent"""
        query_lower = query.lower()
        
        # Detect primary intent
        if any(word in query_lower for word in ['inventory', 'stock', 'quantity']):
            primary_intent = "inventory_analysis"
        elif any(word in query_lower for word in ['supplier', 'vendor']):
            primary_intent = "supplier_analysis"
        elif any(word in query_lower for word in ['order', 'purchase']):
            primary_intent = "order_analysis"
        else:
            primary_intent = "general_query"
        
        # Detect visualization need
        viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'show', 'display']
        suggested_viz = "bar" if any(word in query_lower for word in viz_keywords) else None
        
        return {
            "query": query,
            "primary_intent": primary_intent,
            "suggested_visualization": suggested_viz,
            "complexity": "simple"
        }
    
    async def suggest_queries(self, db_session: Session = None, context: str = None) -> List[str]:
        """Generate query suggestions"""
        base_suggestions = [
            "which suppliers provide products in electronics category",
            "show total inventory value",
            "create a pie chart of order status distribution",
            "find products below reorder point",
            "show supplier performance metrics",
            "analyze order trends by month",
            "which products are most profitable",
            "show suppliers with highest ratings",
            "visualize inventory levels by category",
            "calculate average order processing time"
        ]
        
        return base_suggestions