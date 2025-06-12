import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import httpx

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.model_name = os.getenv("DEFAULT_MODEL", "ollama")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("LLM_MODEL", "mistral")
        self._initialized = False
        
    async def generate_sql(self, query: str, schema: Dict[str, Any], include_explanation: bool = True) -> Dict[str, Any]:
        """Generate SQL from natural language query"""
        
        try:
            # Try Ollama first
            if self.model_name == "ollama":
                return await self._generate_with_ollama(query, schema, include_explanation)
            else:
                # Fallback to mock
                return self._generate_mock_sql(query, schema, include_explanation)
                
        except Exception as e:
            logger.error(f"Error in LLM service: {e}")
            # Fallback to mock on any error
            return self._generate_mock_sql(query, schema, include_explanation)
    
    async def _generate_with_ollama(self, query: str, schema: Dict[str, Any], include_explanation: bool) -> Dict[str, Any]:
        """Generate SQL using Ollama"""
        
        prompt = self._build_prompt(query, schema)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.1
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    sql = self._extract_sql_from_response(result.get("response", ""))
                    
                    return {
                        "sql": sql,
                        "model": f"ollama/{self.ollama_model}",
                        "explanation": self._generate_explanation(query, sql) if include_explanation else None
                    }
                else:
                    raise Exception(f"Ollama API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    def _build_prompt(self, query: str, schema: Dict[str, Any]) -> str:
        """Build prompt for the model"""
        
        # Format schema information
        schema_text = "Database Schema:\n"
        for table_name, table_info in schema.items():
            schema_text += f"\nTable: {table_name}\n"
            for col in table_info.get("columns", []):
                schema_text += f"  - {col['column_name']} ({col['data_type']})\n"
        
        prompt = f"""You are a SQL query generator for a supply chain management system.
Convert the natural language query to SQL.

{schema_text}

User Query: {query}

Generate a PostgreSQL query to answer this question. Return only the SQL query.
"""
        
        return prompt
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL from model response"""
        
        # Clean response
        sql = response.strip()
        
        # Remove markdown code blocks if present
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0]
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0]
        
        # Ensure it ends with semicolon
        sql = sql.strip()
        if sql and not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def _generate_explanation(self, query: str, sql: str) -> str:
        """Generate explanation for the SQL query"""
        
        if "low inventory" in query.lower():
            return "This query finds products where the quantity on hand is below the reorder point"
        elif "supplier" in query.lower() and "rating" in query.lower():
            return "This query filters suppliers based on their rating"
        elif "order" in query.lower():
            return "This query retrieves order information based on the specified criteria"
        else:
            return "This query retrieves data based on your natural language request"
    
    def _generate_mock_sql(self, query: str, schema: Dict[str, Any], include_explanation: bool) -> Dict[str, Any]:
        """Generate mock SQL for testing"""
        
        query_lower = query.lower()
        
        # Generate SQL based on query patterns
        if "low inventory" in query_lower or "below reorder" in query_lower:
            sql = """SELECT 
    p.name, p.sku, i.quantity_on_hand, i.reorder_point,
    (i.reorder_point - i.quantity_on_hand) as shortage
FROM products p
JOIN inventory i ON p.id = i.product_id
WHERE i.quantity_on_hand < i.reorder_point
ORDER BY shortage DESC;"""
            explanation = "This query shows all products where inventory is below reorder point"
            
        elif "supplier" in query_lower and ("rating" in query_lower or "best" in query_lower or "top" in query_lower):
            sql = """SELECT 
    s.name, s.code, s.rating, s.city, s.country
FROM suppliers s
WHERE s.rating > 4.0
ORDER BY s.rating DESC
LIMIT 10;"""
            explanation = "This query shows suppliers with high ratings"
            
        elif "total" in query_lower and "inventory" in query_lower and "value" in query_lower:
            sql = """SELECT 
    SUM(i.quantity_on_hand * p.unit_cost) as total_inventory_value,
    COUNT(DISTINCT p.id) as product_count,
    COUNT(DISTINCT i.location_code) as location_count
FROM inventory i
JOIN products p ON i.product_id = p.id;"""
            explanation = "This query calculates the total value of all inventory"
            
        else:
            # Generic query
            sql = """SELECT * FROM products LIMIT 10;"""
            explanation = "This is a generic query. Please be more specific about what you want to see."
        
        result = {
            "sql": sql,
            "model": "mock-llm"
        }
        
        if include_explanation:
            result["explanation"] = explanation
        
        return result
