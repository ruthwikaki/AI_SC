import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMService:
    def __init__(self):
        self.model_name = os.getenv("DEFAULT_MODEL", "tinyllama")
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    def _initialize_model(self):
        """Initialize the model on first use"""
        if self._initialized:
            return
            
        if self.model_name == "tinyllama":
            try:
                model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                print(f"Loading TinyLlama model...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
                
                self._initialized = True
                print("TinyLlama model loaded successfully")
                
            except Exception as e:
                print(f"Failed to load TinyLlama: {e}")
                print("Falling back to mock mode")
                self.model_name = "mock"
                self._initialized = True
        else:
            # Mock mode
            self._initialized = True
    
    async def generate_sql(self, query: str, schema: Dict[str, Any], include_explanation: bool = True) -> Dict[str, Any]:
        """Generate SQL from natural language query"""
        
        self._initialize_model()
        
        if self.model_name == "mock" or self.model is None:
            # Return mock response for testing
            return self._generate_mock_sql(query, schema, include_explanation)
        
        # Build prompt for TinyLlama
        prompt = self._build_prompt(query, schema)
        
        # Generate with TinyLlama
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract SQL from response
        sql = self._extract_sql_from_response(response, prompt)
        
        result = {
            "sql": sql,
            "model": "tinyllama"
        }
        
        if include_explanation:
            result["explanation"] = self._generate_explanation(query, sql)
        
        return result
    
    def _build_prompt(self, query: str, schema: Dict[str, Any]) -> str:
        """Build prompt for the model"""
        
        # Format schema information
        schema_text = "Database Schema:\n"
        for table_name, table_info in schema.items():
            schema_text += f"\nTable: {table_name}\n"
            for col in table_info.get("columns", []):
                schema_text += f"  - {col['column_name']} ({col['data_type']})\n"
        
        prompt = f"""<|system|>
You are a SQL query generator for a supply chain management system. Convert natural language queries to SQL.
Use only the tables and columns provided in the schema. Return only valid SQL.

{schema_text}
<|user|>
Convert this to SQL: {query}
<|assistant|>
SQL Query:
"""
        
        return prompt
    
    def _extract_sql_from_response(self, response: str, prompt: str) -> str:
        """Extract SQL from model response"""
        
        # Remove the prompt from response
        sql = response.replace(prompt, "").strip()
        
        # Look for SQL patterns
        if "SELECT" in sql.upper():
            # Find the SQL query
            lines = sql.split('\n')
            sql_lines = []
            in_sql = False
            
            for line in lines:
                if 'SELECT' in line.upper() or in_sql:
                    in_sql = True
                    sql_lines.append(line)
                    if ';' in line:
                        break
            
            sql = '\n'.join(sql_lines).strip()
        
        # Clean up
        sql = sql.replace("SQL Query:", "").strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        # Ensure it ends with semicolon
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
            
        elif "inventory" in query_lower and "location" in query_lower:
            location = "WH-MAIN"  # Default
            if "wh-" in query_lower:
                # Extract location
                import re
                match = re.search(r'wh-\w+', query_lower)
                if match:
                    location = match.group().upper()
            
            sql = f"""SELECT 
    p.name, p.sku, i.quantity_on_hand, i.quantity_available, i.reorder_point
FROM inventory i
JOIN products p ON i.product_id = p.id
WHERE i.location_code = '{location}'
ORDER BY i.quantity_on_hand DESC;"""
            explanation = f"This query shows inventory levels for location {location}"
            
        elif "order" in query_lower and ("recent" in query_lower or "last" in query_lower or "latest" in query_lower):
            sql = """SELECT 
    o.order_number, o.order_type, o.status, o.order_date, 
    o.total_amount, s.name as supplier_name
FROM orders o
LEFT JOIN suppliers s ON o.supplier_id = s.id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY o.order_date DESC;"""
            explanation = "This query shows recent orders from the last 30 days"
            
        elif "total" in query_lower and "inventory" in query_lower and "value" in query_lower:
            sql = """SELECT 
    SUM(i.quantity_on_hand * p.unit_cost) as total_inventory_value,
    COUNT(DISTINCT p.id) as product_count,
    COUNT(DISTINCT i.location_code) as location_count
FROM inventory i
JOIN products p ON i.product_id = p.id;"""
            explanation = "This query calculates the total value of all inventory"
            
        elif "pending" in query_lower and "order" in query_lower:
            sql = """SELECT 
    o.order_number, o.order_type, o.order_date, o.total_amount,
    s.name as supplier_name
FROM orders o
LEFT JOIN suppliers s ON o.supplier_id = s.id
WHERE o.status = 'pending'
ORDER BY o.order_date ASC;"""
            explanation = "This query shows all orders with pending status"
            
        elif "no inventory" in query_lower or "zero inventory" in query_lower or "out of stock" in query_lower:
            sql = """SELECT 
    p.name, p.sku, p.category,
    COALESCE(i.quantity_on_hand, 0) as quantity_on_hand
FROM products p
LEFT JOIN inventory i ON p.id = i.product_id
WHERE i.quantity_on_hand = 0 OR i.quantity_on_hand IS NULL
ORDER BY p.name;"""
            explanation = "This query shows products with zero or no inventory"
            
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
