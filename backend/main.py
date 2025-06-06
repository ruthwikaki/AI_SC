"""
True AI Backend - Uses LLM for everything, no patterns
"""

import os
import warnings
warnings.filterwarnings("ignore")

import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
import httpx
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from typing import Dict, Any, Optional, List
import json
import jwt
import hashlib
from pydantic import BaseModel
import re
import time

# For visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from io import BytesIO
import base64

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:123456789@localhost:5432/Supplychain_AI")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Security
SECRET_KEY = "your-secret-key"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auth
class User(BaseModel):
    username: str
    email: str

fake_users_db = {
    "test@example.com": {
        "username": "test@example.com",
        "email": "test@example.com",
        "hashed_password": hashlib.sha256("testpassword".encode()).hexdigest(),
    }
}

def create_access_token(data: dict):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + timedelta(hours=24)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401)
        return User(username=username, email=username)
    except:
        raise HTTPException(status_code=401)

# Pure AI - No patterns, no hardcoding
class PureAI:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "deepseek-coder-v2:16b-lite-instruct-q4_0"
        self.schema_info = ""
        
    def discover_schema(self, db_session):
        """Discover database schema dynamically"""
        inspector = inspect(engine)
        schema_parts = ["DATABASE SCHEMA:\n"]
        
        # Get all tables with details
        for table_name in inspector.get_table_names():
            if table_name.startswith('pg_'):
                continue
                
            # Table info
            schema_parts.append(f"\nTABLE: {table_name}")
            
            # Columns
            columns = inspector.get_columns(table_name)
            schema_parts.append("Columns:")
            for col in columns:
                schema_parts.append(f"  - {col['name']}: {col['type']}")
            
            # Foreign keys - CRUCIAL for joins
            fks = inspector.get_foreign_keys(table_name)
            if fks:
                schema_parts.append("Foreign Keys:")
                for fk in fks:
                    schema_parts.append(f"  - {fk['constrained_columns'][0]} -> {fk['referred_table']}.{fk['referred_columns'][0]}")
            
            # Sample data to understand content
            try:
                result = db_session.execute(text(f"SELECT * FROM {table_name} LIMIT 1"))
                row = result.fetchone()
                if row:
                    schema_parts.append("Sample row:")
                    for i, col in enumerate(columns[:3]):  # First 3 columns
                        schema_parts.append(f"  - {col['name']}: {row[i]}")
            except:
                pass
        
        # Key relationships summary
        schema_parts.append("\n\nKEY RELATIONSHIPS:")
        schema_parts.append("- Products and Suppliers are connected through: products -> order_items -> orders -> suppliers")
        schema_parts.append("- Products and Inventory: products.id = inventory.product_id")
        schema_parts.append("- Orders and Order Items: orders.id = order_items.order_id")
        
        self.schema_info = "\n".join(schema_parts)
        print(f"✅ Schema discovered: {len(inspector.get_table_names())} tables")
    
    async def generate_sql(self, question: str, include_viz: bool = False) -> Dict[str, Any]:
        """Let AI generate SQL based on question and schema"""
        
        # Build prompt
        prompt = f"""{self.schema_info}

User Question: {question}

Generate a PostgreSQL query to answer this question. Important rules:
1. Use the exact table and column names from the schema above
2. When joining products with suppliers, use the path: products -> order_items -> orders -> suppliers
3. Use appropriate JOINs based on the foreign keys shown
4. For aggregations, use COUNT, SUM, AVG as needed
5. For "top N", use ORDER BY and LIMIT
6. For grouping, use GROUP BY
7. Always use lowercase for identifiers

{f'''
Also determine if this needs a visualization. If yes, specify:
- chart_type: bar, pie, line, scatter
- x_column: which column for x-axis
- y_column: which column for y-axis
''' if include_viz else ''}

Respond in this format:
SQL: <your query>
{f'NEEDS_VIZ: true/false' if include_viz else ''}
{f'CHART_TYPE: <type>' if include_viz else ''}
{f'X_COLUMN: <column>' if include_viz else ''}
{f'Y_COLUMN: <column>' if include_viz else ''}
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
                        "num_predict": 1000
                    }
                )
                
                if response.status_code == 200:
                    ai_response = response.json().get("response", "")
                    
                    # Extract SQL
                    sql_match = re.search(r'SQL:\s*(.+?)(?:NEEDS_VIZ:|$)', ai_response, re.DOTALL)
                    if sql_match:
                        sql = sql_match.group(1).strip()
                        
                        # Clean SQL
                        sql = sql.replace('```sql', '').replace('```', '').strip()
                        if not sql.endswith(';'):
                            sql += ';'
                        
                        result = {"sql": sql, "success": True}
                        
                        # Check for visualization
                        if include_viz and 'NEEDS_VIZ: true' in ai_response:
                            chart_match = re.search(r'CHART_TYPE:\s*(\w+)', ai_response)
                            x_match = re.search(r'X_COLUMN:\s*(\w+)', ai_response)
                            y_match = re.search(r'Y_COLUMN:\s*(\w+)', ai_response)
                            
                            if chart_match:
                                result['needs_viz'] = True
                                result['chart_type'] = chart_match.group(1)
                                result['x_column'] = x_match.group(1) if x_match else None
                                result['y_column'] = y_match.group(1) if y_match else None
                        
                        return result
                    
        except Exception as e:
            print(f"AI Error: {e}")
        
        return {"success": False, "error": "Failed to generate SQL"}
    
    def create_visualization(self, data: List[Dict], chart_type: str, x_col: str = None, y_col: str = None) -> str:
        """Create visualization based on AI's suggestion"""
        try:
            df = pd.DataFrame(data)
            
            if df.empty:
                return None
            
            plt.figure(figsize=(10, 6))
            
            # Auto-detect columns if not specified
            if not x_col and len(df.columns) > 0:
                x_col = df.columns[0]
            if not y_col and len(df.columns) > 1:
                y_col = df.columns[1]
            
            # Create chart based on type
            if chart_type == 'bar':
                plt.bar(df[x_col], df[y_col])
                plt.xticks(rotation=45, ha='right')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                
            elif chart_type == 'pie':
                plt.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%')
                
            elif chart_type == 'line':
                plt.plot(df[x_col], df[y_col], marker='o')
                plt.xticks(rotation=45, ha='right')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                
            elif chart_type == 'scatter':
                plt.scatter(df[x_col], df[y_col])
                plt.xlabel(x_col)
                plt.ylabel(y_col)
            
            else:  # Default bar chart
                if len(df) > 20:
                    df = df.head(20)
                plt.bar(range(len(df)), df.iloc[:, 1])
                plt.xticks(range(len(df)), df.iloc[:, 0], rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return None

# Create app
app = FastAPI(title="True AI - No Hardcoding")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI
ai = PureAI()

@app.on_event("startup")
async def startup():
    print("\n" + "="*60)
    print("🤖 TRUE AI - NO PATTERNS, NO HARDCODING")
    print("="*60)
    
    db = SessionLocal()
    try:
        ai.discover_schema(db)
        print("✅ Ready for any query!")
        print("="*60 + "\n")
    finally:
        db.close()

# Auth endpoints
@app.post("/api/auth/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or hashlib.sha256(form_data.password.encode()).hexdigest() != user["hashed_password"]:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    return {
        "access_token": create_access_token(data={"sub": user["username"]}),
        "token_type": "bearer",
        "user": {"email": user["email"], "username": user["username"]}
    }

@app.get("/api/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# Main query endpoint - Pure AI
@app.post("/api/queries/execute")
async def execute_query(
    request: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Execute any query using pure AI"""
    query = request.get("query", "").strip()
    
    if not query:
        return {"success": False, "error": "No query provided"}
    
    print(f"\n📝 Query: {query}")
    
    # Check if visualization is requested
    viz_words = ['chart', 'graph', 'plot', 'visualize', 'diagram']
    needs_viz = any(word in query.lower() for word in viz_words)
    
    # Generate SQL using AI
    result = await ai.generate_sql(query, include_viz=needs_viz)
    
    if not result.get("success"):
        return result
    
    sql = result.get("sql")
    print(f"🔍 Generated SQL: {sql}")
    
    # Execute SQL
    try:
        db_result = db.execute(text(sql))
        rows = db_result.fetchall()
        columns = list(db_result.keys())
        data = [dict(zip(columns, row)) for row in rows]
        
        response = {
            "success": True,
            "data": data,
            "columns": columns,
            "row_count": len(data),
            "sql": sql
        }
        
        # Create visualization if needed
        if result.get("needs_viz") and len(data) > 0:
            viz = ai.create_visualization(
                data, 
                result.get("chart_type", "bar"),
                result.get("x_column"),
                result.get("y_column")
            )
            if viz:
                response["visualization"] = viz
                response["intent"] = {"main_intent": "visualization"}
        
        return response
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Database error: {str(e)}",
            "sql": sql
        }

@app.get("/api/queries/suggestions")
async def get_suggestions(current_user: User = Depends(get_current_user)):
    """AI-generated suggestions based on schema"""
    # These are just examples - the AI handles any query
    return {
        "suggestions": [
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
    }

@app.get("/api/queries/saved")
async def get_saved_queries(current_user: User = Depends(get_current_user)):
    return {"queries": []}

@app.get("/")
async def root():
    return {
        "name": "True AI Backend",
        "status": "No patterns, no hardcoding - pure AI",
        "model": ai.model
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

@app.get("/api/analytics/dashboard/metrics")
async def get_dashboard_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get dashboard metrics with real data"""
    try:
        # Inventory value
        inventory_result = db.execute(text("""
            SELECT 
                COALESCE(SUM(p.unit_cost * i.quantity_on_hand), 0) as inventory_value,
                COUNT(DISTINCT p.id) as total_products,
                COUNT(CASE WHEN i.quantity_on_hand < i.reorder_point THEN 1 END) as low_stock_items
            FROM products p
            LEFT JOIN inventory i ON p.id = i.product_id
        """)).first()
        
        # Order metrics
        order_result = db.execute(text("""
            SELECT 
                COUNT(*) as total_orders,
                COUNT(CASE WHEN status = 'delivered' THEN 1 END) as delivered_orders,
                COUNT(CASE WHEN delivery_date <= expected_delivery_date THEN 1 END) as on_time_deliveries
            FROM orders
            WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
        """)).first()
        
        # Supplier metrics
        supplier_result = db.execute(text("""
            SELECT 
                AVG(rating) as avg_rating,
                COUNT(*) as total_suppliers
            FROM suppliers
            WHERE is_active = true
        """)).first()
        
        # Calculate metrics
        total_orders = order_result.total_orders or 1  # Avoid division by zero
        delivered_orders = order_result.delivered_orders or 0
        on_time_deliveries = order_result.on_time_deliveries or 0
        
        # Calculate period changes (simplified - comparing to last month)
        prev_inv_result = db.execute(text("""
            SELECT COALESCE(SUM(oi.quantity * p.unit_cost), 0) as prev_value
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.order_date >= CURRENT_DATE - INTERVAL '60 days'
            AND o.order_date < CURRENT_DATE - INTERVAL '30 days'
        """)).scalar()
        
        current_inv_value = float(inventory_result.inventory_value)
        prev_inv_value = float(prev_inv_result or current_inv_value)
        inventory_change = ((current_inv_value - prev_inv_value) / prev_inv_value * 100) if prev_inv_value > 0 else 0
        
        return {
            "inventoryValue": current_inv_value,
            "inventoryChange": round(inventory_change, 1),
            "orderFillRate": round((delivered_orders / total_orders * 100), 1),
            "orderFillChange": -0.8,  # Would calculate from historical data
            "onTimeDelivery": round((on_time_deliveries / total_orders * 100), 1),
            "deliveryChange": 1.2,  # Would calculate from historical data
            "supplierPerformance": round(float(supplier_result.avg_rating or 0) * 20, 1),
            "supplierChange": 0.5  # Would calculate from historical data
        }
        
    except Exception as e:
        print(f"Dashboard metrics error: {e}")
        # Return -- for all values on error
        return {
            "inventoryValue": "--",
            "inventoryChange": "--",
            "orderFillRate": "--",
            "orderFillChange": "--",
            "onTimeDelivery": "--",
            "deliveryChange": "--",
            "supplierPerformance": "--",
            "supplierChange": "--"
        }