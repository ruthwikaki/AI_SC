"""
Application entry point with LLM support and SQL execution.
"""

import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
import jwt
import json
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import time

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import config
try:
    from config import get_settings
    settings = get_settings()
    print("Configuration loaded successfully")
except Exception as e:
    print(f"WARNING: Config error: {str(e)}")
    class FallbackSettings:
        app_name = "Supply Chain LLM API"
        api_version = "1.0.0"
        environment = "development"
        host = "0.0.0.0"
        port = 8000
        cors_origins = ["*"]
        llm_api_key = None
        default_model = "mock-llm"
        database_url = "postgresql://postgres:123456789@localhost:5432/Supplychain_AI"
    settings = FallbackSettings()

# Database setup
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Simple logger
def log_info(message):
    print(f"[INFO] {datetime.now().isoformat()} - {message}")

def log_error(message):
    print(f"[ERROR] {datetime.now().isoformat()} - {message}")

# Test user for development
TEST_USER = {
    "email": "test@example.com",
    "username": "testuser",
    "password": "testpassword",
    "role": "user"
}

# Mock LLM for testing without API keys
class MockLLM:
    """Simple mock LLM for testing"""
    
    def __init__(self):
        self.name = "mock-llm"
        
    async def generate(self, prompt: str = None, context: Dict[str, Any] = None, **kwargs):
        """Generate mock response based on query"""
        query = context.get("query", prompt) if context else prompt
        query_lower = query.lower() if query else ""
        
        # Generate contextual SQL responses
        if "low inventory" in query_lower or "below reorder" in query_lower:
            sql = """SELECT 
    p.name as product_name, p.sku, i.quantity_on_hand, i.reorder_point,
    (i.reorder_point - i.quantity_on_hand) as shortage
FROM products p
JOIN inventory i ON p.id = i.product_id
WHERE i.quantity_on_hand < i.reorder_point
ORDER BY shortage DESC
LIMIT 20;"""
            explanation = "This query shows all products with inventory below reorder point"
            
        elif "supplier" in query_lower and ("rating" in query_lower or "best" in query_lower):
            sql = """SELECT 
    s.name as supplier_name, s.code, s.rating, s.city, s.country
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
            
        elif "pending" in query_lower and "order" in query_lower:
            sql = """SELECT 
    o.order_number, o.order_type, o.order_date, o.total_amount,
    s.name as supplier_name
FROM orders o
LEFT JOIN suppliers s ON o.supplier_id = s.id
WHERE o.status = 'pending'
ORDER BY o.order_date ASC
LIMIT 20;"""
            explanation = "This query shows all orders with pending status"
            
        else:
            sql = """SELECT * FROM products LIMIT 10;"""
            explanation = "Please try a more specific query about inventory, suppliers, or orders"
        
        return {
            "sql": sql,
            "explanation": explanation,
            "model_name": self.name
        }
    
    async def health_check(self):
        return {"is_healthy": True, "model_name": self.name}

# Global LLM instance
llm_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager"""
    global llm_instance
    
    # Startup
    log_info(f"Starting {settings.app_name}")
    log_info(f"Environment: {settings.environment}")
    
    # Initialize LLM
    llm_instance = MockLLM()
    log_info("Using mock LLM for SQL generation")
    
    # Test database connection
    try:
        db = SessionLocal()
        result = db.execute(text("SELECT 1"))
        log_info("Database connection successful")
        db.close()
    except Exception as e:
        log_error(f"Database connection failed: {e}")
    
    log_info("Application startup complete")
    
    yield
    
    # Shutdown
    log_info("Shutting down application")

# Create the main application
app = FastAPI(
    title=settings.app_name,
    description="API for Supply Chain LLM with Natural Language Query",
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route
@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": settings.api_version,
        "environment": settings.environment,
        "llm_status": "active" if llm_instance else "not initialized",
        "llm_model": llm_instance.name if llm_instance else None,
        "docs": "/docs"
    }

# Health check
@app.get("/health")
async def health_check():
    llm_health = await llm_instance.health_check() if llm_instance else {"is_healthy": False}
    
    # Test database
    db_health = {"is_healthy": False}
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db_health = {"is_healthy": True}
        db.close()
    except:
        pass
    
    return {
        "status": "healthy",
        "version": settings.api_version,
        "llm": llm_health,
        "database": db_health,
        "timestamp": datetime.now().isoformat()
    }

# Authentication endpoints
@app.post("/api/auth/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username == TEST_USER["email"] and form_data.password == TEST_USER["password"]:
        token = jwt.encode(
            {
                "sub": form_data.username,
                "user_id": "test-user-id",
                "role": TEST_USER["role"],
                "exp": datetime.utcnow() + timedelta(hours=24)
            },
            "secret-key",
            algorithm="HS256"
        )
        return {"access_token": token, "token_type": "bearer"}
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/auth/me")
async def get_current_user():
    return {
        "id": "test-user-id",
        "username": TEST_USER["username"],
        "email": TEST_USER["email"],
        "role": TEST_USER["role"]
    }

# Helper function to execute SQL
def execute_sql_query(db_session, sql: str) -> tuple[List[Dict[str, Any]], float, int]:
    """Execute SQL query and return results with execution time"""
    start_time = time.time()
    
    try:
        result = db_session.execute(text(sql))
        
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
                    elif isinstance(value, (int, float, str, bool)) or value is None:
                        row_dict[col] = value
                    else:
                        row_dict[col] = str(value)
                results.append(row_dict)
            
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            return results, execution_time, len(results)
        else:
            # For non-SELECT queries
            db_session.commit()
            affected_rows = result.rowcount
            execution_time = (time.time() - start_time) * 1000
            return [{"affected_rows": affected_rows}], execution_time, affected_rows
            
    except Exception as e:
        db_session.rollback()
        raise e

# Natural Language Query Endpoint with SQL Execution
@app.post("/api/queries/natural-language")
async def process_natural_language_query(request: dict):
    """Process natural language query using LLM and execute SQL"""
    if not llm_instance:
        return {
            "error": "LLM not initialized",
            "message": "Please configure LLM settings"
        }
    
    query = request.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    db = SessionLocal()
    try:
        # Generate SQL using LLM
        context = {
            "query": query,
            "schema": "products, inventory, suppliers, orders"
        }
        
        llm_result = await llm_instance.generate(
            prompt=query,
            context=context
        )
        
        sql = llm_result.get("sql", "")
        explanation = llm_result.get("explanation", "")
        model_name = llm_result.get("model_name", "mock-llm")
        
        # Execute the SQL
        try:
            results, exec_time, row_count = execute_sql_query(db, sql)
            
            return {
                "query": query,
                "sql": sql,
                "results": results,
                "row_count": row_count,
                "execution_time_ms": exec_time,
                "explanation": explanation,
                "model": model_name,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as sql_error:
            log_error(f"SQL execution error: {str(sql_error)}")
            return {
                "query": query,
                "sql": sql,
                "results": [],
                "row_count": 0,
                "execution_time_ms": 0,
                "explanation": explanation,
                "model": model_name,
                "success": False,
                "error": f"SQL execution failed: {str(sql_error)}",
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        log_error(f"Error processing query: {str(e)}")
        return {
            "error": "Query processing failed",
            "message": str(e),
            "query": query,
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
    finally:
        db.close()

@app.post("/api/queries/execute")
async def execute_query(request: dict):
    """Execute a query (wraps natural language endpoint)"""
    return await process_natural_language_query(request)

@app.get("/api/queries/suggestions")
async def get_query_suggestions():
    """Get query suggestions"""
    return {
        "suggestions": [
            "Show me products with low inventory",
            "Which suppliers have rating above 4?",
            "What products are below reorder point?",
            "Show all pending orders",
            "What is the total inventory value?",
            "List top 10 suppliers by rating",
            "Show inventory levels for location WH-MAIN",
            "Which products have zero inventory?"
        ]
    }

@app.get("/api/database/test")
async def test_database():
    """Test database connection and show statistics"""
    db = SessionLocal()
    try:
        stats = {}
        
        # Get counts from main tables
        tables = ['products', 'suppliers', 'inventory', 'orders', 'natural_language_queries']
        for table in tables:
            try:
                result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                stats[table] = result.scalar()
            except:
                stats[table] = 0
        
        return {
            "status": "connected",
            "database": "Supplychain_AI",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        db.close()

if __name__ == "__main__":
    port = settings.port
    log_info(f"Starting {settings.app_name}")
    log_info(f"API Server: http://localhost:{port}")
    log_info(f"API Documentation: http://localhost:{port}/docs")
    log_info(f"LLM Model: {settings.default_model}")
    log_info(f"Database: {settings.database_url}")
    log_info("Test User: test@example.com / testpassword")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=port,
        reload=settings.environment == "development",
        log_level="info"
    )
