# backend/app/api/routes/reference_data.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any
from app.db.database import get_db
from app.models.supply_chain import Product, Category, Supplier, Order
from app.schemas.analytics import ReferenceDataResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reference", tags=["reference_data"])

@router.get("/warehouses")
async def get_warehouses(db: Session = Depends(get_db)):
    """Get all warehouse locations from database"""
    try:
        # First try to get from warehouses table
        query = text("""
        SELECT DISTINCT 
            warehouse_id as id,
            warehouse_name as name,
            location,
            region
        FROM warehouses
        WHERE active = true
        ORDER BY warehouse_name
        """)
        result = db.execute(query).fetchall()
        
        if result:
            return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.warning(f"Warehouses table not found, trying alternative: {e}")
    
    # Fallback to inventory or orders table
    try:
        query = text("""
        SELECT DISTINCT 
            warehouse_id as id,
            COALESCE(warehouse_name, warehouse_id) as name,
            COALESCE(location, 'Unknown') as location
        FROM inventory
        WHERE warehouse_id IS NOT NULL
        ORDER BY warehouse_id
        """)
        result = db.execute(query).fetchall()
        
        if result:
            return [dict(row._mapping) for row in result]
        
        # If still no data, try orders table
        query = text("""
        SELECT DISTINCT 
            ship_from_warehouse as id,
            ship_from_warehouse as name,
            'Distribution Center' as location
        FROM orders
        WHERE ship_from_warehouse IS NOT NULL
        ORDER BY ship_from_warehouse
        """)
        result = db.execute(query).fetchall()
        return [dict(row._mapping) for row in result] if result else []
        
    except Exception as e:
        logger.error(f"Error fetching warehouses: {e}")
        return []

@router.get("/regions")
async def get_regions(db: Session = Depends(get_db)):
    """Get all sales regions from database"""
    try:
        # Try regions table first
        query = text("""
        SELECT DISTINCT 
            region_id as id,
            region_name as name,
            country,
            timezone
        FROM regions
        WHERE active = true
        ORDER BY region_name
        """)
        result = db.execute(query).fetchall()
        
        if result:
            return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.warning(f"Regions table not found, trying alternative: {e}")
    
    # Fallback to customers or orders
    try:
        query = text("""
        SELECT DISTINCT 
            COALESCE(region, shipping_region, billing_region) as id,
            COALESCE(region, shipping_region, billing_region) as name
        FROM orders
        WHERE COALESCE(region, shipping_region, billing_region) IS NOT NULL
        
        UNION
        
        SELECT DISTINCT
            region as id,
            region as name
        FROM customers
        WHERE region IS NOT NULL
        
        ORDER BY name
        """)
        result = db.execute(query).fetchall()
        return [dict(row._mapping) for row in result] if result else []
        
    except Exception as e:
        logger.error(f"Error fetching regions: {e}")
        return []

@router.get("/product-categories")
async def get_product_categories(db: Session = Depends(get_db)):
    """Get all product categories with statistics"""
    try:
        # Main query to get categories with stats
        query = text("""
        WITH category_stats AS (
            SELECT 
                COALESCE(c.category_name, p.category, 'Uncategorized') as name,
                COUNT(DISTINCT p.product_id) as product_count,
                COALESCE(SUM(i.quantity * p.unit_price), 0) as total_value,
                COALESCE(SUM(i.quantity), 0) as total_quantity
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.category_id
            LEFT JOIN inventory i ON p.product_id = i.product_id
            GROUP BY COALESCE(c.category_name, p.category, 'Uncategorized')
        ),
        total_values AS (
            SELECT SUM(total_value) as grand_total FROM category_stats
        )
        SELECT 
            cs.name,
            cs.product_count,
            cs.total_value as value,
            CASE 
                WHEN tv.grand_total > 0 
                THEN ROUND((cs.total_value / tv.grand_total * 100)::numeric, 1)
                ELSE 0 
            END as percentage
        FROM category_stats cs
        CROSS JOIN total_values tv
        WHERE cs.product_count > 0
        ORDER BY cs.total_value DESC
        """)
        
        categories = db.execute(query).fetchall()
        result = []
        
        for cat in categories:
            cat_dict = dict(cat._mapping)
            
            # Get ABC distribution for this category
            try:
                abc_query = text("""
                SELECT 
                    COALESCE(abc_classification, 'C') as abc_category,
                    COUNT(*) as count
                FROM products
                WHERE COALESCE(category, 'Uncategorized') = :category
                GROUP BY COALESCE(abc_classification, 'C')
                """)
                abc_result = db.execute(abc_query, {"category": cat_dict['name']}).fetchall()
                
                abc_dict = {'A': 0, 'B': 0, 'C': 0}
                for row in abc_result:
                    abc_dict[row.abc_category] = row.count
                
                cat_dict['abc_distribution'] = abc_dict
            except:
                cat_dict['abc_distribution'] = {'A': 0, 'B': 0, 'C': 0}
            
            result.append(cat_dict)
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching product categories: {e}")
        # Return empty list instead of error
        return []

@router.get("/forecast-methods")
async def get_available_forecast_methods():
    """Get available forecast methods from the system"""
    try:
        from app.analytics.inventory_optimization.forecast_engine import ForecastEngine
        
        methods = []
        method_info = {
            "moving_average": {
                "name": "Moving Average",
                "description": "Simple average of recent periods, best for stable demand",
                "best_for": "Stable, predictable products",
                "accuracy": "Medium",
                "complexity": "Low"
            },
            "exponential_smoothing": {
                "name": "Exponential Smoothing",
                "description": "Weights recent data more heavily, good for trends",
                "best_for": "Products with mild trends",
                "accuracy": "Medium-High",
                "complexity": "Low"
            },
            "holt_winters": {
                "name": "Holt-Winters",
                "description": "Captures both trend and seasonality",
                "best_for": "Seasonal products with trends",
                "accuracy": "High",
                "complexity": "Medium"
            },
            "arima": {
                "name": "ARIMA",
                "description": "Advanced time series model for complex patterns",
                "best_for": "Complex demand patterns",
                "accuracy": "High",
                "complexity": "High"
            },
            "sarima": {
                "name": "SARIMA",
                "description": "ARIMA with seasonal components",
                "best_for": "Highly seasonal products",
                "accuracy": "Very High",
                "complexity": "High"
            },
            "prophet": {
                "name": "Prophet",
                "description": "Facebook's algorithm for holiday effects and changepoints",
                "best_for": "Products affected by holidays/events",
                "accuracy": "High",
                "complexity": "Medium"
            },
            "lstm": {
                "name": "LSTM Neural Network",
                "description": "Deep learning for complex non-linear patterns",
                "best_for": "High-variability items",
                "accuracy": "Very High",
                "complexity": "Very High"
            },
            "ensemble": {
                "name": "Ensemble",
                "description": "Combines multiple models for best accuracy",
                "best_for": "Critical high-value products",
                "accuracy": "Highest",
                "complexity": "Very High"
            }
        }
        
        # Get methods from ForecastEngine
        for method_id in ForecastEngine.METHODS:
            if method_id in method_info:
                methods.append({
                    "id": method_id,
                    **method_info[method_id]
                })
        
        return methods
        
    except Exception as e:
        logger.error(f"Error getting forecast methods: {e}")
        # Return default methods if ForecastEngine not available
        return [
            {
                "id": "exponential_smoothing",
                "name": "Exponential Smoothing",
                "description": "Standard forecasting method",
                "best_for": "General use",
                "accuracy": "Medium",
                "complexity": "Low"
            }
        ]

@router.get("/suppliers")
async def get_suppliers(db: Session = Depends(get_db)):
    """Get all active suppliers"""
    try:
        query = text("""
        SELECT 
            supplier_id as id,
            supplier_name as name,
            COALESCE(country, region, 'Unknown') as location,
            COALESCE(rating, 0) as rating
        FROM suppliers
        WHERE active = true
        ORDER BY supplier_name
        """)
        result = db.execute(query).fetchall()
        return [dict(row._mapping) for row in result] if result else []
    except Exception as e:
        logger.error(f"Error fetching suppliers: {e}")
        return []

@router.get("/time-frames")
async def get_time_frames():
    """Get available time frame options"""
    return [
        {"id": "last_week", "name": "Last Week", "days": 7},
        {"id": "last_month", "name": "Last Month", "days": 30},
        {"id": "last_quarter", "name": "Last Quarter", "days": 90},
        {"id": "last_year", "name": "Last Year", "days": 365},
        {"id": "year_to_date", "name": "Year to Date", "days": None},
        {"id": "custom", "name": "Custom Range", "days": None}
    ]

@router.get("/abc-classes")
async def get_abc_classes():
    """Get ABC classification definitions"""
    return [
        {
            "class": "A",
            "name": "High Value",
            "description": "Top 20% of products by value",
            "color": "green",
            "threshold": 0.8
        },
        {
            "class": "B",
            "name": "Medium Value",
            "description": "Next 30% of products by value",
            "color": "yellow",
            "threshold": 0.5
        },
        {
            "class": "C",
            "name": "Low Value",
            "description": "Bottom 50% of products by value",
            "color": "red",
            "threshold": 0.0
        }
    ]
