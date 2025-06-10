# backend/app/api/routes/reference_data.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct
from datetime import datetime, timedelta
from uuid import UUID

from app.db.database import get_db
from app.api.middleware.auth import get_current_user
from app.models.user import User
from app.models.supply_chain import Product, Supplier, Warehouse, Order, Inventory
from app.models.extended_models import ForecastModel
from app.utils.logger import logger

router = APIRouter(prefix="/reference", tags=["reference-data"])

@router.get("/forecast-methods")
async def get_forecast_methods(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get available forecast methods with their configurations"""
    try:
        methods = db.query(ForecastModel).filter(
            ForecastModel.is_active == True
        ).all()
        
        if not methods:
            # Return default methods if none in database
            return [
                {
                    "id": "arima",
                    "name": "ARIMA",
                    "display_name": "Auto-Regressive Integrated Moving Average",
                    "description": "Statistical model for time series forecasting",
                    "accuracy_score": 0.85,
                    "best_for": ["seasonal_data", "trending_data"],
                    "parameters": {
                        "p": {"default": 1, "min": 0, "max": 5},
                        "d": {"default": 1, "min": 0, "max": 2},
                        "q": {"default": 1, "min": 0, "max": 5}
                    }
                },
                {
                    "id": "exponential_smoothing",
                    "name": "Exponential Smoothing",
                    "display_name": "Holt-Winters Exponential Smoothing",
                    "description": "Weighted average forecasting with trend and seasonality",
                    "accuracy_score": 0.82,
                    "best_for": ["stable_demand", "short_term"],
                    "parameters": {
                        "alpha": {"default": 0.3, "min": 0, "max": 1},
                        "beta": {"default": 0.1, "min": 0, "max": 1},
                        "gamma": {"default": 0.1, "min": 0, "max": 1}
                    }
                },
                {
                    "id": "prophet",
                    "name": "Prophet",
                    "display_name": "Facebook Prophet",
                    "description": "Robust forecasting for data with strong seasonal patterns",
                    "accuracy_score": 0.88,
                    "best_for": ["multiple_seasonality", "holidays", "missing_data"],
                    "parameters": {
                        "changepoint_prior_scale": {"default": 0.05, "min": 0.001, "max": 0.5},
                        "seasonality_prior_scale": {"default": 10, "min": 0.01, "max": 10}
                    }
                },
                {
                    "id": "lstm",
                    "name": "LSTM",
                    "display_name": "Long Short-Term Memory Neural Network",
                    "description": "Deep learning model for complex patterns",
                    "accuracy_score": 0.90,
                    "best_for": ["complex_patterns", "large_datasets", "non_linear"],
                    "parameters": {
                        "epochs": {"default": 50, "min": 10, "max": 200},
                        "batch_size": {"default": 32, "min": 16, "max": 128},
                        "hidden_units": {"default": 50, "min": 10, "max": 200}
                    }
                },
                {
                    "id": "ensemble",
                    "name": "Ensemble",
                    "display_name": "Ensemble Method",
                    "description": "Combines multiple models for better accuracy",
                    "accuracy_score": 0.92,
                    "best_for": ["high_accuracy", "critical_forecasts"],
                    "parameters": {
                        "models": {"default": ["arima", "prophet", "lstm"], "options": ["arima", "prophet", "lstm", "exponential_smoothing"]},
                        "weights": {"default": "auto", "options": ["auto", "equal", "custom"]}
                    }
                }
            ]
        
        return [
            {
                "id": str(method.id),
                "name": method.name,
                "display_name": method.display_name,
                "description": method.description,
                "accuracy_score": method.average_accuracy,
                "best_for": method.best_use_cases.split(',') if method.best_use_cases else [],
                "parameters": method.default_parameters
            }
            for method in methods
        ]
    except Exception as e:
        logger.error(f"Error fetching forecast methods: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/warehouses")
async def get_warehouses(
    include_stats: bool = Query(False, description="Include warehouse statistics"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get list of warehouses with optional statistics"""
    try:
        warehouses = db.query(Warehouse).filter(
            Warehouse.is_active == True
        ).all()
        
        result = []
        for warehouse in warehouses:
            warehouse_data = {
                "id": str(warehouse.id),
                "name": warehouse.name,
                "code": warehouse.code,
                "location": warehouse.location,
                "capacity": warehouse.capacity,
                "type": warehouse.warehouse_type,
                "region": warehouse.region
            }
            
            if include_stats:
                # Get current inventory stats
                inventory_stats = db.query(
                    func.count(distinct(Inventory.product_id)).label('product_count'),
                    func.sum(Inventory.quantity).label('total_quantity'),
                    func.sum(Inventory.quantity * Inventory.unit_cost).label('total_value')
                ).filter(
                    Inventory.warehouse_id == warehouse.id
                ).first()
                
                warehouse_data['stats'] = {
                    'product_count': inventory_stats.product_count or 0,
                    'total_quantity': float(inventory_stats.total_quantity or 0),
                    'total_value': float(inventory_stats.total_value or 0),
                    'utilization': (float(inventory_stats.total_quantity or 0) / warehouse.capacity * 100) if warehouse.capacity > 0 else 0
                }
            
            result.append(warehouse_data)
        
        return result
    except Exception as e:
        logger.error(f"Error fetching warehouses: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/regions")
async def get_regions(
    include_metrics: bool = Query(False, description="Include regional metrics"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get list of regions with optional metrics"""
    try:
        # Get unique regions from warehouses
        regions = db.query(distinct(Warehouse.region)).filter(
            Warehouse.region.isnot(None)
        ).all()
        
        result = []
        for (region,) in regions:
            region_data = {
                "id": region.lower().replace(' ', '_'),
                "name": region,
                "display_name": region.title()
            }
            
            if include_metrics:
                # Get warehouses in region
                warehouse_count = db.query(func.count(Warehouse.id)).filter(
                    Warehouse.region == region
                ).scalar()
                
                # Get order metrics for region
                order_metrics = db.query(
                    func.count(Order.id).label('order_count'),
                    func.sum(Order.total_amount).label('total_revenue')
                ).join(
                    Warehouse, Order.warehouse_id == Warehouse.id
                ).filter(
                    Warehouse.region == region,
                    Order.created_at >= datetime.utcnow() - timedelta(days=30)
                ).first()
                
                region_data['metrics'] = {
                    'warehouse_count': warehouse_count,
                    'monthly_orders': order_metrics.order_count or 0,
                    'monthly_revenue': float(order_metrics.total_revenue or 0)
                }
            
            result.append(region_data)
        
        # Sort by name
        result.sort(key=lambda x: x['name'])
        
        return result
    except Exception as e:
        logger.error(f"Error fetching regions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/product-categories")
async def get_product_categories(
    include_stats: bool = Query(True, description="Include category statistics"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get product categories with statistics"""
    try:
        # Get unique categories
        categories = db.query(distinct(Product.category)).filter(
            Product.category.isnot(None)
        ).all()
        
        result = []
        for (category,) in categories:
            category_data = {
                "id": category.lower().replace(' ', '_'),
                "name": category,
                "display_name": category.title()
            }
            
            if include_stats:
                # Get product count
                product_count = db.query(func.count(Product.id)).filter(
                    Product.category == category,
                    Product.is_active == True
                ).scalar()
                
                # Get inventory stats
                inventory_stats = db.query(
                    func.sum(Inventory.quantity).label('total_stock'),
                    func.sum(Inventory.quantity * Inventory.unit_cost).label('total_value'),
                    func.avg(Inventory.quantity).label('avg_stock_per_warehouse')
                ).join(
                    Product, Inventory.product_id == Product.id
                ).filter(
                    Product.category == category
                ).first()
                
                # Get order stats for last 30 days
                order_stats = db.query(
                    func.count(distinct(Order.id)).label('order_count'),
                    func.sum(Order.quantity).label('total_ordered')
                ).join(
                    Product, Order.product_id == Product.id
                ).filter(
                    Product.category == category,
                    Order.created_at >= datetime.utcnow() - timedelta(days=30)
                ).first()
                
                category_data['stats'] = {
                    'product_count': product_count,
                    'total_stock': float(inventory_stats.total_stock or 0),
                    'total_value': float(inventory_stats.total_value or 0),
                    'avg_stock_per_warehouse': float(inventory_stats.avg_stock_per_warehouse or 0),
                    'monthly_orders': order_stats.order_count or 0,
                    'monthly_quantity_ordered': float(order_stats.total_ordered or 0)
                }
            
            result.append(category_data)
        
        # Sort by total value descending
        if include_stats:
            result.sort(key=lambda x: x['stats']['total_value'], reverse=True)
        else:
            result.sort(key=lambda x: x['name'])
        
        return result
    except Exception as e:
        logger.error(f"Error fetching product categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/suppliers")
async def get_suppliers_list(
    category: Optional[str] = Query(None, description="Filter by product category"),
    region: Optional[str] = Query(None, description="Filter by region"),
    include_metrics: bool = Query(False, description="Include supplier metrics"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get suppliers list with optional filters and metrics"""
    try:
        query = db.query(Supplier).filter(Supplier.status == 'active')
        
        # Apply filters
        if category:
            # Filter suppliers that supply products in this category
            query = query.join(Product, Supplier.products).filter(
                Product.category == category
            ).distinct()
        
        if region:
            query = query.filter(Supplier.region == region)
        
        suppliers = query.all()
        
        result = []
        for supplier in suppliers:
            supplier_data = {
                "id": str(supplier.id),
                "name": supplier.name,
                "code": supplier.code,
                "contact_person": supplier.contact_person,
                "email": supplier.email,
                "phone": supplier.phone,
                "region": supplier.region,
                "rating": supplier.rating
            }
            
            if include_metrics:
                # Get supplier metrics
                order_metrics = db.query(
                    func.count(Order.id).label('total_orders'),
                    func.avg(Order.delivery_time).label('avg_delivery_time'),
                    func.sum(Order.total_amount).label('total_business')
                ).filter(
                    Order.supplier_id == supplier.id,
                    Order.created_at >= datetime.utcnow() - timedelta(days=90)
                ).first()
                
                supplier_data['metrics'] = {
                    'total_orders': order_metrics.total_orders or 0,
                    'avg_delivery_time': float(order_metrics.avg_delivery_time or 0),
                    'total_business': float(order_metrics.total_business or 0),
                    'product_count': len(supplier.products)
                }
            
            result.append(supplier_data)
        
        # Sort by rating descending
        result.sort(key=lambda x: x['rating'], reverse=True)
        
        return result
    except Exception as e:
        logger.error(f"Error fetching suppliers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/currencies")
async def get_currencies(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get supported currencies"""
    return [
        {"code": "USD", "name": "US Dollar", "symbol": "$"},
        {"code": "EUR", "name": "Euro", "symbol": "€"},
        {"code": "GBP", "name": "British Pound", "symbol": "£"},
        {"code": "JPY", "name": "Japanese Yen", "symbol": "¥"},
        {"code": "CNY", "name": "Chinese Yuan", "symbol": "¥"},
        {"code": "INR", "name": "Indian Rupee", "symbol": "₹"},
        {"code": "CAD", "name": "Canadian Dollar", "symbol": "$"},
        {"code": "AUD", "name": "Australian Dollar", "symbol": "$"}
    ]

@router.get("/time-zones")
async def get_time_zones(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get supported time zones"""
    return [
        {"value": "UTC", "label": "UTC", "offset": "+00:00"},
        {"value": "America/New_York", "label": "Eastern Time", "offset": "-05:00"},
        {"value": "America/Chicago", "label": "Central Time", "offset": "-06:00"},
        {"value": "America/Denver", "label": "Mountain Time", "offset": "-07:00"},
        {"value": "America/Los_Angeles", "label": "Pacific Time", "offset": "-08:00"},
        {"value": "Europe/London", "label": "London", "offset": "+00:00"},
        {"value": "Europe/Paris", "label": "Paris", "offset": "+01:00"},
        {"value": "Asia/Tokyo", "label": "Tokyo", "offset": "+09:00"},
        {"value": "Asia/Shanghai", "label": "Shanghai", "offset": "+08:00"},
        {"value": "Asia/Kolkata", "label": "India", "offset": "+05:30"},
        {"value": "Australia/Sydney", "label": "Sydney", "offset": "+11:00"}
    ]