# backend/app/api/routes/analytics_enhanced.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, case, distinct
from datetime import datetime, timedelta
import json

from app.db.database import get_db
from app.api.routes.auth import get_current_active_user
from app.models.user import User
from app.models.supply_chain import Product, Order, Inventory, Supplier, Warehouse
from app.models.analytics import AnalyticsMetric
from app.analytics.inventory_optimization.forecast_engine import ForecastEngine
from app.analytics.inventory_optimization.abc_analysis import ABCAnalyzer
from app.analytics.logistics_analytics.delivery_analytics import DeliveryAnalytics
from app.analytics.supplier_performance.scorecard import SupplierScorecard
from app.cache.result_cache import ResultCache
from app.utils.logger import get_logger


# Initialize logger
logger = get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics-enhanced"])

# Initialize cache
cache = ResultCache()

@router.post("/inventory/forecast")
async def run_inventory_forecast(
    request_body: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Run inventory forecast with specified parameters"""
    try:
        params = request_body.get('request_parameters', {})
        
        # Extract parameters
        method = params.get('method', 'auto')
        time_frame = params.get('time_frame', 'last_quarter')
        forecast_periods = params.get('forecast_periods', 12)
        period_type = params.get('period_type', 'month')
        confidence_level = params.get('confidence_level', 0.95)
        filters = params.get('filters', {})
        
        # Initialize forecast engine
        forecast_engine = ForecastEngine(db)
        
        # Get historical data based on filters
        query = db.query(Order).join(Product)
        
        # Apply filters
        if filters.get('product_category'):
            query = query.filter(Product.category == filters['product_category'])
        if filters.get('warehouse_id'):
            query = query.filter(Order.warehouse_id == filters['warehouse_id'])
        if filters.get('region'):
            query = query.join(Warehouse).filter(Warehouse.region == filters['region'])
        if filters.get('supplier_id'):
            query = query.filter(Order.supplier_id == filters['supplier_id'])
        
        # Apply time frame
        start_date = _get_start_date(time_frame)
        orders = query.filter(Order.created_at >= start_date).all()
        
        # Run forecast
        forecast_results = forecast_engine.forecast(
            historical_data=orders,
            method=method,
            periods=forecast_periods,
            confidence_level=confidence_level
        )
        
        # Calculate insights
        insights = _generate_forecast_insights(forecast_results, orders, db)
        
        return {
            "status": "success",
            "forecast": forecast_results,
            "insights": insights,
            "parameters": params,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/inventory/forecast")
async def get_forecast_data(
    product_id: Optional[int] = Query(None),
    warehouse_id: Optional[int] = Query(None),
    limit: int = Query(50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get existing forecast data"""
    try:
        # Check cache first
        cache_key = f"forecast_{product_id}_{warehouse_id}_{current_user.id}"
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # Query forecast data from analytics metrics
        query = db.query(AnalyticsMetric).filter(
            AnalyticsMetric.metric_type == 'forecast'
        )
        
        if product_id:
            query = query.filter(AnalyticsMetric.entity_id == product_id)
        if warehouse_id:
            query = query.filter(AnalyticsMetric.warehouse_id == warehouse_id)
        
        forecasts = query.order_by(AnalyticsMetric.created_at.desc()).limit(limit).all()
        
        result = {
            "forecasts": [
                {
                    "id": f.id,
                    "product_id": f.entity_id,
                    "warehouse_id": f.warehouse_id,
                    "forecast_date": f.metric_date,
                    "predicted_value": f.value,
                    "confidence_interval": json.loads(f.metadata).get('confidence_interval') if f.metadata else None,
                    "method": json.loads(f.metadata).get('method') if f.metadata else None,
                    "created_at": f.created_at
                }
                for f in forecasts
            ],
            "count": len(forecasts)
        }
        
        # Cache result
        cache.set(cache_key, result, ttl=300)  # 5 minutes
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching forecast data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast/performance")
async def get_forecast_performance(
    days: int = Query(30, description="Number of days to analyze"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get historical forecast performance metrics"""
    try:
        # Get forecasts and actuals for comparison
        start_date = datetime.utcnow() - timedelta(days=days)
        
        performance_data = db.query(
            AnalyticsMetric.entity_id,
            func.avg(
                case(
                    (AnalyticsMetric.metric_type == 'forecast_accuracy', AnalyticsMetric.value),
                    else_=None
                )
            ).label('accuracy'),
            func.count(AnalyticsMetric.id).label('forecast_count')
        ).filter(
            AnalyticsMetric.metric_type.in_(['forecast', 'forecast_accuracy']),
            AnalyticsMetric.created_at >= start_date
        ).group_by(AnalyticsMetric.entity_id).all()
        
        return {
            "performance_metrics": [
                {
                    "product_id": p.entity_id,
                    "average_accuracy": float(p.accuracy) if p.accuracy else None,
                    "forecast_count": p.forecast_count
                }
                for p in performance_data
            ],
            "period_days": days,
            "overall_accuracy": sum(p.accuracy for p in performance_data if p.accuracy) / len(performance_data) if performance_data else 0
        }
        
    except Exception as e:
        logger.error(f"Error fetching forecast performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/products/{product_id}/forecast")
async def get_product_forecast(
    product_id: int,
    periods: int = Query(12, description="Number of periods to forecast"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get forecast for specific product"""
    try:
        # Verify product exists
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Get historical data
        orders = db.query(Order).filter(
            Order.product_id == product_id,
            Order.created_at >= datetime.utcnow() - timedelta(days=365)
        ).all()
        
        if not orders:
            return {
                "product_id": product_id,
                "product_name": product.name,
                "forecast": [],
                "message": "Insufficient historical data for forecasting"
            }
        
        # Run forecast
        forecast_engine = ForecastEngine(db)
        forecast = forecast_engine.forecast_single_product(
            product_id=product_id,
            historical_orders=orders,
            periods=periods
        )
        
        return {
            "product_id": product_id,
            "product_name": product.name,
            "forecast": forecast,
            "historical_periods": len(orders),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching product forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast/batch")
async def batch_product_forecast(
    request_body: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Run batch forecast for multiple products"""
    try:
        product_ids = request_body.get('product_ids', [])
        periods = request_body.get('periods', 12)
        method = request_body.get('method', 'auto')
        
        if not product_ids:
            raise HTTPException(status_code=400, detail="Product IDs required")
        
        # Verify products exist
        products = db.query(Product).filter(Product.id.in_(product_ids)).all()
        if len(products) != len(product_ids):
            raise HTTPException(status_code=404, detail="Some products not found")
        
        forecast_engine = ForecastEngine(db)
        results = []
        
        for product in products:
            # Get historical data
            orders = db.query(Order).filter(
                Order.product_id == product.id,
                Order.created_at >= datetime.utcnow() - timedelta(days=365)
            ).all()
            
            if orders:
                forecast = forecast_engine.forecast_single_product(
                    product_id=product.id,
                    historical_orders=orders,
                    periods=periods,
                    method=method
                )
                
                results.append({
                    "product_id": product.id,
                    "product_name": product.name,
                    "forecast": forecast,
                    "status": "success"
                })
            else:
                results.append({
                    "product_id": product.id,
                    "product_name": product.name,
                    "forecast": [],
                    "status": "insufficient_data"
                })
        
        return {
            "batch_results": results,
            "total_products": len(results),
            "successful": len([r for r in results if r['status'] == 'success']),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running batch forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/kpi/realtime")
async def get_realtime_kpis(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get real-time KPI metrics"""
    try:
        # Calculate various KPIs
        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)
        last_week = today - timedelta(days=7)
        last_month = today - timedelta(days=30)
        
        # Order metrics
        order_metrics = db.query(
            func.count(Order.id).label('total_orders'),
            func.sum(Order.total_amount).label('total_revenue'),
            func.avg(Order.total_amount).label('avg_order_value')
        ).filter(
            func.date(Order.created_at) == today
        ).first()
        
        yesterday_orders = db.query(func.count(Order.id)).filter(
            func.date(Order.created_at) == yesterday
        ).scalar()
        
        # Inventory metrics
        inventory_metrics = db.query(
            func.count(distinct(Inventory.product_id)).label('products_in_stock'),
            func.sum(Inventory.quantity * Inventory.unit_cost).label('total_inventory_value')
        ).filter(
            Inventory.quantity > 0
        ).first()
        
        low_stock_count = db.query(func.count(Inventory.id)).filter(
            Inventory.quantity <= Inventory.reorder_point
        ).scalar()
        
        # Supplier metrics
        active_suppliers = db.query(func.count(Supplier.id)).filter(
            Supplier.status == 'active'
        ).scalar()
        
        # Delivery metrics
        pending_deliveries = db.query(func.count(Order.id)).filter(
            Order.status == 'pending'
        ).scalar()
        
        on_time_deliveries = db.query(func.count(Order.id)).filter(
            Order.status == 'delivered',
            Order.actual_delivery_date <= Order.expected_delivery_date,
            Order.created_at >= last_week
        ).scalar()
        
        total_deliveries = db.query(func.count(Order.id)).filter(
            Order.status == 'delivered',
            Order.created_at >= last_week
        ).scalar()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "kpis": {
                "orders": {
                    "today_count": order_metrics.total_orders or 0,
                    "today_revenue": float(order_metrics.total_revenue or 0),
                    "avg_order_value": float(order_metrics.avg_order_value or 0),
                    "change_from_yesterday": ((order_metrics.total_orders or 0) - yesterday_orders) / yesterday_orders * 100 if yesterday_orders else 0
                },
                "inventory": {
                    "total_value": float(inventory_metrics.total_inventory_value or 0),
                    "products_in_stock": inventory_metrics.products_in_stock or 0,
                    "low_stock_alerts": low_stock_count,
                    "stockout_risk": low_stock_count / (inventory_metrics.products_in_stock or 1) * 100
                },
                "suppliers": {
                    "active_count": active_suppliers,
                    "pending_orders": pending_deliveries
                },
                "delivery": {
                    "pending_count": pending_deliveries,
                    "on_time_rate": (on_time_deliveries / total_deliveries * 100) if total_deliveries else 0,
                    "weekly_deliveries": total_deliveries
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching real-time KPIs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export")
async def export_analytics_data(
    export_request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Export analytics data in various formats"""
    try:
        data = export_request.get('data', {})
        format = export_request.get('format', 'csv')
        export_type = export_request.get('type', 'general')
        
        # This would integrate with the export service
        # For now, return a success response
        return {
            "status": "success",
            "message": f"Export initiated for {export_type} in {format} format",
            "export_id": f"exp_{datetime.utcnow().timestamp()}"
        }
        
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/preferences")
async def get_dashboard_preferences(
    preference_type: str = Query(..., description="Type of preference to fetch"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get user's dashboard preferences"""
    try:
        # This would fetch from user preferences table
        # For now, return default preferences
        if preference_type == 'forecast_config':
            return {
                "preferences": {
                    "default_method": "auto",
                    "default_periods": 12,
                    "default_confidence": 0.95,
                    "auto_refresh": False,
                    "refresh_interval": 3600
                }
            }
        
        return {"preferences": {}}
        
    except Exception as e:
        logger.error(f"Error fetching preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dashboard/preferences")
async def save_dashboard_preferences(
    preference_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Save user's dashboard preferences"""
    try:
        preference_type = preference_data.get('preference_type')
        preferences = preference_data.get('preferences', {})
        
        # Save to database (implementation would go here)
        
        return {
            "status": "success",
            "message": f"Preferences saved for {preference_type}",
            "saved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error saving preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _get_start_date(time_frame: str) -> datetime:
    """Get start date based on time frame"""
    now = datetime.utcnow()
    
    if time_frame == 'last_week':
        return now - timedelta(days=7)
    elif time_frame == 'last_month':
        return now - timedelta(days=30)
    elif time_frame == 'last_quarter':
        return now - timedelta(days=90)
    elif time_frame == 'last_year':
        return now - timedelta(days=365)
    elif time_frame == 'year_to_date':
        return datetime(now.year, 1, 1)
    else:
        return now - timedelta(days=90)  # Default to last quarter

def _generate_forecast_insights(forecast_results: Dict, historical_orders: List, db: Session) -> List[Dict]:
    """Generate insights from forecast results"""
    insights = []
    
    # Trend analysis
    if forecast_results.get('trend'):
        trend = forecast_results['trend']
        if trend > 0.1:
            insights.append({
                "type": "trend",
                "severity": "info",
                "message": f"Demand is trending upward by {trend:.1%} per period",
                "recommendation": "Consider increasing safety stock levels"
            })
        elif trend < -0.1:
            insights.append({
                "type": "trend",
                "severity": "warning",
                "message": f"Demand is trending downward by {abs(trend):.1%} per period",
                "recommendation": "Review inventory levels to avoid excess stock"
            })
    
    # Seasonality detection
    if forecast_results.get('seasonality_detected'):
        insights.append({
            "type": "seasonality",
            "severity": "info",
            "message": "Seasonal patterns detected in demand",
            "recommendation": "Adjust procurement schedule to match seasonal variations"
        })
    
    # Volatility analysis
    if forecast_results.get('volatility', 0) > 0.3:
        insights.append({
            "type": "volatility",
            "severity": "warning",
            "message": "High demand volatility detected",
            "recommendation": "Increase safety stock or implement more frequent ordering"
        })
    
    return insights