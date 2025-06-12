from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import random

from app.db.database import get_db
from app.api.middleware.auth import get_current_user
from app.models.user import User
from app.utils.logger import logger

router = APIRouter(prefix="/api/forecasting", tags=["forecasting"])

@router.post("/generate")
async def generate_forecast(
    forecast_params: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate forecast based on parameters"""
    try:
        product_id = forecast_params.get("product_id")
        periods = forecast_params.get("periods", 12)
        method = forecast_params.get("method", "moving_average")
        
        # Generate mock forecast data
        dates = [(datetime.now() + timedelta(days=30*i)).strftime("%Y-%m-%d") for i in range(periods)]
        values = [random.randint(80, 120) for _ in range(periods)]
        
        return {
            "forecast_id": f"fc-{int(datetime.now().timestamp())}",
            "product_id": product_id,
            "method": method,
            "forecast_data": {
                "dates": dates,
                "values": values,
                "confidence_lower": [v - 10 for v in values],
                "confidence_upper": [v + 10 for v in values]
            },
            "metrics": {
                "mae": 5.2,
                "rmse": 7.1,
                "mape": 4.3
            }
        }
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/{product_id}")
async def get_forecast_data(
    product_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get forecast data for a specific product"""
    try:
        # Generate mock data
        forecast_dates = [(datetime.now() + timedelta(days=30*i)).strftime("%Y-%m-%d") for i in range(12)]
        historical_dates = [(datetime.now() - timedelta(days=30*i)).strftime("%Y-%m-%d") for i in range(12, 0, -1)]
        
        return {
            "productId": product_id,
            "forecast": {
                "dates": forecast_dates,
                "values": [random.randint(80, 120) for _ in range(12)],
                "confidence_lower": [random.randint(70, 90) for _ in range(12)],
                "confidence_upper": [random.randint(110, 130) for _ in range(12)]
            },
            "historicalData": {
                "dates": historical_dates,
                "values": [random.randint(75, 115) for _ in range(12)]
            }
        }
    except Exception as e:
        logger.error(f"Error fetching forecast data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical/{product_id}")
async def get_historical_data(
    product_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get historical data for a product"""
    return {
        "productId": product_id,
        "data": [
            {"date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"), "value": random.randint(75, 115)}
            for i in range(30, 0, -1)
        ]
    }

@router.get("/accuracy/{forecast_id}")
async def get_forecast_accuracy(
    forecast_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get forecast accuracy metrics"""
    return {
        "forecast_id": forecast_id,
        "mae": round(random.uniform(4.0, 8.0), 2),
        "rmse": round(random.uniform(6.0, 10.0), 2),
        "mape": round(random.uniform(3.0, 6.0), 2)
    }

@router.get("/models")
async def get_available_models(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get available forecasting models"""
    return {
        "models": [
            {"id": "moving_average", "name": "Moving Average", "description": "Simple moving average"},
            {"id": "exponential_smoothing", "name": "Exponential Smoothing", "description": "Weighted moving average"},
            {"id": "arima", "name": "ARIMA", "description": "Autoregressive Integrated Moving Average"},
            {"id": "prophet", "name": "Prophet", "description": "Facebook Prophet model"},
            {"id": "lstm", "name": "LSTM", "description": "Long Short-Term Memory neural network"}
        ]
    }

@router.get("/history")
async def get_forecast_history(
    product_id: Optional[str] = None,
    limit: int = Query(10, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get forecast history"""
    return {
        "forecasts": [
            {
                "id": f"fc-{i}",
                "product_id": product_id or f"PROD-{i}",
                "created_at": (datetime.now() - timedelta(days=i)).isoformat(),
                "method": "arima",
                "accuracy": round(random.uniform(0.8, 0.95), 2)
            }
            for i in range(min(5, limit))
        ],
        "total": 5
    }

@router.post("/compare")
async def compare_forecasts(
    comparison_params: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Compare multiple forecast methods"""
    try:
        methods = comparison_params.get("methods", ["moving_average", "exponential_smoothing"])
        results = {}
        
        for method in methods:
            results[method] = {
                "forecast_values": [random.randint(80, 120) for _ in range(12)],
                "accuracy": round(random.uniform(0.8, 0.95), 2),
                "mae": round(random.uniform(4.0, 8.0), 2),
                "rmse": round(random.uniform(6.0, 10.0), 2)
            }
        
        return {
            "comparison_id": f"cmp-{int(datetime.now().timestamp())}",
            "results": results,
            "best_method": max(results.items(), key=lambda x: x[1]["accuracy"])[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods/{method_id}")
async def get_method_details(
    method_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get details for a specific forecasting method"""
    methods = {
        "moving_average": {
            "id": "moving_average",
            "name": "Moving Average",
            "description": "Simple moving average forecasting",
            "parameters": {
                "window_size": {"type": "integer", "default": 3, "min": 2, "max": 12}
            }
        },
        "exponential_smoothing": {
            "id": "exponential_smoothing",
            "name": "Exponential Smoothing",
            "description": "Weighted moving average with exponential decay",
            "parameters": {
                "alpha": {"type": "float", "default": 0.3, "min": 0.1, "max": 0.9}
            }
        },
        "arima": {
            "id": "arima",
            "name": "ARIMA",
            "description": "Autoregressive Integrated Moving Average",
            "parameters": {
                "p": {"type": "integer", "default": 1, "min": 0, "max": 5},
                "d": {"type": "integer", "default": 1, "min": 0, "max": 2},
                "q": {"type": "integer", "default": 1, "min": 0, "max": 5}
            }
        }
    }
    
    if method_id not in methods:
        raise HTTPException(status_code=404, detail="Method not found")
    
    return methods[method_id]

@router.get("/methods/{method_id}/parameters")
async def get_method_parameters(
    method_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get parameters for a specific method"""
    # Reuse the method details endpoint
    method = await get_method_details(method_id, db, current_user)
    return method.get("parameters", {})

@router.post("/methods/{method_id}/validate")
async def validate_method(
    method_id: str,
    params: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Validate parameters for a method"""
    return {"valid": True, "warnings": [], "method_id": method_id}

@router.post("/methods/compare")
async def compare_methods(
    comparison_request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Compare accuracy of different methods"""
    product_id = comparison_request.get("product_id")
    methods = comparison_request.get("methods", [])
    
    results = {}
    for method in methods:
        results[method] = {
            "accuracy": round(random.uniform(0.75, 0.95), 3),
            "mae": round(random.uniform(3, 8), 2),
            "rmse": round(random.uniform(5, 12), 2),
            "training_time": round(random.uniform(0.5, 2.5), 2)
        }
    
    best_method = max(results.items(), key=lambda x: x[1]["accuracy"])[0]
    
    return {
        "comparison_id": f"cmp-{int(datetime.now().timestamp())}",
        "results": results,
        "recommendation": best_method
    }

@router.put("/settings")
async def update_forecast_settings(
    settings: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update forecast settings"""
    return {
        "message": "Settings updated successfully",
        "settings": settings
    }



@router.get("/config")
async def get_forecast_config(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get forecast configuration"""
    return {
        "defaultMethod": "exponential_smoothing",
        "defaultPeriods": 12,
        "defaultConfidenceLevel": 0.95,
        "enabledMethods": ["moving_average", "exponential_smoothing", "arima", "prophet", "lstm"],
        "maxPeriods": 36,
        "minPeriods": 1,
        "autoDetectSeasonality": True,
        "outlierDetection": True,
        "dataRequirements": {
            "minHistoricalPoints": 24,
            "preferredHistoricalPoints": 36
        }
    }

@router.put("/config")
async def update_forecast_config(
    config: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update forecast configuration"""
    # In a real app, save to database
    return {"message": "Configuration updated", "config": config}

@router.get("/default-params")
async def get_default_params(
    product_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get default forecast parameters"""
    return {
        "method": "exponential_smoothing",
        "periods": 12,
        "confidence_level": 0.95,
        "include_seasonality": True,
        "include_trend": True
    }

@router.post("/config/validate")
async def validate_config(
    config: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Validate forecast configuration"""
    errors = []
    warnings = []
    
    # Basic validation
    if config.get("periods", 0) < 1:
        errors.append("Periods must be at least 1")
    if config.get("periods", 0) > 36:
        warnings.append("Forecasting beyond 36 periods may be less accurate")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

@router.get("/presets")
async def get_forecast_presets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get available forecast presets"""
    return {
        "presets": [
            {
                "id": "short_term",
                "name": "Short Term (3 months)",
                "description": "Quick forecast for immediate planning",
                "config": {"periods": 3, "method": "moving_average"}
            },
            {
                "id": "medium_term",
                "name": "Medium Term (6 months)",
                "description": "Standard forecast for quarterly planning",
                "config": {"periods": 6, "method": "exponential_smoothing"}
            },
            {
                "id": "long_term",
                "name": "Long Term (12 months)",
                "description": "Annual forecast for strategic planning",
                "config": {"periods": 12, "method": "arima"}
            }
        ]
    }

@router.post("/presets/{preset_id}/apply")
async def apply_preset(
    preset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Apply a forecast preset"""
    presets = {
        "short_term": {"periods": 3, "method": "moving_average"},
        "medium_term": {"periods": 6, "method": "exponential_smoothing"},
        "long_term": {"periods": 12, "method": "arima"}
    }
    
    if preset_id not in presets:
        raise HTTPException(status_code=404, detail="Preset not found")
    
    return {
        "message": f"Applied {preset_id} preset",
        "config": presets[preset_id]
    }


@router.get("/config")
async def get_forecast_config(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get forecast configuration"""
    return {
        "defaultMethod": "exponential_smoothing",
        "defaultPeriods": 12,
        "defaultConfidenceLevel": 0.95,
        "enabledMethods": ["moving_average", "exponential_smoothing", "arima", "prophet", "lstm"],
        "maxPeriods": 36,
        "minPeriods": 1,
        "autoDetectSeasonality": True,
        "outlierDetection": True,
        "dataRequirements": {
            "minHistoricalPoints": 24,
            "preferredHistoricalPoints": 36
        }
    }

@router.put("/config")
async def update_forecast_config(
    config: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update forecast configuration"""
    # In a real app, save to database
    return {"message": "Configuration updated", "config": config}

@router.get("/default-params")
async def get_default_params(
    product_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get default forecast parameters"""
    return {
        "method": "exponential_smoothing",
        "periods": 12,
        "confidence_level": 0.95,
        "include_seasonality": True,
        "include_trend": True
    }

@router.post("/config/validate")
async def validate_config(
    config: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Validate forecast configuration"""
    errors = []
    warnings = []
    
    # Basic validation
    if config.get("periods", 0) < 1:
        errors.append("Periods must be at least 1")
    if config.get("periods", 0) > 36:
        warnings.append("Forecasting beyond 36 periods may be less accurate")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

@router.get("/presets")
async def get_forecast_presets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get available forecast presets"""
    return {
        "presets": [
            {
                "id": "short_term",
                "name": "Short Term (3 months)",
                "description": "Quick forecast for immediate planning",
                "config": {"periods": 3, "method": "moving_average"}
            },
            {
                "id": "medium_term",
                "name": "Medium Term (6 months)",
                "description": "Standard forecast for quarterly planning",
                "config": {"periods": 6, "method": "exponential_smoothing"}
            },
            {
                "id": "long_term",
                "name": "Long Term (12 months)",
                "description": "Annual forecast for strategic planning",
                "config": {"periods": 12, "method": "arima"}
            }
        ]
    }

@router.post("/presets/{preset_id}/apply")
async def apply_preset(
    preset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Apply a forecast preset"""
    presets = {
        "short_term": {"periods": 3, "method": "moving_average"},
        "medium_term": {"periods": 6, "method": "exponential_smoothing"},
        "long_term": {"periods": 12, "method": "arima"}
    }
    
    if preset_id not in presets:
        raise HTTPException(status_code=404, detail="Preset not found")
    
    return {
        "message": f"Applied {preset_id} preset",
        "config": presets[preset_id]
    }


@router.get("/time-series")
async def get_time_series_data(
    product_id: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    periods: int = Query(24, ge=1, le=60),
    frequency: str = Query("monthly", regex="^(daily|weekly|monthly|quarterly|yearly)$"),
    metric: str = Query("quantity", regex="^(quantity|value|revenue)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get time series data from database for forecasting"""
    try:
        # Build query based on frequency
        date_trunc = {
            "daily": "day",
            "weekly": "week",
            "monthly": "month",
            "quarterly": "quarter",
            "yearly": "year"
        }.get(frequency, "month")
        
        # Base query - adjust table/column names to match your schema
        query = f"""
            SELECT 
                DATE_TRUNC('{date_trunc}', order_date) as period,
                SUM(quantity) as quantity,
                SUM(total_amount) as value
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE 1=1
        """
        
        params = {}
        if product_id:
            query += " AND oi.product_id = :product_id"
            params["product_id"] = product_id
            
        if warehouse_id:
            query += " AND o.warehouse_id = :warehouse_id"
            params["warehouse_id"] = warehouse_id
        
        query += f"""
            GROUP BY DATE_TRUNC('{date_trunc}', order_date)
            ORDER BY period DESC
            LIMIT :periods
        """
        params["periods"] = periods
        
        result = db.execute(text(query), params)
        
        data = [
            {
                "date": row[0].strftime("%Y-%m-%d"),
                "quantity": float(row[1]) if row[1] else 0,
                "value": float(row[2]) if row[2] else 0
            }
            for row in reversed(list(result))
        ]
        
        return {
            "data": data,
            "metadata": {
                "periods": len(data),
                "frequency": frequency,
                "product_id": product_id,
                "warehouse_id": warehouse_id,
                "start_date": data[0]["date"] if data else None,
                "end_date": data[-1]["date"] if data else None
            }
        }
    except Exception as e:
        logger.error(f"Error fetching time series: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
