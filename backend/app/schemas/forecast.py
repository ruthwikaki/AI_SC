# backend/app/schemas/forecast.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ForecastMethod(str, Enum):
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    HOLT_WINTERS = "holt_winters"
    ARIMA = "arima"
    SARIMA = "sarima"
    PROPHET = "prophet"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"

class TimeFrame(str, Enum):
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_QUARTER = "last_quarter"
    LAST_YEAR = "last_year"
    YEAR_TO_DATE = "year_to_date"
    CUSTOM = "custom"

class ForecastRequest(BaseModel):
    method: ForecastMethod
    time_frame: TimeFrame
    forecast_periods: int = Field(default=12, ge=1, le=36)
    period_type: str = Field(default="month", pattern="^(day|week|month|quarter|year)$")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    filters: Optional[Dict[str, Any]] = None
    include_insights: bool = True
    include_anomaly_detection: bool = True

class ForecastPoint(BaseModel):
    period: str
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    confidence: Optional[float] = None

class ProductForecast(BaseModel):
    product_id: str
    product_name: str
    category: str
    current_demand: float
    forecast_next_month: float
    trend: float
    confidence: float
    mape: float
    abc_class: Optional[str] = None
    forecast_data: List[ForecastPoint]
    historical_data: List[ForecastPoint]

class ForecastResponse(BaseModel):
    request_parameters: ForecastRequest
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    generated_at: datetime

class ReferenceDataResponse(BaseModel):
    data: List[Dict[str, Any]]
    count: int
    
class WarehouseInfo(BaseModel):
    id: str
    name: str
    location: str
    region: Optional[str] = None
    
class RegionInfo(BaseModel):
    id: str
    name: str
    country: Optional[str] = None
    timezone: Optional[str] = None
