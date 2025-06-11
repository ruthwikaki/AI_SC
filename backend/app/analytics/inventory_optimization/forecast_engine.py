"""
Forecast Engine Module

This module provides demand forecasting functionality for inventory planning,
supporting various forecasting methods and parameters.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
import math
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.supply_chain import Order, Product, Inventory
from app.models.extended_models import ExtendedAnalyticsMetric as AnalyticsMetric
from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class ForecastEngine:
    """
    Engine for generating demand forecasts using various methods.
    """
    
    # Available forecasting methods
    METHODS = {
        "moving_average": "Simple Moving Average",
        "weighted_moving_average": "Weighted Moving Average",
        "exponential_smoothing": "Exponential Smoothing",
        "holt_winters": "Holt-Winters Exponential Smoothing",
        "arima": "ARIMA (AutoRegressive Integrated Moving Average)",
        "sarima": "SARIMA (Seasonal ARIMA)",
        "linear_regression": "Linear Regression",
        "automatic": "Automatic Method Selection",
        "auto": "Automatic Method Selection"
    }
    
    def __init__(self, method: str = "automatic", db: Optional[Session] = None):
        """
        Initialize the forecast engine.
        
        Args:
            method: Forecasting method to use (default: automatic selection)
            db: Optional database session for storing results
        """
        if method not in self.METHODS:
            logger.warning(f"Unknown forecasting method: {method}. Falling back to automatic selection.")
            method = "automatic"
        
        self.method = method
        self.db = db
    
    def forecast(
        self,
        historical_data: Union[List[Order], List[float], List[Dict[str, Any]]],
        method: str = 'auto',
        periods: int = 12,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Generate forecast based on historical data"""
        
        # Handle Order objects
        if historical_data and isinstance(historical_data[0], Order):
            if not historical_data:
                return {
                    "status": "error",
                    "message": "No historical data available",
                    "forecast": []
                }
            
            # Prepare time series data from Orders
            time_series = self._prepare_time_series_from_orders(historical_data)
        else:
            # Handle regular list data
            time_series = historical_data if isinstance(historical_data, list) else []
        
        if not time_series:
            return {
                "status": "error",
                "message": "No time series data available",
                "forecast": []
            }
        
        # Select forecasting method
        if method == 'auto' or method == 'automatic':
            method = self._select_best_method(time_series, "M")
        
        # Generate forecast based on method
        forecast_result = self._generate_forecast_by_method(
            time_series=time_series,
            method=method,
            periods=periods,
            confidence_level=confidence_level
        )
        
        # Extract forecast values for analysis
        forecast_values = []
        if isinstance(forecast_result, dict) and 'forecast' in forecast_result:
            forecast_values = [f['value'] for f in forecast_result['forecast']]
        else:
            forecast_values = forecast_result if isinstance(forecast_result, list) else []
        
        # Detect patterns
        trend = self._detect_trend(time_series)
        seasonality = self._detect_seasonality(time_series)
        volatility = self._calculate_volatility(time_series)
        
        # Prepare forecast results in new format if needed
        if not isinstance(forecast_result, dict) or 'forecast' not in forecast_result:
            forecast_dates = self._generate_forecast_dates(periods, "M")
            confidence_intervals = self._calculate_confidence_intervals(
                time_series, forecast_values, confidence_level
            )
            
            forecast_results = []
            for i, date in enumerate(forecast_dates):
                forecast_results.append({
                    "date": date.isoformat(),
                    "predicted_value": float(forecast_values[i]) if i < len(forecast_values) else 0.0,
                    "lower_bound": float(confidence_intervals[i][0]) if i < len(confidence_intervals) else 0.0,
                    "upper_bound": float(confidence_intervals[i][1]) if i < len(confidence_intervals) else 0.0,
                    "confidence_level": confidence_level
                })
        else:
            forecast_results = forecast_result.get('forecast', [])
        
        # Store forecast in database if db session is available
        if self.db:
            self._store_forecast_results(forecast_results, method)
        
        return {
            "status": "success",
            "method": method,
            "forecast": forecast_results,
            "trend": trend,
            "seasonality_detected": seasonality,
            "volatility": volatility,
            "historical_periods": len(time_series),
            "forecast_periods": periods,
            "methodology": forecast_result.get('methodology', self.METHODS.get(method, method))
        }
    
    def forecast_single_product(
        self,
        product_id: UUID,
        historical_orders: List[Order],
        periods: int = 12,
        method: str = 'auto'
    ) -> List[Dict[str, Any]]:
        """Forecast for a single product"""
        
        # Filter orders for the specific product
        product_orders = [o for o in historical_orders if o.product_id == product_id]
        
        if not product_orders:
            return []
        
        # Use the main forecast method
        result = self.forecast(product_orders, method, periods)
        
        if result['status'] == 'success':
            return result['forecast']
        else:
            return []
    
    async def generate_forecast(
        self,
        historical_data: Union[List[float], List[Dict[str, Any]]],
        periods: int = 12,
        frequency: str = "M",
        confidence_level: float = 0.95,
        include_history: bool = True,
        additional_factors: Optional[Dict[str, Any]] = None,
        method_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a forecast based on historical data (async version).
        
        Args:
            historical_data: Historical data (list of values or dictionaries)
            periods: Number of periods to forecast
            frequency: Data frequency (D=daily, W=weekly, M=monthly, Q=quarterly, Y=yearly)
            confidence_level: Confidence level for prediction intervals
            include_history: Whether to include historical data in the result
            additional_factors: Optional additional factors for advanced forecasting
            method_params: Optional parameters for the forecasting method
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Process input data to ensure consistent format
            processed_data, dates = self._process_input_data(historical_data, frequency)
            
            # Choose forecasting method if automatic
            method = self.method
            if method == "automatic" or method == "auto":
                method = self._select_best_method(processed_data, frequency)
            
            # Set default method parameters if not provided
            if not method_params:
                method_params = {}
            
            # Generate forecast using the selected method
            forecast_result = self._generate_forecast_by_method(
                time_series=processed_data,
                method=method,
                periods=periods,
                confidence_level=confidence_level,
                frequency=frequency,
                dates=dates,
                method_params=method_params
            )
            
            # Add historical data if requested
            if include_history:
                history = []
                for i, value in enumerate(processed_data):
                    history.append({
                        "period": dates[i].strftime("%Y-%m-%d") if dates and i < len(dates) else f"Period-{i+1}",
                        "value": float(value),
                        "type": "history"
                    })
                forecast_result["history"] = history
            
            # Add method and parameters to result
            forecast_result["method"] = {
                "name": method,
                "display_name": self.METHODS.get(method, method),
                "parameters": method_params
            }
            
            # Process additional factors if provided
            if additional_factors:
                forecast_result = self._adjust_forecast_with_factors(
                    forecast_result, additional_factors, processed_data, frequency
                )
            
            # Add anomalies and insights
            forecast_result["anomalies"] = self._detect_anomalies(
                processed_data, forecast_result["forecast"]
            )
            
            forecast_result["insights"] = self._generate_insights(
                processed_data, forecast_result["forecast"], frequency
            )
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return {
                "error": str(e),
                "forecast": [],
                "method": {"name": self.method}
            }
    
    def _generate_forecast_by_method(
        self,
        time_series: List[float],
        method: str,
        periods: int,
        confidence_level: float,
        frequency: str = "M",
        dates: Optional[List[datetime]] = None,
        method_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate forecast using the specified method"""
        
        if not method_params:
            method_params = {}
        
        if not dates:
            dates = self._generate_historical_dates(len(time_series), frequency)
        
        if method == "moving_average":
            return self._forecast_moving_average(
                data=time_series,
                periods=periods,
                window=method_params.get("window", 3),
                dates=dates,
                frequency=frequency,
                confidence_level=confidence_level
            )
        
        elif method == "weighted_moving_average":
            return self._forecast_weighted_moving_average(
                data=time_series,
                periods=periods,
                weights=method_params.get("weights", [0.5, 0.3, 0.2]),
                dates=dates,
                frequency=frequency,
                confidence_level=confidence_level
            )
        
        elif method == "exponential_smoothing":
            return self._forecast_exponential_smoothing(
                data=time_series,
                periods=periods,
                alpha=method_params.get("alpha", 0.3),
                dates=dates,
                frequency=frequency,
                confidence_level=confidence_level
            )
        
        elif method == "holt_winters":
            return self._forecast_holt_winters(
                data=time_series,
                periods=periods,
                seasonal_periods=method_params.get("seasonal_periods", 12 if frequency == "M" else 4),
                trend=method_params.get("trend", "add"),
                seasonal=method_params.get("seasonal", "add"),
                dates=dates,
                frequency=frequency,
                confidence_level=confidence_level
            )
        
        elif method == "arima":
            return self._forecast_arima(
                data=time_series,
                periods=periods,
                order=method_params.get("order", (1, 1, 1)),
                dates=dates,
                frequency=frequency,
                confidence_level=confidence_level
            )
        
        elif method == "sarima":
            return self._forecast_sarima(
                data=time_series,
                periods=periods,
                order=method_params.get("order", (1, 1, 1)),
                seasonal_order=method_params.get("seasonal_order", (1, 1, 1, 12 if frequency == "M" else 4)),
                dates=dates,
                frequency=frequency,
                confidence_level=confidence_level
            )
        
        elif method == "linear_regression":
            return self._forecast_linear_regression(
                data=time_series,
                periods=periods,
                dates=dates,
                frequency=frequency,
                confidence_level=confidence_level
            )
        
        else:
            # Default to moving average
            return self._forecast_moving_average(
                data=time_series,
                periods=periods,
                window=3,
                dates=dates,
                frequency=frequency,
                confidence_level=confidence_level
            )
    
    def _prepare_time_series_from_orders(self, orders: List[Order]) -> List[float]:
        """Convert orders to time series data"""
        # Group by month and sum quantities
        monthly_data = {}
        
        for order in orders:
            month_key = order.created_at.strftime('%Y-%m')
            if month_key not in monthly_data:
                monthly_data[month_key] = 0
            monthly_data[month_key] += order.quantity
        
        # Sort by date and extract values
        sorted_months = sorted(monthly_data.keys())
        return [monthly_data[month] for month in sorted_months]
    
    def _process_input_data(
        self,
        data: Union[List[float], List[Dict[str, Any]]],
        frequency: str
    ) -> Tuple[List[float], List[datetime]]:
        """
        Process input data to ensure consistent format.
        
        Args:
            data: Input data (list of values or dictionaries)
            frequency: Data frequency
            
        Returns:
            Tuple of (values, dates)
        """
        values = []
        dates = []
        
        if not data:
            return [], []
        
        # Check if data is a list of dictionaries
        if isinstance(data[0], dict):
            # Extract values and dates from dictionaries
            for item in data:
                # Extract value
                if "value" in item:
                    values.append(float(item["value"]))
                elif "demand" in item:
                    values.append(float(item["demand"]))
                else:
                    # Try to find any numeric field
                    numeric_fields = [k for k, v in item.items() if isinstance(v, (int, float))]
                    if numeric_fields:
                        values.append(float(item[numeric_fields[0]]))
                    else:
                        raise ValueError("No numeric value found in data item")
                
                # Extract date if available
                if "date" in item:
                    if isinstance(item["date"], datetime):
                        dates.append(item["date"])
                    else:
                        try:
                            dates.append(pd.to_datetime(item["date"]))
                        except:
                            # If date parsing fails, create a date based on position
                            pass
                elif "period" in item:
                    try:
                        dates.append(pd.to_datetime(item["period"]))
                    except:
                        # If period parsing fails, create a date based on position
                        pass
        else:
            # Data is already a list of values
            values = [float(x) for x in data]
        
        # If no dates were extracted or parsing failed, create dates based on frequency
        if not dates or len(dates) != len(values):
            dates = self._generate_historical_dates(len(values), frequency)
        
        return values, dates
    
    def _generate_historical_dates(self, num_periods: int, frequency: str) -> List[datetime]:
        """Generate historical dates based on frequency"""
        dates = []
        end_date = datetime.now()
        
        if frequency == "D":
            # Daily data
            for i in range(num_periods - 1, -1, -1):
                dates.insert(0, end_date - timedelta(days=i))
        
        elif frequency == "W":
            # Weekly data
            for i in range(num_periods - 1, -1, -1):
                dates.insert(0, end_date - timedelta(weeks=i))
        
        elif frequency == "M":
            # Monthly data
            end_month = end_date.month
            end_year = end_date.year
            
            for i in range(num_periods - 1, -1, -1):
                month = end_month - (i % 12)
                year = end_year - (i // 12)
                if month <= 0:
                    month += 12
                    year -= 1
                dates.insert(0, datetime(year, month, 1))
        
        elif frequency == "Q":
            # Quarterly data
            end_quarter = (end_date.month - 1) // 3 + 1
            end_year = end_date.year
            
            for i in range(num_periods - 1, -1, -1):
                quarter = end_quarter - (i % 4)
                year = end_year - (i // 4)
                if quarter <= 0:
                    quarter += 4
                    year -= 1
                month = (quarter - 1) * 3 + 1
                dates.insert(0, datetime(year, month, 1))
        
        elif frequency == "Y":
            # Yearly data
            end_year = end_date.year
            
            for i in range(num_periods - 1, -1, -1):
                year = end_year - i
                dates.insert(0, datetime(year, 1, 1))
        
        else:
            # Unknown frequency, use sequential dates
            for i in range(num_periods):
                dates.append(end_date + timedelta(days=i))
        
        return dates
    
    def _select_best_method(self, data: List[float], frequency: str) -> str:
        """
        Automatically select the best forecasting method based on data characteristics.
        
        Args:
            data: Historical data
            frequency: Data frequency
            
        Returns:
            Selected forecasting method
        """
        try:
            if len(data) < 3:
                return 'moving_average'
            
            # Check for trend
            trend = self._detect_trend(data)
            
            # Check for seasonality
            seasonality = self._detect_seasonality(data)
            
            # Check data volatility
            volatility = self._calculate_volatility(data)
            
            # Check data length
            if len(data) < 24 and frequency in ("M", "W"):
                # For short series, use simpler methods
                if volatility > 0.5:
                    return 'exponential_smoothing'
                elif abs(trend) > 0.1:
                    return 'linear_regression'
                else:
                    return 'moving_average'
            
            # Check for seasonality in longer series
            if len(data) >= 24 and frequency in ("M", "W", "D") and seasonality:
                # Use seasonal methods for longer series
                return "holt_winters"
            
            # For longer series without clear seasonality
            if len(data) >= 24:
                return "arima"
            
            # Default based on characteristics
            if volatility > 0.5:
                return 'exponential_smoothing'
            elif abs(trend) > 0.1:
                return 'linear_regression'
            else:
                return 'moving_average'
        
        except Exception as e:
            logger.error(f"Error selecting best forecasting method: {str(e)}")
            return "exponential_smoothing"  # Default to a simple method
    
    def _forecast_moving_average(
        self,
        data: List[float],
        periods: int,
        window: int,
        dates: List[datetime],
        frequency: str,
        confidence_level: float
    ) -> Dict[str, Any]:
        """
        Generate forecast using simple moving average.
        
        Args:
            data: Historical data
            periods: Number of periods to forecast
            window: Moving average window size
            dates: Historical dates
            frequency: Data frequency
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Ensure window size is valid
            window = min(window, len(data))
            if window < 1:
                window = 1
            
            # Calculate moving average for recent periods
            recent_avg = sum(data[-window:]) / window
            
            # Apply slight trend adjustment
            trend = 0
            if len(data) >= 2:
                trend = (data[-1] - data[-2]) * 0.1
            
            # Generate forecast with trend
            forecast = [recent_avg + (i * trend) for i in range(1, periods + 1)]
            
            # Calculate prediction intervals
            # For moving average, use standard deviation of historical data
            std_dev = np.std(data[-window:], ddof=1) if len(data) > 1 else 0
            z_score = 1.96  # Approximately 95% confidence interval
            if confidence_level == 0.90:
                z_score = 1.645
            elif confidence_level == 0.99:
                z_score = 2.576
            
            lower_bounds = [max(0, x - z_score * std_dev) for x in forecast]
            upper_bounds = [x + z_score * std_dev for x in forecast]
            
            # Generate forecast dates
            forecast_dates = self._generate_forecast_dates(periods, frequency, dates[-1] if dates else None)
            
            # Prepare forecast result
            forecast_result = {
                "forecast": [
                    {
                        "period": date.strftime("%Y-%m-%d"),
                        "value": float(forecast[i]),
                        "lower_bound": float(lower_bounds[i]),
                        "upper_bound": float(upper_bounds[i]),
                        "type": "forecast"
                    }
                    for i, date in enumerate(forecast_dates)
                ],
                "methodology": "Simple Moving Average Forecast"
            }
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error generating moving average forecast: {str(e)}")
            return {
                "error": str(e),
                "forecast": [],
                "methodology": "Simple Moving Average Forecast"
            }
    
    def _forecast_weighted_moving_average(
        self,
        data: List[float],
        periods: int,
        weights: List[float],
        dates: List[datetime],
        frequency: str,
        confidence_level: float
    ) -> Dict[str, Any]:
        """
        Generate forecast using weighted moving average.
        
        Args:
            data: Historical data
            periods: Number of periods to forecast
            weights: Weights for the weighted moving average
            dates: Historical dates
            frequency: Data frequency
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Ensure weights array is valid
            if not weights:
                weights = [0.5, 0.3, 0.2]
            
            # Normalize weights to sum to 1
            weights = [w / sum(weights) for w in weights]
            
            # Ensure weights array is not longer than data
            weights = weights[:len(data)]
            
            # If data is shorter than weights, truncate weights
            if len(data) < len(weights):
                weights = weights[-len(data):]
                weights = [w / sum(weights) for w in weights]  # Renormalize
            
            # Calculate weighted average of recent periods
            window = len(weights)
            recent_values = data[-window:]
            weighted_avg = sum(w * v for w, v in zip(weights, recent_values)) / sum(weights)
            
            # Generate forecast (flat forecast)
            forecast = [weighted_avg] * periods
            
            # Calculate prediction intervals
            # For weighted MA, use weighted standard deviation
            weighted_var = sum(w * ((v - weighted_avg) ** 2) for w, v in zip(weights, recent_values)) / sum(weights)
            std_dev = math.sqrt(weighted_var)
            
            z_score = 1.96  # Approximately 95% confidence interval
            if confidence_level == 0.90:
                z_score = 1.645
            elif confidence_level == 0.99:
                z_score = 2.576
            
            lower_bounds = [max(0, x - z_score * std_dev) for x in forecast]
            upper_bounds = [x + z_score * std_dev for x in forecast]
            
            # Generate forecast dates
            forecast_dates = self._generate_forecast_dates(periods, frequency, dates[-1] if dates else None)
            
            # Prepare forecast result
            forecast_result = {
                "forecast": [
                    {
                        "period": date.strftime("%Y-%m-%d"),
                        "value": float(forecast[i]),
                        "lower_bound": float(lower_bounds[i]),
                        "upper_bound": float(upper_bounds[i]),
                        "type": "forecast"
                    }
                    for i, date in enumerate(forecast_dates)
                ],
                "methodology": "Weighted Moving Average Forecast",
                "weights": weights
            }
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error generating weighted moving average forecast: {str(e)}")
            return {
                "error": str(e),
                "forecast": [],
                "methodology": "Weighted Moving Average Forecast"
            }
    
    def _forecast_exponential_smoothing(
        self,
        data: List[float],
        periods: int,
        alpha: float,
        dates: List[datetime],
        frequency: str,
        confidence_level: float
    ) -> Dict[str, Any]:
        """
        Generate forecast using exponential smoothing.
        
        Args:
            data: Historical data
            periods: Number of periods to forecast
            alpha: Smoothing parameter (0 < alpha < 1)
            dates: Historical dates
            frequency: Data frequency
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Ensure alpha is valid
            alpha = max(0.01, min(0.99, alpha))
            
            # Simple exponential smoothing implementation
            if len(data) == 0:
                return {
                    "forecast": [],
                    "methodology": "Exponential Smoothing Forecast"
                }
            
            # Calculate exponential smoothing
            s = [data[0]]
            for i in range(1, len(data)):
                s.append(alpha * data[i] + (1 - alpha) * s[-1])
            
            # Forecast
            last_value = s[-1]
            trend = 0
            if len(s) >= 2:
                trend = (s[-1] - s[-2]) * 0.5
            
            forecast = [last_value + (i * trend) for i in range(1, periods + 1)]
            
            # Calculate prediction intervals
            residuals = [data[i] - s[i] for i in range(len(data))]
            residual_std = np.std(residuals, ddof=1) if len(residuals) > 1 else 0
            
            z_score = 1.96  # Approximately 95% confidence interval
            if confidence_level == 0.90:
                z_score = 1.645
            elif confidence_level == 0.99:
                z_score = 2.576
            
            lower_bounds = [max(0, x - z_score * residual_std) for x in forecast]
            upper_bounds = [x + z_score * residual_std for x in forecast]
            
            # Generate forecast dates
            forecast_dates = self._generate_forecast_dates(periods, frequency, dates[-1] if dates else None)
            
            # Prepare forecast result
            forecast_result = {
                "forecast": [
                    {
                        "period": date.strftime("%Y-%m-%d"),
                        "value": float(forecast[i]),
                        "lower_bound": float(lower_bounds[i]),
                        "upper_bound": float(upper_bounds[i]),
                        "type": "forecast"
                    }
                    for i, date in enumerate(forecast_dates)
                ],
                "methodology": "Exponential Smoothing Forecast",
                "parameters": {"alpha": alpha}
            }
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error generating exponential smoothing forecast: {str(e)}")
            return {
                "error": str(e),
                "forecast": [],
                "methodology": "Exponential Smoothing Forecast"
            }
    
    def _forecast_linear_regression(
        self,
        data: List[float],
        periods: int,
        dates: List[datetime],
        frequency: str,
        confidence_level: float
    ) -> Dict[str, Any]:
        """Linear regression forecast"""
        try:
            n = len(data)
            if n < 2:
                return {
                    "forecast": [{"period": "", "value": data[-1] if data else 0, 
                                "lower_bound": 0, "upper_bound": 0, "type": "forecast"}] * periods,
                    "methodology": "Linear Regression Forecast"
                }
            
            # Calculate linear regression
            x = list(range(n))
            x_mean = np.mean(x)
            y_mean = np.mean(data)
            
            numerator = sum((x[i] - x_mean) * (data[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
            
            intercept = y_mean - slope * x_mean
            
            # Generate forecast
            forecast = []
            for i in range(periods):
                forecast_x = n + i
                forecast_y = slope * forecast_x + intercept
                forecast.append(max(0, forecast_y))  # Ensure non-negative
            
            # Calculate confidence intervals
            residuals = [data[i] - (slope * i + intercept) for i in range(n)]
            residual_std = np.std(residuals, ddof=1) if len(residuals) > 1 else 0
            
            z_score = 1.96
            if confidence_level == 0.90:
                z_score = 1.645
            elif confidence_level == 0.99:
                z_score = 2.576
            
            # Generate forecast dates
            forecast_dates = self._generate_forecast_dates(periods, frequency, dates[-1] if dates else None)
            
            # Prepare results
            forecast_result = {
                "forecast": [
                    {
                        "period": date.strftime("%Y-%m-%d"),
                        "value": float(forecast[i]),
                        "lower_bound": float(max(0, forecast[i] - z_score * residual_std * np.sqrt(1 + 1/n + (n+i-x_mean)**2/denominator))),
                        "upper_bound": float(forecast[i] + z_score * residual_std * np.sqrt(1 + 1/n + (n+i-x_mean)**2/denominator)),
                        "type": "forecast"
                    }
                    for i, date in enumerate(forecast_dates)
                ],
                "methodology": "Linear Regression Forecast",
                "parameters": {
                    "slope": float(slope),
                    "intercept": float(intercept)
                }
            }
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error generating linear regression forecast: {str(e)}")
            return {
                "error": str(e),
                "forecast": [],
                "methodology": "Linear Regression Forecast"
            }
    
    def _forecast_holt_winters(
        self,
        data: List[float],
        periods: int,
        seasonal_periods: int,
        trend: str,
        seasonal: str,
        dates: List[datetime],
        frequency: str,
        confidence_level: float
    ) -> Dict[str, Any]:
        """
        Generate forecast using Holt-Winters Exponential Smoothing.
        
        Args:
            data: Historical data
            periods: Number of periods to forecast
            seasonal_periods: Number of periods in a seasonal cycle
            trend: Trend type ('add' or 'mul')
            seasonal: Seasonal type ('add' or 'mul')
            dates: Historical dates
            frequency: Data frequency
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Create pandas Series with datetime index
            if dates and len(dates) == len(data):
                ts = pd.Series(data, index=pd.DatetimeIndex(dates, freq=frequency))
            else:
                # Create a synthetic DatetimeIndex
                end_date = datetime.now()
                if frequency == "M":
                    # Monthly data
                    start_date = end_date - pd.DateOffset(months=len(data))
                    idx = pd.date_range(start=start_date, periods=len(data), freq='M')
                else:
                    # Default to daily
                    start_date = end_date - pd.DateOffset(days=len(data))
                    idx = pd.date_range(start=start_date, periods=len(data), freq='D')
                
                ts = pd.Series(data, index=idx)
            
            # Validate seasonal periods
            if len(data) <= seasonal_periods * 2:
                # Not enough data for seasonal model, fall back to non-seasonal
                model = ExponentialSmoothing(ts, trend=trend, seasonal=None)
                logger.warning(f"Not enough data for seasonal model (need {seasonal_periods*2}, have {len(data)}). Using non-seasonal model.")
            else:
                # Use seasonal model
                model = ExponentialSmoothing(
                    ts, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods
                )
            
            # Fit the model
            fitted_model = model.fit(optimized=True)
            
            # Generate forecast
            forecast = fitted_model.forecast(periods)
            
            # Calculate prediction intervals
            residuals = fitted_model.resid
            residual_std = residuals.std()
            
            z_score = 1.96  # Approximately 95% confidence interval
            if confidence_level == 0.90:
                z_score = 1.645
            elif confidence_level == 0.99:
                z_score = 2.576
            
            lower_bounds = [max(0, x - z_score * residual_std * np.sqrt(i+1)) for i, x in enumerate(forecast)]
            upper_bounds = [x + z_score * residual_std * np.sqrt(i+1) for i, x in enumerate(forecast)]
            
            # Prepare forecast result
            forecast_result = {
                "forecast": [
                    {
                        "period": date.strftime("%Y-%m-%d"),
                        "value": float(forecast.iloc[i]) if hasattr(forecast, 'iloc') else float(forecast[i]),
                        "lower_bound": float(lower_bounds[i]),
                        "upper_bound": float(upper_bounds[i]),
                        "type": "forecast"
                    }
                    for i, date in enumerate(forecast.index if hasattr(forecast, 'index') else self._generate_forecast_dates(periods, frequency, dates[-1] if dates else None))
                ],
                "methodology": "Holt-Winters Exponential Smoothing Forecast",
                "parameters": {
                    "trend": trend,
                    "seasonal": seasonal,
                    "seasonal_periods": seasonal_periods,
                    "fitted_params": {
                        "smoothing_level": round(fitted_model.params['smoothing_level'], 4),
                        "smoothing_trend": round(fitted_model.params.get('smoothing_trend', 0), 4),
                        "smoothing_seasonal": round(fitted_model.params.get('smoothing_seasonal', 0), 4)
                    }
                }
            }
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error generating Holt-Winters forecast: {str(e)}")
            return {
                "error": str(e),
                "forecast": [],
                "methodology": "Holt-Winters Exponential Smoothing Forecast"
            }
    
    def _forecast_arima(
        self,
        data: List[float],
        periods: int,
        order: Tuple[int, int, int],
        dates: List[datetime],
        frequency: str,
        confidence_level: float
    ) -> Dict[str, Any]:
        """
        Generate forecast using ARIMA (AutoRegressive Integrated Moving Average).
        
        Args:
            data: Historical data
            periods: Number of periods to forecast
            order: ARIMA order (p, d, q)
            dates: Historical dates
            frequency: Data frequency
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Create pandas Series with datetime index
            if dates and len(dates) == len(data):
                ts = pd.Series(data, index=pd.DatetimeIndex(dates, freq=frequency))
            else:
                # Create a synthetic DatetimeIndex
                end_date = datetime.now()
                if frequency == "M":
                    # Monthly data
                    start_date = end_date - pd.DateOffset(months=len(data))
                    idx = pd.date_range(start=start_date, periods=len(data), freq='M')
                else:
                    # Default to daily
                    start_date = end_date - pd.DateOffset(days=len(data))
                    idx = pd.date_range(start=start_date, periods=len(data), freq='D')
                
                ts = pd.Series(data, index=idx)
            
            # Fit ARIMA model
            model = ARIMA(ts, order=order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(periods)
            
            # Get forecast index (dates)
            forecast_index = pd.date_range(
                start=ts.index[-1] + pd.Timedelta(days=1),
                periods=periods,
                freq=ts.index.freq
            )
            
            # Get prediction intervals
            forecast_obj = fitted_model.get_forecast(periods)
            intervals = forecast_obj.conf_int(alpha=1-confidence_level)
            
            lower_bounds = intervals.iloc[:, 0].values
            upper_bounds = intervals.iloc[:, 1].values
            
            # Ensure no negative lower bounds
            lower_bounds = np.maximum(0, lower_bounds)
            
            # Prepare forecast result
            forecast_result = {
                "forecast": [
                    {
                        "period": date.strftime("%Y-%m-%d"),
                        "value": float(forecast.iloc[i]) if hasattr(forecast, 'iloc') else float(forecast[i]),
                        "lower_bound": float(lower_bounds[i]),
                        "upper_bound": float(upper_bounds[i]),
                        "type": "forecast"
                    }
                    for i, date in enumerate(forecast_index)
                ],
                "methodology": "ARIMA Forecast",
                "parameters": {
                    "order": order,
                    "aic": round(fitted_model.aic, 2),
                    "bic": round(fitted_model.bic, 2)
                }
            }
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error generating ARIMA forecast: {str(e)}")
            return {
                "error": str(e),
                "forecast": [],
                "methodology": "ARIMA Forecast"
            }
    
    def _forecast_sarima(
        self,
        data: List[float],
        periods: int,
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int],
        dates: List[datetime],
        frequency: str,
        confidence_level: float
    ) -> Dict[str, Any]:
        """
        Generate forecast using SARIMA (Seasonal ARIMA).
        
        Args:
            data: Historical data
            periods: Number of periods to forecast
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            dates: Historical dates
            frequency: Data frequency
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Create pandas Series with datetime index
            if dates and len(dates) == len(data):
                ts = pd.Series(data, index=pd.DatetimeIndex(dates, freq=frequency))
            else:
                # Create a synthetic DatetimeIndex
                end_date = datetime.now()
                if frequency == "M":
                    # Monthly data
                    start_date = end_date - pd.DateOffset(months=len(data))
                    idx = pd.date_range(start=start_date, periods=len(data), freq='M')
                else:
                    # Default to daily
                    start_date = end_date - pd.DateOffset(days=len(data))
                    idx = pd.date_range(start=start_date, periods=len(data), freq='D')
                
                ts = pd.Series(data, index=idx)
            
            # Fit SARIMA model
            model = SARIMAX(
                ts, 
                order=order, 
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)
            
            # Generate forecast
            forecast = fitted_model.forecast(periods)
            
            # Get forecast index (dates)
            forecast_index = pd.date_range(
                start=ts.index[-1] + pd.Timedelta(days=1),
                periods=periods,
                freq=ts.index.freq
            )
            
            # Get prediction intervals
            forecast_obj = fitted_model.get_forecast(periods)
            intervals = forecast_obj.conf_int(alpha=1-confidence_level)
            
            lower_bounds = intervals.iloc[:, 0].values
            upper_bounds = intervals.iloc[:, 1].values
            
            # Ensure no negative lower bounds
            lower_bounds = np.maximum(0, lower_bounds)
            
            # Prepare forecast result
            forecast_result = {
                "forecast": [
                    {
                        "period": date.strftime("%Y-%m-%d"),
                        "value": float(forecast.iloc[i]) if hasattr(forecast, 'iloc') else float(forecast[i]),
                        "lower_bound": float(lower_bounds[i]),
                        "upper_bound": float(upper_bounds[i]),
                        "type": "forecast"
                    }
                    for i, date in enumerate(forecast_index)
                ],
                "methodology": "SARIMA Forecast",
                "parameters": {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "aic": round(fitted_model.aic, 2),
                    "bic": round(fitted_model.bic, 2)
                }
            }
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error generating SARIMA forecast: {str(e)}")
            return {
                "error": str(e),
                "forecast": [],
                "methodology": "SARIMA Forecast"
            }
    
    def _generate_forecast_dates(
        self,
        periods: int,
        frequency: str,
        last_date: Optional[datetime] = None
    ) -> List[datetime]:
        """
        Generate dates for the forecast periods.
        
        Args:
            periods: Number of periods to forecast
            frequency: Data frequency
            last_date: Last date in historical data (optional)
            
        Returns:
            List of dates for the forecast periods
        """
        if not last_date:
            last_date = datetime.now()
        
        dates = []
        
        for i in range(1, periods + 1):
            if frequency == "D":
                # Daily data
                date = last_date + timedelta(days=i)
            
            elif frequency == "W":
                # Weekly data
                date = last_date + timedelta(weeks=i)
            
            elif frequency == "M":
                # Monthly data
                month = last_date.month + i
                year = last_date.year
                while month > 12:
                    month -= 12
                    year += 1
                date = datetime(year, month, 1)
            
            elif frequency == "Q":
                # Quarterly data
                quarter = ((last_date.month - 1) // 3 + 1) + i
                year = last_date.year
                while quarter > 4:
                    quarter -= 4
                    year += 1
                month = (quarter - 1) * 3 + 1
                date = datetime(year, month, 1)
            
            elif frequency == "Y":
                # Yearly data
                date = datetime(last_date.year + i, 1, 1)
            
            else:
                # Unknown frequency, use daily
                date = last_date + timedelta(days=i)
            
            dates.append(date)
        
        return dates
    
    def _calculate_confidence_intervals(
        self,
        time_series: List[float],
        forecast: List[float],
        confidence_level: float
    ) -> List[tuple]:
        """Calculate confidence intervals for forecast"""
        # Calculate standard deviation of historical data
        if len(time_series) < 2:
            std_dev = 0
        else:
            std_dev = np.std(time_series)
        
        # Z-score for confidence level
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        z = z_scores.get(confidence_level, 1.96)
        
        # Calculate intervals
        intervals = []
        for i, value in enumerate(forecast):
            # Increase uncertainty for further forecasts
            uncertainty = std_dev * z * (1 + i * 0.1)
            lower = max(0, value - uncertainty)
            upper = value + uncertainty
            intervals.append((lower, upper))
        
        return intervals
    
    def _detect_trend(self, time_series: List[float]) -> float:
        """Detect trend in time series data"""
        if len(time_series) < 2:
            return 0.0
        
        # Simple linear regression to detect trend
        x = list(range(len(time_series)))
        x_mean = np.mean(x)
        y_mean = np.mean(time_series)
        
        numerator = sum((x[i] - x_mean) * (time_series[i] - y_mean) for i in range(len(x)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope / y_mean if y_mean != 0 else 0.0
    
    def _detect_seasonality(self, time_series: List[float]) -> bool:
        """Detect seasonality in time series data"""
        if len(time_series) < 24:  # Need at least 2 years of monthly data
            return False
        
        # Simple seasonality detection based on autocorrelation
        # Check if there's a pattern every 12 months
        if len(time_series) >= 24:
            correlation = np.corrcoef(time_series[:-12], time_series[12:])[0, 1]
            return correlation > 0.7
        
        return False
    
    def _calculate_volatility(self, time_series: List[float]) -> float:
        """Calculate volatility of time series data"""
        if len(time_series) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean = np.mean(time_series)
        std = np.std(time_series)
        
        if mean == 0:
            return 0.0
        
        return std / mean
    
    def _store_forecast_results(self, forecast_results: List[Dict], method: str):
        """Store forecast results in the database"""
        if not self.db:
            return
        
        try:
            # Store as AnalyticsMetric
            metric = AnalyticsMetric(
                metric_name=f"forecast_{method}",
                metric_value=json.dumps({
                    "forecast": forecast_results,
                    "method": method,
                    "generated_at": datetime.now().isoformat()
                }),
                calculated_at=datetime.now()
            )
            self.db.add(metric)
            self.db.commit()
            logger.info(f"Stored forecast results using method: {method}")
        except Exception as e:
            logger.error(f"Error storing forecast results: {str(e)}")
            self.db.rollback()
    
    def _adjust_forecast_with_factors(
        self,
        forecast_result: Dict[str, Any],
        additional_factors: Dict[str, Any],
        historical_data: List[float],
        frequency: str
    ) -> Dict[str, Any]:
        """
        Adjust forecast based on additional factors.
        
        Args:
            forecast_result: Original forecast result
            additional_factors: Additional factors for adjustment
            historical_data: Historical data
            frequency: Data frequency
            
        Returns:
            Adjusted forecast result
        """
        adjusted_forecast = forecast_result.copy()
        forecast_periods = adjusted_forecast["forecast"]
        factors_applied = []
        
        try:
            # Process promotions
            if "promotions" in additional_factors:
                promotions = additional_factors["promotions"]
                for promotion in promotions:
                    start_date = pd.to_datetime(promotion.get("start_date"))
                    end_date = pd.to_datetime(promotion.get("end_date"))
                    impact = float(promotion.get("impact", 1.2))  # Default 20% increase
                    
                    # Apply to relevant periods
                    for i, period in enumerate(forecast_periods):
                        period_date = pd.to_datetime(period["period"])
                        if start_date <= period_date <= end_date:
                            # Adjust forecast value
                            forecast_periods[i]["value"] *= impact
                            forecast_periods[i]["lower_bound"] *= impact
                            forecast_periods[i]["upper_bound"] *= impact
                            forecast_periods[i]["factors"] = forecast_periods[i].get("factors", []) + [
                                f"Promotion: {impact:.2f}x"
                            ]
                    
                    factors_applied.append(f"Promotion from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}: {impact:.2f}x")
            
            # Process seasonality
            if "seasonality" in additional_factors:
                seasonality = additional_factors["seasonality"]
                for season in seasonality:
                    months = season.get("months", [])
                    factor = float(season.get("factor", 1.0))
                    
                    # Apply to relevant periods
                    for i, period in enumerate(forecast_periods):
                        period_date = pd.to_datetime(period["period"])
                        if period_date.month in months:
                            # Adjust forecast value
                            forecast_periods[i]["value"] *= factor
                            forecast_periods[i]["lower_bound"] *= factor
                            forecast_periods[i]["upper_bound"] *= factor
                            forecast_periods[i]["factors"] = forecast_periods[i].get("factors", []) + [
                                f"Seasonality: {factor:.2f}x"
                            ]
                    
                    months_str = ", ".join(str(m) for m in months)
                    factors_applied.append(f"Seasonal factor for months {months_str}: {factor:.2f}x")
            
            # Process events
            if "events" in additional_factors:
                events = additional_factors["events"]
                for event in events:
                    date = pd.to_datetime(event.get("date"))
                    impact = float(event.get("impact", 1.0))
                    duration = int(event.get("duration", 1))  # Days
                    
                    # Apply to relevant periods
                    for i, period in enumerate(forecast_periods):
                        period_date = pd.to_datetime(period["period"])
                        days_diff = abs((period_date - date).days)
                        if days_diff <= duration:
                            # Adjust forecast value
                            forecast_periods[i]["value"] *= impact
                            forecast_periods[i]["lower_bound"] *= impact
                            forecast_periods[i]["upper_bound"] *= impact
                            forecast_periods[i]["factors"] = forecast_periods[i].get("factors", []) + [
                                f"Event: {impact:.2f}x"
                            ]
                    
                    factors_applied.append(f"Event on {date.strftime('%Y-%m-%d')} (duration: {duration} days): {impact:.2f}x")
            
            # Update forecast result
            adjusted_forecast["forecast"] = forecast_periods
            adjusted_forecast["factors_applied"] = factors_applied
            
            return adjusted_forecast
            
        except Exception as e:
            logger.error(f"Error adjusting forecast with factors: {str(e)}")
            return forecast_result
    
    def _detect_anomalies(
        self,
        historical_data: List[float],
        forecast: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in historical data and forecast.
        
        Args:
            historical_data: Historical data
            forecast: Forecast data
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        try:
            # Calculate statistics for historical data
            mean = np.mean(historical_data)
            std_dev = np.std(historical_data, ddof=1)
            
            # Define anomaly threshold (Z-score > 3)
            threshold = 3
            
            # Detect anomalies in historical data
            for i, value in enumerate(historical_data):
                z_score = abs((value - mean) / std_dev) if std_dev > 0 else 0
                if z_score > threshold:
                    anomalies.append({
                        "type": "historical",
                        "index": i,
                        "value": float(value),
                        "z_score": float(z_score),
                        "description": f"Historical value with Z-score of {z_score:.2f}"
                    })
            
            # Detect potential anomalies in forecast
            forecast_values = [f["value"] for f in forecast]
            # Check for big jumps between historical and forecast
            if historical_data and forecast_values:
                last_historical = historical_data[-1]
                first_forecast = forecast_values[0]
                
                percent_change = (first_forecast - last_historical) / last_historical * 100 if last_historical != 0 else 0
                if abs(percent_change) > 50:  # 50% change is suspicious
                    anomalies.append({
                        "type": "forecast_jump",
                        "from_value": float(last_historical),
                        "to_value": float(first_forecast),
                        "percent_change": float(percent_change),
                        "description": f"Large jump between historical and forecast: {percent_change:.1f}%"
                    })
            
            # Check for outliers within forecast
            if forecast_values:
                forecast_mean = np.mean(forecast_values)
                forecast_std = np.std(forecast_values, ddof=1)
                
                for i, value in enumerate(forecast_values):
                    z_score = abs((value - forecast_mean) / forecast_std) if forecast_std > 0 else 0
                    if z_score > threshold:
                        anomalies.append({
                            "type": "forecast_outlier",
                            "index": i,
                            "value": float(value),
                            "z_score": float(z_score),
                            "description": f"Forecast value with Z-score of {z_score:.2f}"
                        })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def _generate_insights(
        self,
        historical_data: List[float],
        forecast: List[Dict[str, Any]],
        frequency: str
    ) -> List[str]:
        """
        Generate insights based on historical data and forecast.
        
        Args:
            historical_data: Historical data
            forecast: Forecast data
            frequency: Data frequency
            
        Returns:
            List of insights
        """
        insights = []
        
        try:
            # Calculate statistics
            if not historical_data or len(historical_data) < 2:
                return ["Insufficient historical data for analysis"]
            
            # Calculate trends
            forecast_values = [f["value"] for f in forecast]
            
            # Historical trend
            historical_start = historical_data[0]
            historical_end = historical_data[-1]
            historical_change = (historical_end - historical_start) / historical_start * 100 if historical_start != 0 else 0
            
            # Forecast trend
            if forecast_values:
                forecast_start = forecast_values[0]
                forecast_end = forecast_values[-1]
                forecast_change = (forecast_end - forecast_start) / forecast_start * 100 if forecast_start != 0 else 0
            else:
                forecast_change = 0
            
            # Generate insights based on trends
            if abs(historical_change) < 5:
                insights.append(f"Historical demand was relatively stable ({historical_change:.1f}% change)")
            elif historical_change > 0:
                insights.append(f"Historical demand showed an increasing trend (+{historical_change:.1f}%)")
            else:
                insights.append(f"Historical demand showed a decreasing trend ({historical_change:.1f}%)")
            
            if abs(forecast_change) < 5:
                insights.append(f"Forecast demand is expected to remain stable ({forecast_change:.1f}% change)")
            elif forecast_change > 0:
                insights.append(f"Forecast demand shows an increasing trend (+{forecast_change:.1f}%)")
            else:
                insights.append(f"Forecast demand shows a decreasing trend ({forecast_change:.1f}%)")
            
            # Check for seasonality
            if len(historical_data) >= 12 and frequency == "M":
                # Simple seasonality check for monthly data
                monthly_avg = []
                for month in range(1, 13):
                    values = [historical_data[i] for i in range(len(historical_data)) if (i % 12) + 1 == month]
                    if values:
                        monthly_avg.append(np.mean(values))
                
                if monthly_avg:
                    overall_avg = np.mean(monthly_avg)
                    month_variation = [((m / overall_avg) - 1) * 100 for m in monthly_avg]
                    
                    # Check if there's significant variation between months
                    if max(month_variation) - min(month_variation) > 20:
                        high_month = month_variation.index(max(month_variation)) + 1
                        low_month = month_variation.index(min(month_variation)) + 1
                        insights.append(
                            f"Significant seasonality detected: Month {high_month} is typically {max(month_variation):.1f}% " +
                            f"above average, while month {low_month} is {abs(min(month_variation)):.1f}% below average"
                        )
            
            # Add forecast reliability insight
            if len(forecast) > 0:
                avg_confidence_interval = np.mean([f["upper_bound"] - f["lower_bound"] for f in forecast])
                avg_forecast = np.mean([f["value"] for f in forecast])
                confidence_ratio = avg_confidence_interval / avg_forecast if avg_forecast > 0 else 0
                
                if confidence_ratio < 0.2:
                    insights.append("Forecast has relatively narrow confidence intervals, suggesting high reliability")
                elif confidence_ratio < 0.5:
                    insights.append("Forecast has moderate confidence intervals")
                else:
                    insights.append("Forecast has wide confidence intervals, suggesting higher uncertainty")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return ["Error generating insights"]

    @staticmethod
    async def get_product_demand_data(
        product_id: str,
        client_id: str,
        connection_id: Optional[str] = None,
        period: str = "last_12_months",
        frequency: str = "M"
    ) -> Dict[str, Any]:
        """
        Get historical demand data for a product.
        
        Args:
            product_id: Product ID
            client_id: Client ID
            connection_id: Optional connection ID
            period: Time period for data
            frequency: Data frequency
            
        Returns:
            Dictionary with historical demand data
        """
        try:
            # This is a placeholder implementation
            # In a real implementation, you would fetch data from a database or API
            
            # Get data from database
            from app.db.interfaces.product_interface import ProductInterface
            
            # Create interface
            product_interface = ProductInterface(client_id=client_id, connection_id=connection_id)
            
            # Get historical demand
            demand_data = await product_interface.get_historical_demand(
                product_id=product_id, 
                period=period,
                frequency=frequency
            )
            
            return demand_data
            
        except Exception as e:
            logger.error(f"Error getting product demand data: {str(e)}")
            
            # Generate mock data for demonstration or testing
            mock_data = ForecastEngine._generate_mock_demand_data(
                frequency=frequency,
                periods=12 if period == "last_12_months" else 24
            )
            
            logger.warning(f"Using mock demand data for product {product_id}: {len(mock_data['values'])} periods")
            return mock_data
    
    @staticmethod
    def _generate_mock_demand_data(
        frequency: str = "M",
        periods: int = 12,
        base_value: float = 100,
        trend: float = 0.05,
        seasonality: float = 0.2,
        noise: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate mock demand data for testing.
        
        Args:
            frequency: Data frequency
            periods: Number of periods
            base_value: Base demand value
            trend: Trend factor (% increase per period)
            seasonality: Seasonality factor
            noise: Noise factor
            
        Returns:
            Dictionary with mock demand data
        """
        values = []
        dates = []
        
        # Generate period dates
        end_date = datetime.now()
        
        for i in range(periods - 1, -1, -1):
            if frequency == "D":
                # Daily data
                date = end_date - timedelta(days=i)
            elif frequency == "W":
                # Weekly data
                date = end_date - timedelta(weeks=i)
            elif frequency == "M":
                # Monthly data
                month = end_date.month - (i % 12)
                year = end_date.year - (i // 12)
                if month <= 0:
                    month += 12
                    year -= 1
                date = datetime(year, month, 1)
            elif frequency == "Q":
                # Quarterly data
                quarter = ((end_date.month - 1) // 3 + 1) - (i % 4)
                year = end_date.year - (i // 4)
                if quarter <= 0:
                    quarter += 4
                    year -= 1
                month = (quarter - 1) * 3 + 1
                date = datetime(year, month, 1)
            else:
                # Default to monthly
                month = end_date.month - (i % 12)
                year = end_date.year - (i // 12)
                if month <= 0:
                    month += 12
                    year -= 1
                date = datetime(year, month, 1)
            
            dates.append(date)
        
        # Generate demand values with trend, seasonality, and noise
        for i in range(periods):
            # Apply trend
            trend_component = base_value * (1 + trend) ** i
            
            # Apply seasonality
            season_component = 1.0
            if frequency in ("M", "Q"):
                # Monthly or quarterly seasonality
                month = dates[i].month
                # Higher demand in months 1, 6, 11, 12
                if month in (1, 6, 11, 12):
                    season_component = 1.0 + seasonality
                # Lower demand in months 2, 7, 8
                elif month in (2, 7, 8):
                    season_component = 1.0 - seasonality
            
            # Apply noise
            noise_component = np.random.normal(1.0, noise)
            
            # Calculate final value
            value = trend_component * season_component * noise_component
            values.append(max(0, value))  # Ensure non-negative values
        
        return {
            "values": values,
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "product_id": "MOCK_PRODUCT",
            "is_mock_data": True
        }


# Add these functions at the end of the file for module-level access

async def generate_forecast(
    historical_data: Union[List[float], List[Dict[str, Any]]],
    periods: int = 12,
    frequency: str = "M",
    confidence_level: float = 0.95,
    include_history: bool = True,
    additional_factors: Optional[Dict[str, Any]] = None,
    method_params: Optional[Dict[str, Any]] = None,
    method: str = "automatic"
) -> Dict[str, Any]:
    """
    Generate a forecast based on historical data.
    
    Args:
        historical_data: Historical data (list of values or dictionaries)
        periods: Number of periods to forecast
        frequency: Data frequency (D=daily, W=weekly, M=monthly, Q=quarterly, Y=yearly)
        confidence_level: Confidence level for prediction intervals
        include_history: Whether to include historical data in the result
        additional_factors: Optional additional factors for advanced forecasting
        method_params: Optional parameters for the forecasting method
        method: Forecasting method to use (default: automatic selection)
        
    Returns:
        Dictionary with forecast results
    """
    # Create a forecast engine with the specified method
    engine = ForecastEngine(method=method)
    
    # Call the instance method
    return await engine.generate_forecast(
        historical_data=historical_data,
        periods=periods,
        frequency=frequency,
        confidence_level=confidence_level,
        include_history=include_history,
        additional_factors=additional_factors,
        method_params=method_params
    )

async def get_product_demand_data(
    product_id: str,
    client_id: str,
    connection_id: Optional[str] = None,
    period: str = "last_12_months",
    frequency: str = "M"
) -> Dict[str, Any]:
    """
    Get historical demand data for a product.
    
    Args:
        product_id: Product ID
        client_id: Client ID
        connection_id: Optional connection ID
        period: Time period for data
        frequency: Data frequency
        
    Returns:
        Dictionary with historical demand data
    """
    return await ForecastEngine.get_product_demand_data(
        product_id=product_id,
        client_id=client_id,
        connection_id=connection_id,
        period=period,
        frequency=frequency
    )