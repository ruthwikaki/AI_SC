from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import re
from functools import partial
from app.utils.logger import get_logger
# Initialize logger
logger = get_logger(__name__)

class UnitConverter:
    """
    Converts units of measurement in supply chain datasets.
    
    This class provides methods to standardize units for weight, distance,
    volume, currency, and other measurements commonly used in supply chain data.
    """
    
    def __init__(self):
        """Initialize the unit converter."""
        # Define common conversion factors
        # Weight conversions (to kg)
        self.weight_conversions = {
            "kg": 1.0,
            "g": 0.001,
            "mg": 0.000001,
            "lb": 0.45359237,
            "oz": 0.02834952,
            "ton": 1000.0,
            "metric_ton": 1000.0,
            "tonne": 1000.0,
            "short_ton": 907.18474,
            "long_ton": 1016.0469088
        }
        
        # Distance conversions (to meters)
        self.distance_conversions = {
            "m": 1.0,
            "km": 1000.0,
            "cm": 0.01,
            "mm": 0.001,
            "in": 0.0254,
            "ft": 0.3048,
            "yd": 0.9144,
            "mi": 1609.344
        }
        
        # Volume conversions (to liters)
        self.volume_conversions = {
            "l": 1.0,
            "liter": 1.0,
            "ml": 0.001,
            "milliliter": 0.001,
            "cubic_meter": 1000.0,
            "m3": 1000.0,
            "gal": 3.78541,
            "gallon": 3.78541,
            "qt": 0.946353,
            "quart": 0.946353,
            "pt": 0.473176,
            "pint": 0.473176,
            "fl_oz": 0.0295735,
            "fluid_ounce": 0.0295735,
            "cup": 0.24,
            "tbsp": 0.0147868,
            "tablespoon": 0.0147868,
            "tsp": 0.00492892,
            "teaspoon": 0.00492892,
            "cubic_foot": 28.3168,
            "ft3": 28.3168,
            "cubic_inch": 0.0163871,
            "in3": 0.0163871
        }
        
        # Area conversions (to square meters)
        self.area_conversions = {
            "m2": 1.0,
            "square_meter": 1.0,
            "km2": 1000000.0,
            "square_kilometer": 1000000.0,
            "cm2": 0.0001,
            "square_centimeter": 0.0001,
            "square_foot": 0.092903,
            "ft2": 0.092903,
            "square_inch": 0.00064516,
            "in2": 0.00064516,
            "square_yard": 0.836127,
            "yd2": 0.836127,
            "square_mile": 2589988.11,
            "mi2": 2589988.11,
            "acre": 4046.86,
            "hectare": 10000.0
        }
        
        # Temperature conversions (special case)
        self.temperature_conversions = {
            "celsius": "celsius",
            "c": "celsius",
            "fahrenheit": "fahrenheit",
            "f": "fahrenheit",
            "kelvin": "kelvin",
            "k": "kelvin"
        }
        
        # Currency conversions (to USD - these would typically be updated regularly)
        self.currency_conversions = {
            "usd": 1.0,
            "eur": 1.09,  # Example rate
            "gbp": 1.28,  # Example rate
            "jpy": 0.0067,  # Example rate
            "cny": 0.14,  # Example rate
            "inr": 0.012,  # Example rate
            "cad": 0.74,  # Example rate
            "aud": 0.66,  # Example rate
            "chf": 1.11,  # Example rate
            "mxn": 0.049  # Example rate
        }
        
        # Time conversions (to seconds)
        self.time_conversions = {
            "s": 1.0,
            "sec": 1.0,
            "second": 1.0,
            "min": 60.0,
            "minute": 60.0,
            "h": 3600.0,
            "hr": 3600.0,
            "hour": 3600.0,
            "d": 86400.0,
            "day": 86400.0,
            "week": 604800.0,
            "month": 2592000.0,  # Approximation (30 days)
            "year": 31536000.0,  # Approximation (365 days)
        }

    def convert_weight(self, value: float, from_unit: str, to_unit: str = "kg") -> float:
        """
        Convert weight from one unit to another.
        
        Args:
            value: The weight value to convert
            from_unit: The source unit
            to_unit: The target unit (defaults to kg)
            
        Returns:
            The converted weight value
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit not in self.weight_conversions:
            logger.warning(f"Unknown weight unit: {from_unit}")
            return value
        
        if to_unit not in self.weight_conversions:
            logger.warning(f"Unknown weight unit: {to_unit}")
            return value
        
        # Convert to standard unit (kg)
        kg_value = value * self.weight_conversions[from_unit]
        
        # Convert from standard unit to target unit
        return kg_value / self.weight_conversions[to_unit]
    
    def convert_distance(self, value: float, from_unit: str, to_unit: str = "m") -> float:
        """
        Convert distance from one unit to another.
        
        Args:
            value: The distance value to convert
            from_unit: The source unit
            to_unit: The target unit (defaults to meters)
            
        Returns:
            The converted distance value
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit not in self.distance_conversions:
            logger.warning(f"Unknown distance unit: {from_unit}")
            return value
        
        if to_unit not in self.distance_conversions:
            logger.warning(f"Unknown distance unit: {to_unit}")
            return value
        
        # Convert to standard unit (meters)
        m_value = value * self.distance_conversions[from_unit]
        
        # Convert from standard unit to target unit
        return m_value / self.distance_conversions[to_unit]
    
    def convert_volume(self, value: float, from_unit: str, to_unit: str = "l") -> float:
        """
        Convert volume from one unit to another.
        
        Args:
            value: The volume value to convert
            from_unit: The source unit
            to_unit: The target unit (defaults to liters)
            
        Returns:
            The converted volume value
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit not in self.volume_conversions:
            logger.warning(f"Unknown volume unit: {from_unit}")
            return value
        
        if to_unit not in self.volume_conversions:
            logger.warning(f"Unknown volume unit: {to_unit}")
            return value
        
        # Convert to standard unit (liters)
        l_value = value * self.volume_conversions[from_unit]
        
        # Convert from standard unit to target unit
        return l_value / self.volume_conversions[to_unit]
    
    def convert_area(self, value: float, from_unit: str, to_unit: str = "m2") -> float:
        """
        Convert area from one unit to another.
        
        Args:
            value: The area value to convert
            from_unit: The source unit
            to_unit: The target unit (defaults to square meters)
            
        Returns:
            The converted area value
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit not in self.area_conversions:
            logger.warning(f"Unknown area unit: {from_unit}")
            return value
        
        if to_unit not in self.area_conversions:
            logger.warning(f"Unknown area unit: {to_unit}")
            return value
        
        # Convert to standard unit (square meters)
        m2_value = value * self.area_conversions[from_unit]
        
        # Convert from standard unit to target unit
        return m2_value / self.area_conversions[to_unit]
    
    def convert_temperature(self, value: float, from_unit: str, to_unit: str = "celsius") -> float:
        """
        Convert temperature from one unit to another.
        
        Args:
            value: The temperature value to convert
            from_unit: The source unit
            to_unit: The target unit (defaults to celsius)
            
        Returns:
            The converted temperature value
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        from_unit = self.temperature_conversions.get(from_unit, from_unit)
        to_unit = self.temperature_conversions.get(to_unit, to_unit)
        
        # Convert to Celsius first
        if from_unit == "fahrenheit":
            celsius_value = (value - 32) * 5/9
        elif from_unit == "kelvin":
            celsius_value = value - 273.15
        elif from_unit == "celsius":
            celsius_value = value
        else:
            logger.warning(f"Unknown temperature unit: {from_unit}")
            return value
        
        # Convert from Celsius to target unit
        if to_unit == "fahrenheit":
            return celsius_value * 9/5 + 32
        elif to_unit == "kelvin":
            return celsius_value + 273.15
        elif to_unit == "celsius":
            return celsius_value
        else:
            logger.warning(f"Unknown temperature unit: {to_unit}")
            return value
    
    def convert_currency(self, value: float, from_unit: str, to_unit: str = "usd") -> float:
        """
        Convert currency from one unit to another.
        
        Args:
            value: The currency value to convert
            from_unit: The source currency code
            to_unit: The target currency code (defaults to USD)
            
        Returns:
            The converted currency value
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit not in self.currency_conversions:
            logger.warning(f"Unknown currency: {from_unit}")
            return value
        
        if to_unit not in self.currency_conversions:
            logger.warning(f"Unknown currency: {to_unit}")
            return value
        
        # Convert to standard unit (USD)
        usd_value = value * self.currency_conversions[from_unit]
        
        # Convert from standard unit to target unit
        return usd_value / self.currency_conversions[to_unit]
    
    def convert_time(self, value: float, from_unit: str, to_unit: str = "s") -> float:
        """
        Convert time from one unit to another.
        
        Args:
            value: The time value to convert
            from_unit: The source unit
            to_unit: The target unit (defaults to seconds)
            
        Returns:
            The converted time value
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit not in self.time_conversions:
            logger.warning(f"Unknown time unit: {from_unit}")
            return value
        
        if to_unit not in self.time_conversions:
            logger.warning(f"Unknown time unit: {to_unit}")
            return value
        
        # Convert to standard unit (seconds)
        s_value = value * self.time_conversions[from_unit]
        
        # Convert from standard unit to target unit
        return s_value / self.time_conversions[to_unit]
    
    def extract_unit(self, text: str) -> Tuple[float, str]:
        """
        Extract numeric value and unit from text string.
        
        Args:
            text: The input text (e.g., "5.2 kg", "10 meters")
            
        Returns:
            A tuple of (value, unit)
        """
        pattern = r'(-?\d+\.?\d*)\s*([a-zA-Z0-9_]+)'
        match = re.search(pattern, text)
        
        if match:
            value_str, unit = match.groups()
            try:
                value = float(value_str)
                return value, unit.lower()
            except ValueError:
                logger.warning(f"Failed to parse value from: {text}")
                return None, None
        else:
            logger.warning(f"Failed to extract unit from: {text}")
            return None, None
    
    def standardize_column(self, df: pd.DataFrame, column: str, target_unit: str) -> pd.DataFrame:
        """
        Standardize a column in a dataframe to a target unit.
        
        Args:
            df: The input dataframe
            column: The column to standardize
            target_unit: The target unit
            
        Returns:
            The dataframe with standardized column
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found in dataframe")
            return df
        
        result_df = df.copy()
        unit_type = self._detect_unit_type(target_unit)
        
        if unit_type is None:
            logger.warning(f"Could not determine unit type for: {target_unit}")
            return df
        
        # Create a new standardized column
        standardized_column = f"{column}_standardized"
        result_df[standardized_column] = None
        
        # Process each row
        for idx, row in df.iterrows():
            value_str = str(row[column])
            value, unit = self.extract_unit(value_str)
            
            if value is not None and unit is not None:
                # Get the appropriate conversion method
                convert_method = getattr(self, f"convert_{unit_type}", None)
                
                if convert_method:
                    try:
                        converted_value = convert_method(value, unit, target_unit)
                        result_df.at[idx, standardized_column] = converted_value
                    except Exception as e:
                        logger.error(f"Error converting {value} {unit} to {target_unit}: {e}")
                else:
                    logger.warning(f"No conversion method for unit type: {unit_type}")
        
        return result_df
    
    def _detect_unit_type(self, unit: str) -> Optional[str]:
        """
        Detect the type of unit (weight, distance, etc.)
        
        Args:
            unit: The unit to detect
            
        Returns:
            The unit type or None if not found
        """
        unit = unit.lower()
        
        if unit in self.weight_conversions:
            return "weight"
        elif unit in self.distance_conversions:
            return "distance"
        elif unit in self.volume_conversions:
            return "volume"
        elif unit in self.area_conversions:
            return "area"
        elif unit in self.temperature_conversions:
            return "temperature"
        elif unit in self.currency_conversions:
            return "currency"
        elif unit in self.time_conversions:
            return "time"
        else:
            return None
    
    def batch_convert(self, data: List[Dict[str, Any]], 
                     conversions: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Batch convert multiple fields in a list of dictionaries.
        
        Args:
            data: List of dictionaries containing data
            conversions: Dictionary mapping field names to their conversion specs
                         e.g., {"weight": {"from_unit": "lb", "to_unit": "kg"}}
            
        Returns:
            Converted data
        """
        result = []
        
        for item in data:
            converted_item = item.copy()
            
            for field, conversion in conversions.items():
                if field in item and item[field] is not None:
                    from_unit = conversion.get("from_unit")
                    to_unit = conversion.get("to_unit")
                    
                    if from_unit and to_unit:
                        unit_type = self._detect_unit_type(to_unit)
                        
                        if unit_type:
                            convert_method = getattr(self, f"convert_{unit_type}", None)
                            
                            if convert_method:
                                try:
                                    value = float(item[field])
                                    converted_value = convert_method(value, from_unit, to_unit)
                                    converted_item[field] = converted_value
                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Error converting {field}: {e}")
            
            result.append(converted_item)
        
        return result
    
    def update_currency_rates(self, rates: Dict[str, float]) -> None:
        """
        Update currency conversion rates.
        
        Args:
            rates: Dictionary mapping currency codes to their rates against USD
        """
        for currency, rate in rates.items():
            currency = currency.lower()
            if currency in self.currency_conversions:
                self.currency_conversions[currency] = rate
            else:
                logger.info(f"Adding new currency: {currency}")
                self.currency_conversions[currency] = rate