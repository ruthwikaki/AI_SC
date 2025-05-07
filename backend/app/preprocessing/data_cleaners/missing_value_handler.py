# app/preprocessing/data_cleaners/missing_value_handler.py

from typing import Dict, List, Any, Optional, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class MissingValueHandler:
    """
    Handles missing values in datasets using various strategies.
    
    This class provides methods to detect and handle missing values
    in supply chain datasets using methods appropriate for each data type.
    """
    
    def __init__(self):
        """Initialize the missing value handler."""
        # Define accepted strategies
        self.numeric_strategies = [
            "mean", "median", "mode", "zero", "constant", 
            "interpolate", "forward_fill", "backward_fill", "drop"
        ]
        
        self.categorical_strategies = [
            "mode", "constant", "most_frequent", "new_category", 
            "forward_fill", "backward_fill", "drop"
        ]
        
        self.datetime_strategies = [
            "interpolate", "forward_fill", "backward_fill", 
            "constant", "drop"
        ]
        
        # Track columns that were modified
        self.modified_columns = []
        
    def detect_missing(
        self,
        df: pd.DataFrame,
        min_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Detect missing values in a DataFrame.
        
        Args:
            df: DataFrame to check
            min_threshold: Minimum threshold of missing values to report (0-1)
            
        Returns:
            Dictionary with missing value statistics
        """
        # Check if DataFrame is empty
        if df.empty:
            return {
                "total_missing": 0,
                "total_cells": 0,
                "missing_rate": 0.0,
                "columns": {}
            }
            
        # Calculate missing values
        missing = df.isna().sum()
        total_missing = missing.sum()
        total_cells = df.size
        missing_rate = total_missing / total_cells
        
        # Get missing rates per column
        column_stats = {}
        for column in df.columns:
            col_missing = df[column].isna().sum()
            col_total = len(df)
            col_rate = col_missing / col_total
            
            if col_rate >= min_threshold:
                column_stats[column] = {
                    "missing_count": int(col_missing),
                    "total_count": int(col_total),
                    "missing_rate": float(col_rate),
                    "data_type": str(df[column].dtype),
                    "recommended_strategy": self._recommend_strategy(df[column])
                }
                
        return {
            "total_missing": int(total_missing),
            "total_cells": int(total_cells),
            "missing_rate": float(missing_rate),
            "columns": column_stats
        }
        
    def handle_missing(
        self,
        df: pd.DataFrame,
        strategies: Optional[Dict[str, str]] = None,
        default_strategy: str = "auto",
        min_threshold: float = 0.0,
        constant_values: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in a DataFrame.
        
        Args:
            df: DataFrame to process
            strategies: Optional dictionary mapping column names to strategies
            default_strategy: Default strategy to use if not specified ("auto" for automatic selection)
            min_threshold: Minimum threshold of missing values to handle (0-1)
            constant_values: Optional dictionary of constant values to use for 'constant' strategy
            
        Returns:
            Processed DataFrame with missing values handled
        """
        # Reset modified columns
        self.modified_columns = []
        
        # Make a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        if result_df.empty:
            return result_df
            
        # Get columns with missing values
        missing_stats = self.detect_missing(result_df, min_threshold)
        
        # Initialize strategy map
        strategies = strategies or {}
        constant_values = constant_values or {}
        
        # Process each column with missing values
        for column, stats in missing_stats["columns"].items():
            # Skip columns with no missing values or below threshold
            if stats["missing_count"] == 0 or stats["missing_rate"] < min_threshold:
                continue
                
            # Determine strategy
            strategy = strategies.get(column, default_strategy)
            
            # If 'auto', determine appropriate strategy based on data type
            if strategy == "auto":
                strategy = stats["recommended_strategy"]
                
            # Get constant value if using 'constant' strategy
            constant_value = constant_values.get(column, None)
            
            # Apply the strategy
            try:
                result_df = self._apply_strategy(
                    result_df, 
                    column, 
                    strategy, 
                    constant_value=constant_value
                )
                self.modified_columns.append(column)
            except Exception as e:
                logger.warning(f"Error handling missing values in column '{column}': {str(e)}")
                
        # Log statistics
        logger.info(f"Handled missing values in {len(self.modified_columns)} columns")
        return result_df
        
    def _recommend_strategy(self, series: pd.Series) -> str:
        """
        Recommend a strategy for handling missing values in a series.
        
        Args:
            series: Series to analyze
            
        Returns:
            Recommended strategy name
        """
        # Determine data type
        data_type = series.dtype
        missing_rate = series.isna().sum() / len(series)
        
        # If missing rate is very high, recommend dropping
        if missing_rate > 0.5:
            return "drop"
            
        # For numeric data
        if pd.api.types.is_numeric_dtype(data_type):
            # For integer columns, mode is often better than mean to maintain integer nature
            if pd.api.types.is_integer_dtype(data_type):
                # Check if there's a clear mode
                mode_values = series.mode()
                if len(mode_values) == 1:
                    return "mode"
                else:
                    return "median"  # Median preserves integer nature better than mean
            else:
                # For float columns, mean is typically good
                return "mean"
                
        # For categorical data
        elif pd.api.types.is_string_dtype(data_type) or pd.api.types.is_categorical_dtype(data_type):
            # Mode for categorical data
            mode_values = series.mode()
            if len(mode_values) == 1:
                return "mode"
            else:
                return "new_category"
                
        # For datetime data
        elif pd.api.types.is_datetime64_dtype(data_type):
            # Interpolation works well for time series
            return "interpolate"
            
        # Default for other data types
        return "forward_fill"
        
    def _apply_strategy(
        self,
        df: pd.DataFrame,
        column: str,
        strategy: str,
        constant_value: Any = None
    ) -> pd.DataFrame:
        """
        Apply a missing value strategy to a column.
        
        Args:
            df: DataFrame to modify
            column: Column name to process
            strategy: Strategy name to apply
            constant_value: Value to use for 'constant' strategy
            
        Returns:
            Modified DataFrame
        """
        # Make a copy for safety
        result = df.copy()
        series = result[column]
        
        # Determine data type
        is_numeric = pd.api.types.is_numeric_dtype(series.dtype)
        is_categorical = pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_categorical_dtype(series.dtype)
        is_datetime = pd.api.types.is_datetime64_dtype(series.dtype)
        
        # Apply strategy based on type
        if is_numeric:
            # Check if strategy is valid for numeric data
            if strategy not in self.numeric_strategies:
                logger.warning(f"Strategy '{strategy}' not valid for numeric column '{column}'. Using 'mean' instead.")
                strategy = "mean"
                
            # Apply strategy
            if strategy == "mean":
                result[column] = series.fillna(series.mean())
            elif strategy == "median":
                result[column] = series.fillna(series.median())
            elif strategy == "mode":
                mode_value = series.mode()
                if len(mode_value) > 0:
                    result[column] = series.fillna(mode_value[0])
            elif strategy == "zero":
                result[column] = series.fillna(0)
            elif strategy == "constant":
                result[column] = series.fillna(constant_value or 0)
            elif strategy == "interpolate":
                result[column] = series.interpolate(method="linear")
            elif strategy == "forward_fill":
                result[column] = series.ffill()
            elif strategy == "backward_fill":
                result[column] = series.bfill()
            elif strategy == "drop":
                result = result.dropna(subset=[column])
                
        elif is_categorical:
            # Check if strategy is valid for categorical data
            if strategy not in self.categorical_strategies:
                logger.warning(f"Strategy '{strategy}' not valid for categorical column '{column}'. Using 'mode' instead.")
                strategy = "mode"
                
            # Apply strategy
            if strategy == "mode" or strategy == "most_frequent":
                mode_value = series.mode()
                if len(mode_value) > 0:
                    result[column] = series.fillna(mode_value[0])
            elif strategy == "constant":
                result[column] = series.fillna(constant_value or "unknown")
            elif strategy == "new_category":
                result[column] = series.fillna("unknown")
            elif strategy == "forward_fill":
                result[column] = series.ffill()
            elif strategy == "backward_fill":
                result[column] = series.bfill()
            elif strategy == "drop":
                result = result.dropna(subset=[column])
                
        elif is_datetime:
            # Check if strategy is valid for datetime data
            if strategy not in self.datetime_strategies:
                logger.warning(f"Strategy '{strategy}' not valid for datetime column '{column}'. Using 'interpolate' instead.")
                strategy = "interpolate"
                
            # Apply strategy
            if strategy == "interpolate":
                result[column] = series.interpolate(method="time")
            elif strategy == "forward_fill":
                result[column] = series.ffill()
            elif strategy == "backward_fill":
                result[column] = series.bfill()
            elif strategy == "constant":
                if constant_value is None:
                    constant_value = datetime.now()
                result[column] = series.fillna(constant_value)
            elif strategy == "drop":
                result = result.dropna(subset=[column])
                
        else:
            # For unknown types, use forward fill as a safe default
            logger.warning(f"Unrecognized data type for column '{column}'. Using forward fill as default.")
            result[column] = series.ffill()
            
        return result
        
    def handle_missing_with_custom_func(
        self,
        df: pd.DataFrame,
        column: str,
        func: Callable[[pd.Series], pd.Series]
    ) -> pd.DataFrame:
        """
        Handle missing values using a custom function.
        
        Args:
            df: DataFrame to process
            column: Column name to process
            func: Function that takes a Series and returns a Series with missing values filled
            
        Returns:
            Modified DataFrame
        """
        # Make a copy
        result = df.copy()
        
        try:
            # Apply custom function
            result[column] = func(result[column])
            self.modified_columns.append(column)
        except Exception as e:
            logger.error(f"Error applying custom function to column '{column}': {str(e)}")
            
        return result
        
    def get_modified_columns(self) -> List[str]:
        """
        Get list of columns that were modified.
        
        Returns:
            List of column names
        """
        return self.modified_columns