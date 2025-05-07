# app/preprocessing/normalization/date_normalizer.py

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import re
from datetime import datetime, timezone
import dateutil.parser
import pytz

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class DateNormalizer:
    """
    Normalizes and standardizes date formats in datasets.
    
    This class provides methods to detect date formats and convert
    them to standardized formats for consistent analysis.
    """
    
    def __init__(self):
        """Initialize the date normalizer."""
        # Define common date formats
        self.date_formats = {
            "iso": "%Y-%m-%d",
            "iso_datetime": "%Y-%m-%dT%H:%M:%S",
            "us_date": "%m/%d/%Y",
            "eu_date": "%d/%m/%Y",
            "us_datetime": "%m/%d/%Y %H:%M:%S",
            "eu_datetime": "%d/%m/%Y %H:%M:%S",
            "year_month": "%Y-%m",
            "date_only": "%Y%m%d",
            "slash_date": "%Y/%m/%d",
            "dot_date": "%Y.%m.%d",
            "dash_date": "%Y-%m-%d",
            "text_date": "%B %d, %Y",
            "short_text_date": "%b %d, %Y"
        }
        
        # Date format patterns for detection
        self.format_patterns = {
            r"^\d{4}-\d{2}-\d{2}$": "iso",
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}": "iso_datetime",
            r"^\d{1,2}/\d{1,2}/\d{4}$": "us_date",
            r"^\d{1,2}/\d{1,2}/\d{2}$": "us_short_date",
            r"^\d{1,2}\.\d{1,2}\.\d{4}$": "dot_date",
            r"^\d{1,2}\.\d{1,2}\.\d{2}$": "dot_short_date",
            r"^\d{4}/\d{1,2}/\d{1,2}$": "slash_date",
            r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$": "us_datetime",
            r"^\d{2}/\d{2}/\d{4} \d{1,2}:\d{2}$": "us_datetime_short",
            r"^\d{4}$": "year_only",
            r"^\d{4}-\d{2}$": "year_month",
            r"^\d{8}$": "date_only",
            r"^[A-Za-z]+ \d{1,2}, \d{4}$": "text_date",
            r"^[A-Za-z]{3} \d{1,2}, \d{4}$": "short_text_date"
        }
        
        # Track columns that were modified
        self.modified_columns = []
        
    def detect_date_columns(
        self,
        df: pd.DataFrame,
        sample_size: int = 100,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Detect potential date columns in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            sample_size: Number of rows to sample for detection
            threshold: Minimum percentage of successful conversions for detection
            
        Returns:
            Dictionary with detected date columns and formats
        """
        # Check if DataFrame is empty
        if df.empty:
            return {
                "date_columns": {}
            }
            
        # Sample rows if DataFrame is large
        if len(df) > sample_size:
            sample_df = df.sample(sample_size, random_state=42)
        else:
            sample_df = df
            
        # Initialize results
        date_columns = {}
        
        # Check each string column
        for column in df.columns:
            # Skip columns that are already datetime
            if pd.api.types.is_datetime64_any_dtype(df[column].dtype):
                date_columns[column] = {
                    "format": "datetime64",
                    "confidence": 1.0
                }
                continue
                
            # Skip numeric columns
            if pd.api.types.is_numeric_dtype(df[column].dtype):
                continue
                
            # Sample column values
            sample_values = sample_df[column].dropna().astype(str)
            
            if len(sample_values) == 0:
                continue
                
            # Try to detect date format
            format_scores = {}
            format_counts = {}
            
            for pattern, format_key in self.format_patterns.items():
                # Count pattern matches
                matches = sample_values.str.match(pattern)
                match_count = matches.sum()
                
                if match_count > 0:
                    format_scores[format_key] = match_count / len(sample_values)
                    format_counts[format_key] = match_count
                    
            # Detect using dateutil parser
            parser_successes = 0
            for value in sample_values:
                try:
                    dateutil.parser.parse(value)
                    parser_successes += 1
                except (ValueError, TypeError, OverflowError):
                    pass
                    
            parser_score = parser_successes / len(sample_values)
            
            # Determine the best format
            best_format = None
            best_score = 0
            
            for format_key, score in format_scores.items():
                if score > best_score:
                    best_score = score
                    best_format = format_key
                    
            # If dateutil parser performs better, use it
            if parser_score > best_score:
                best_format = "auto"
                best_score = parser_score
                
            # Add to results if confidence is above threshold
            if best_format and best_score >= threshold:
                date_columns[column] = {
                    "format": best_format,
                    "confidence": best_score,
                    "format_pattern": self.date_formats.get(best_format, "auto")
                }
                
        return {
            "date_columns": date_columns
        }
        
    def normalize_dates(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        target_format: str = "iso",
        target_timezone: Optional[str] = None,
        detect_formats: bool = True,
        format_map: Optional[Dict[str, str]] = None,
        errors: str = "coerce",
        **kwargs
    ) -> pd.DataFrame:
        """
        Normalize date columns to a consistent format.
        
        Args:
            df: DataFrame to process
            columns: Optional list of columns to normalize (None for auto-detection)
            target_format: Target date format (key in self.date_formats or format string)
            target_timezone: Target timezone for conversion
            detect_formats: Whether to auto-detect date formats
            format_map: Optional mapping of column names to date formats
            errors: How to handle errors ('ignore', 'raise', 'coerce')
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with normalized date columns
        """
        # Reset modified columns
        self.modified_columns = []
        
        # Make a copy of the DataFrame
        result_df = df.copy()
        
        if result_df.empty:
            return result_df
            
        # Get target format
        if target_format in self.date_formats:
            target_format_str = self.date_formats[target_format]
        else:
            target_format_str = target_format
            
        # Columns to process
        columns_to_process = {}
        
        # Auto-detect date columns if needed
        if detect_formats and not columns:
            detection_result = self.detect_date_columns(result_df)
            columns_to_process = detection_result["date_columns"]
        else:
            # Use provided columns and formats
            columns_to_process = {}
            
            for column in columns or []:
                if column in result_df.columns:
                    if format_map and column in format_map:
                        # Use provided format
                        format_key = format_map[column]
                        format_str = self.date_formats.get(format_key, format_key)
                        
                        columns_to_process[column] = {
                            "format": format_key,
                            "format_pattern": format_str
                        }
                    else:
                        # Auto-detect format for this column
                        column_result = self.detect_date_columns(
                            result_df[[column]],
                            threshold=0.5
                        )
                        
                        if column in column_result["date_columns"]:
                            columns_to_process[column] = column_result["date_columns"][column]
                        else:
                            # Try with auto-detection
                            columns_to_process[column] = {
                                "format": "auto",
                                "format_pattern": "auto"
                            }
                            
        # Process each column
        for column, info in columns_to_process.items():
            try:
                # Get source format
                source_format = info.get("format", "auto")
                source_pattern = info.get("format_pattern", "auto")
                
                # Try to convert to datetime
                if source_format == "datetime64":
                    # Already datetime, no conversion needed
                    datetime_series = result_df[column]
                elif source_format == "auto" or source_pattern == "auto":
                    # Use pandas to infer format
                    datetime_series = pd.to_datetime(
                        result_df[column],
                        infer_datetime_format=True,
                        errors=errors
                    )
                else:
                    # Use specific format
                    datetime_series = pd.to_datetime(
                        result_df[column],
                        format=source_pattern,
                        errors=errors
                    )
                    
                # Apply timezone conversion if needed
                if target_timezone and datetime_series.dt.tz:
                    # Convert from original timezone to target
                    datetime_series = datetime_series.dt.tz_convert(target_timezone)
                elif target_timezone and not datetime_series.dt.tz:
                    # Apply target timezone as localization
                    datetime_series = datetime_series.dt.tz_localize(target_timezone)
                    
                # Store datetime version
                result_df[column] = datetime_series
                self.modified_columns.append(column)
                
                # Convert to target string format if requested
                if kwargs.get("store_as_string", False):
                    # Convert to string with target format
                    result_df[column] = datetime_series.dt.strftime(target_format_str)
                    
            except Exception as e:
                if errors == "raise":
                    raise
                logger.warning(f"Error normalizing date column '{column}': {str(e)}")
                
        # Log statistics
        logger.info(f"Normalized {len(self.modified_columns)} date columns")
        return result_df
        
    def add_date_features(
        self,
        df: pd.DataFrame,
        date_column: str,
        features: Optional[List[str]] = None,
        prefix: Optional[str] = None,
        drop_original: bool = False
    ) -> pd.DataFrame:
        """
        Add date-based features from a date column.
        
        Args:
            df: DataFrame to process
            date_column: Date column to extract features from
            features: List of features to extract (None for all)
            prefix: Optional prefix for feature column names
            drop_original: Whether to drop the original date column
            
        Returns:
            DataFrame with added date features
        """
        # Make a copy of the DataFrame
        result_df = df.copy()
        
        if result_df.empty or date_column not in result_df.columns:
            return result_df
            
        # Available features
        all_features = [
            "year", "quarter", "month", "day", "dayofweek", "dayofyear",
            "weekofyear", "hour", "minute", "second", "is_month_start",
            "is_month_end", "is_quarter_start", "is_quarter_end",
            "is_year_start", "is_year_end", "is_weekend"
        ]
        
        # Select features to extract
        features_to_extract = features or all_features
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(result_df[date_column].dtype):
            try:
                result_df[date_column] = pd.to_datetime(result_df[date_column])
            except Exception as e:
                logger.error(f"Error converting '{date_column}' to datetime: {str(e)}")
                return result_df
                
        # Create prefix
        col_prefix = f"{prefix}_" if prefix else f"{date_column}_"
        
        # Extract features
        for feature in features_to_extract:
            if feature in all_features:
                feature_col = f"{col_prefix}{feature}"
                
                # Handle special cases
                if feature == "is_weekend":
                    # Weekend is Saturday (5) or Sunday (6)
                    result_df[feature_col] = result_df[date_column].dt.dayofweek.isin([5, 6])
                else:
                    # Use pandas datetime accessor
                    result_df[feature_col] = getattr(result_df[date_column].dt, feature)
                    
        # Drop original if requested
        if drop_original:
            result_df = result_df.drop(columns=[date_column])
            
        return result_df
        
    def create_time_period_column(
        self,
        df: pd.DataFrame,
        date_column: str,
        period: str = "month",
        target_format: Optional[str] = None,
        column_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create a column representing a time period from a date column.
        
        Args:
            df: DataFrame to process
            date_column: Date column to use
            period: Time period ('day', 'week', 'month', 'quarter', 'year')
            target_format: Optional format for the new column
            column_name: Name for the new column
            
        Returns:
            DataFrame with added time period column
        """
        # Make a copy of the DataFrame
        result_df = df.copy()
        
        if result_df.empty or date_column not in result_df.columns:
            return result_df
            
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(result_df[date_column].dtype):
            try:
                result_df[date_column] = pd.to_datetime(result_df[date_column])
            except Exception as e:
                logger.error(f"Error converting '{date_column}' to datetime: {str(e)}")
                return result_df
                
        # Determine column name
        if not column_name:
            column_name = f"{date_column}_{period}"
            
        # Create period column
        if period == "day":
            result_df[column_name] = result_df[date_column].dt.date
        elif period == "week":
            # Format as YYYY-WW
            result_df[column_name] = (
                result_df[date_column].dt.year.astype(str) + "-W" +
                result_df[date_column].dt.isocalendar().week.astype(str).str.zfill(2)
            )
        elif period == "month":
            # Format as YYYY-MM
            result_df[column_name] = (
                result_df[date_column].dt.year.astype(str) + "-" +
                result_df[date_column].dt.month.astype(str).str.zfill(2)
            )
        elif period == "quarter":
            # Format as YYYY-Q#
            result_df[column_name] = (
                result_df[date_column].dt.year.astype(str) + "-Q" +
                result_df[date_column].dt.quarter.astype(str)
            )
        elif period == "year":
            result_df[column_name] = result_df[date_column].dt.year
        else:
            logger.warning(f"Unknown period type '{period}', using 'month' instead")
            # Format as YYYY-MM
            result_df[column_name] = (
                result_df[date_column].dt.year.astype(str) + "-" +
                result_df[date_column].dt.month.astype(str).str.zfill(2)
            )
            
        # Format if specified
        if target_format and period != "day":  # Skip for day, as it's already date type
            if target_format in self.date_formats:
                format_str = self.date_formats[target_format]
            else:
                format_str = target_format
                
            try:
                # Try to apply format if column is date/datetime
                if period == "day":
                    result_df[column_name] = result_df[column_name].dt.strftime(format_str)
            except Exception as e:
                logger.warning(f"Error formatting time period column: {str(e)}")
                
        return result_df
        
    def get_modified_columns(self) -> List[str]:
        """
        Get list of columns that were modified.
        
        Returns:
            List of column names
        """
        return self.modified_columns