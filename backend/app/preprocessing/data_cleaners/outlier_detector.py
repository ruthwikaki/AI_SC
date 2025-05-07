# app/preprocessing/data_cleaners/outlier_detector.py

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np
from scipy import stats

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class OutlierDetector:
    """
    Detects and handles outliers in datasets using various methods.
    
    This class provides methods to identify and process outliers in
    supply chain data using statistical techniques.
    """
    
    def __init__(self):
        """Initialize the outlier detector."""
        # Define accepted methods
        self.detection_methods = [
            "z_score", "iqr", "percentile", "isolation_forest", 
            "lof", "dbscan", "mad"
        ]
        
        # Define accepted handling methods
        self.handling_methods = [
            "remove", "cap", "replace_mean", "replace_median", 
            "replace_mode", "impute", "none"
        ]
        
        # Track detected outliers
        self.outliers_detected = {}
        self.modified_columns = []
        
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "z_score",
        threshold: float = 3.0,
        contamination: float = 0.05,
        return_mask: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect outliers in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            columns: Optional list of columns to check (defaults to all numeric columns)
            method: Detection method to use
            threshold: Threshold for outlier detection (interpretation depends on method)
            contamination: Expected proportion of outliers (for ML-based methods)
            return_mask: Whether to return boolean masks for outliers
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary with outlier statistics
        """
        # Reset tracking
        self.outliers_detected = {}
        
        # Check if DataFrame is empty
        if df.empty:
            return {
                "total_outliers": 0,
                "total_values": 0,
                "outlier_rate": 0.0,
                "columns": {}
            }
            
        # Select columns to analyze
        if columns is None:
            # Default to numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            columns = numeric_cols
            
        # Check if we have columns to analyze
        if not columns:
            return {
                "total_outliers": 0,
                "total_values": 0,
                "outlier_rate": 0.0,
                "columns": {}
            }
            
        # Initialize results
        column_stats = {}
        total_outliers = 0
        total_values = 0
        masks = {}
        
        # Analyze each column
        for column in columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[column].dtype):
                logger.warning(f"Column '{column}' is not numeric, skipping outlier detection")
                continue
                
            # Skip columns with all NaNs
            if df[column].isna().all():
                continue
                
            # Get non-NA values
            series = df[column].dropna()
            if len(series) == 0:
                continue
                
            # Apply detection method
            if method == "z_score":
                mask = self._z_score_method(series, threshold)
            elif method == "iqr":
                mask = self._iqr_method(series, threshold)
            elif method == "percentile":
                lower = kwargs.get("lower_percentile", 0.01)
                upper = kwargs.get("upper_percentile", 0.99)
                mask = self._percentile_method(series, lower, upper)
            elif method == "mad":
                mask = self._mad_method(series, threshold)
            elif method == "isolation_forest":
                mask = self._isolation_forest_method(series, contamination)
            elif method == "lof":
                mask = self._lof_method(series, contamination)
            elif method == "dbscan":
                eps = kwargs.get("eps", 0.5)
                min_samples = kwargs.get("min_samples", 5)
                mask = self._dbscan_method(series, eps, min_samples)
            else:
                logger.warning(f"Unknown detection method '{method}', using Z-score instead")
                mask = self._z_score_method(series, threshold)
                
            # Convert index-based mask to original DataFrame index
            full_mask = pd.Series(False, index=df.index)
            full_mask.loc[series.index] = mask
            
            # Store mask if requested
            if return_mask:
                masks[column] = full_mask
                
            # Compute statistics
            outlier_indices = series.index[mask]
            outlier_values = series[mask].tolist()
            n_outliers = sum(mask)
            n_values = len(series)
            outlier_rate = n_outliers / n_values if n_values > 0 else 0
            
            # Store for later use
            self.outliers_detected[column] = {
                "indices": outlier_indices,
                "values": outlier_values,
                "mask": full_mask
            }
            
            # Store statistics
            column_stats[column] = {
                "outlier_count": int(n_outliers),
                "total_count": int(n_values),
                "outlier_rate": float(outlier_rate),
                "min_value": float(series.min()),
                "max_value": float(series.max()),
                "outlier_min": float(series[mask].min()) if n_outliers > 0 else None,
                "outlier_max": float(series[mask].max()) if n_outliers > 0 else None,
                "method": method
            }
            
            # Update totals
            total_outliers += n_outliers
            total_values += n_values
            
        # Compute overall statistics
        outlier_rate = total_outliers / total_values if total_values > 0 else 0
            
        result = {
            "total_outliers": int(total_outliers),
            "total_values": int(total_values),
            "outlier_rate": float(outlier_rate),
            "columns": column_stats
        }
        
        if return_mask:
            result["masks"] = masks
            
        return result
        
    def handle_outliers(
        self,
        df: pd.DataFrame,
        methods: Optional[Dict[str, str]] = None,
        detection_method: str = "z_score",
        threshold: float = 3.0,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Detect and handle outliers in a DataFrame.
        
        Args:
            df: DataFrame to process
            methods: Dictionary mapping column names to handling methods
            detection_method: Method to use for detection
            threshold: Threshold for outlier detection
            columns: Optional list of columns to process
            **kwargs: Additional parameters for specific methods
            
        Returns:
            DataFrame with outliers handled
        """
        # Reset modified columns
        self.modified_columns = []
        
        # Make a copy of the DataFrame
        result_df = df.copy()
        
        if result_df.empty:
            return result_df
            
        # Detect outliers
        detection_result = self.detect_outliers(
            result_df, 
            columns=columns,
            method=detection_method,
            threshold=threshold,
            **kwargs
        )
        
        # Initialize methods
        methods = methods or {}
        default_method = kwargs.get("default_method", "cap")
        
        # Process each column with outliers
        for column, stats in detection_result["columns"].items():
            # Skip columns with no outliers
            if stats["outlier_count"] == 0:
                continue
                
            # Get outlier mask and indices
            outlier_info = self.outliers_detected.get(column, {})
            mask = outlier_info.get("mask", pd.Series(False, index=result_df.index))
            
            # Determine handling method
            method = methods.get(column, default_method)
            
            # Apply handling method
            try:
                result_df = self._apply_handling_method(
                    result_df, 
                    column, 
                    mask, 
                    method,
                    **kwargs
                )
                self.modified_columns.append(column)
            except Exception as e:
                logger.warning(f"Error handling outliers in column '{column}': {str(e)}")
                
        # Log statistics
        logger.info(f"Handled outliers in {len(self.modified_columns)} columns")
        return result_df
        
    def _z_score_method(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using Z-score method.
        
        Args:
            series: Data series
            threshold: Z-score threshold
            
        Returns:
            Boolean mask of outliers
        """
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        return pd.Series(z_scores > threshold, index=series.index)
        
    def _iqr_method(self, series: pd.Series, factor: float = 1.5) -> pd.Series:
        """
        Detect outliers using IQR method.
        
        Args:
            series: Data series
            factor: IQR factor (typically 1.5)
            
        Returns:
            Boolean mask of outliers
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        return (series < lower_bound) | (series > upper_bound)
        
    def _percentile_method(
        self,
        series: pd.Series,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99
    ) -> pd.Series:
        """
        Detect outliers using percentile method.
        
        Args:
            series: Data series
            lower_percentile: Lower percentile bound
            upper_percentile: Upper percentile bound
            
        Returns:
            Boolean mask of outliers
        """
        lower_bound = series.quantile(lower_percentile)
        upper_bound = series.quantile(upper_percentile)
        return (series < lower_bound) | (series > upper_bound)
        
    def _mad_method(self, series: pd.Series, threshold: float = 3.5) -> pd.Series:
        """
        Detect outliers using Median Absolute Deviation method.
        
        Args:
            series: Data series
            threshold: MAD threshold
            
        Returns:
            Boolean mask of outliers
        """
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        # Adjust MAD for normal distribution
        mad_adjusted = mad * 1.4826
        
        z_scores = np.abs((series - median) / mad_adjusted)
        return z_scores > threshold
        
    def _isolation_forest_method(
        self,
        series: pd.Series,
        contamination: float = 0.05
    ) -> pd.Series:
        """
        Detect outliers using Isolation Forest.
        
        Args:
            series: Data series
            contamination: Expected proportion of outliers
            
        Returns:
            Boolean mask of outliers
        """
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            # Train model
            model = IsolationForest(
                contamination=contamination, 
                random_state=42
            )
            
            # Predict outliers
            # IsolationForest: -1 for outliers, 1 for inliers
            predictions = model.fit_predict(X)
            
            return pd.Series(predictions == -1, index=series.index)
            
        except ImportError:
            logger.warning("scikit-learn not available, falling back to IQR method")
            return self._iqr_method(series)
            
    def _lof_method(
        self,
        series: pd.Series,
        contamination: float = 0.05
    ) -> pd.Series:
        """
        Detect outliers using Local Outlier Factor.
        
        Args:
            series: Data series
            contamination: Expected proportion of outliers
            
        Returns:
            Boolean mask of outliers
        """
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            # Train model
            model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination
            )
            
            # Predict outliers
            # LOF: -1 for outliers, 1 for inliers
            predictions = model.fit_predict(X)
            
            return pd.Series(predictions == -1, index=series.index)
            
        except ImportError:
            logger.warning("scikit-learn not available, falling back to IQR method")
            return self._iqr_method(series)
            
    def _dbscan_method(
        self,
        series: pd.Series,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> pd.Series:
        """
        Detect outliers using DBSCAN clustering.
        
        Args:
            series: Data series
            eps: Maximum distance between points
            min_samples: Minimum points for a core point
            
        Returns:
            Boolean mask of outliers
        """
        try:
            from sklearn.cluster import DBSCAN
            
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            # Train model
            model = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )
            
            # Predict clusters
            # DBSCAN: -1 for outliers, cluster numbers for inliers
            clusters = model.fit_predict(X)
            
            return pd.Series(clusters == -1, index=series.index)
            
        except ImportError:
            logger.warning("scikit-learn not available, falling back to IQR method")
            return self._iqr_method(series)
            
    def _apply_handling_method(
        self,
        df: pd.DataFrame,
        column: str,
        mask: pd.Series,
        method: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Apply a handling method to outliers.
        
        Args:
            df: DataFrame to modify
            column: Column name
            mask: Boolean mask of outliers
            method: Handling method name
            **kwargs: Additional parameters
            
        Returns:
            Modified DataFrame
        """
        # Make a copy for safety
        result = df.copy()
        series = result[column]
        
        # Get non-outlier values
        non_outliers = series[~mask]
        
        # Apply method
        if method == "remove":
            # Remove rows with outliers
            result = result[~mask]
            
        elif method == "cap":
            # Cap outliers at percentile values
            lower_percentile = kwargs.get("lower_percentile", 0.01)
            upper_percentile = kwargs.get("upper_percentile", 0.99)
            
            lower_bound = series.quantile(lower_percentile)
            upper_bound = series.quantile(upper_percentile)
            
            # Apply caps
            result.loc[mask & (series < lower_bound), column] = lower_bound
            result.loc[mask & (series > upper_bound), column] = upper_bound
            
        elif method == "replace_mean":
            # Replace with mean of non-outliers
            mean_value = non_outliers.mean()
            result.loc[mask, column] = mean_value
            
        elif method == "replace_median":
            # Replace with median of non-outliers
            median_value = non_outliers.median()
            result.loc[mask, column] = median_value
            
        elif method == "replace_mode":
            # Replace with mode of non-outliers
            mode_values = non_outliers.mode()
            if len(mode_values) > 0:
                mode_value = mode_values[0]
                result.loc[mask, column] = mode_value
                
        elif method == "impute":
            # Use more sophisticated imputation
            try:
                # Try KNN imputation if available
                from sklearn.impute import KNNImputer
                
                # Create a temporary column with outliers set to NaN
                temp_col = f"{column}_temp"
                result[temp_col] = series.copy()
                result.loc[mask, temp_col] = np.nan
                
                # Prepare imputer
                imputer = KNNImputer(n_neighbors=5)
                
                # Get numeric columns for imputation context
                numeric_cols = result.select_dtypes(include=np.number).columns.tolist()
                
                # Remove the column itself and the temp column from context
                context_cols = [col for col in numeric_cols if col != column and col != temp_col]
                
                # If we have context columns, use them for imputation
                if context_cols:
                    # Prepare data for imputation
                    impute_data = result[[temp_col] + context_cols].copy()
                    
                    # Apply imputation
                    imputed_data = imputer.fit_transform(impute_data)
                    
                    # Update the original column with imputed values
                    imputed_series = pd.Series(imputed_data[:, 0], index=result.index)
                    result.loc[mask, column] = imputed_series[mask]
                else:
                    # If no context columns, fall back to median
                    median_value = non_outliers.median()
                    result.loc[mask, column] = median_value
                    
                # Remove temporary column
                result = result.drop(columns=[temp_col])
                
            except ImportError:
                # Fall back to median if sklearn not available
                logger.warning("scikit-learn not available, falling back to median replacement")
                median_value = non_outliers.median()
                result.loc[mask, column] = median_value
                
        elif method == "none":
            # No handling, just keep the outliers
            pass
            
        else:
            logger.warning(f"Unknown handling method '{method}', no changes applied")
            
        return result
        
    def visualize_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        detection_method: str = "z_score",
        threshold: float = 3.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate visualization data for outliers.
        
        Args:
            df: DataFrame to analyze
            column: Column to visualize
            detection_method: Method for outlier detection
            threshold: Detection threshold
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with visualization data
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import io
            import base64
            
            # Detect outliers
            detection_result = self.detect_outliers(
                df,
                columns=[column],
                method=detection_method,
                threshold=threshold,
                return_mask=True,
                **kwargs
            )
            
            mask = detection_result.get("masks", {}).get(column, pd.Series(False, index=df.index))
            
            # Create visualization data
            # Create a figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Boxplot
            sns.boxplot(y=df[column], ax=ax1)
            ax1.set_title(f"Boxplot of {column}")
            
            # Histogram with outliers highlighted
            sns.histplot(df.loc[~mask, column], ax=ax2, color='blue', alpha=0.5, label='Normal')
            if mask.any():
                sns.histplot(df.loc[mask, column], ax=ax2, color='red', alpha=0.5, label='Outliers')
            ax2.set_title(f"Histogram of {column}")
            ax2.legend()
            
            # Convert to base64 image
            buffer = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Get stats for visualization
            stats = detection_result["columns"].get(column, {})
            
            return {
                "column": column,
                "image": f"data:image/png;base64,{image_base64}",
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"Error generating outlier visualization: {str(e)}")
            return {
                "column": column,
                "error": str(e)
            }
            
    def get_modified_columns(self) -> List[str]:
        """
        Get list of columns that were modified.
        
        Returns:
            List of column names
        """
        return self.modified_columns