# app/preprocessing/data_cleaners/duplicate_remover.py

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class DuplicateRemover:
    """
    Detects and removes duplicate data in datasets.
    
    This class provides methods to identify and handle duplicate records
    in supply chain data with configurable strategies.
    """
    
    def __init__(self):
        """Initialize the duplicate remover."""
        # Define accepted strategies
        self.duplicate_strategies = [
            "keep_first", "keep_last", "keep_max", "keep_min", 
            "aggregate_mean", "aggregate_sum", "mark"
        ]
        
        # Track duplicates
        self.duplicates_detected = {}
        self.removed_indices = []
        
    def detect_duplicates(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        ignore_case: bool = False,
        fuzzy_match: bool = False,
        fuzzy_threshold: float = 0.9,
        return_groups: bool = False,
        time_window: Optional[str] = None,
        time_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect duplicates in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            columns: Columns to consider for duplicate detection (None for all)
            ignore_case: Whether to ignore case in string columns
            fuzzy_match: Whether to use fuzzy matching for string columns
            fuzzy_threshold: Threshold for fuzzy matching (0-1)
            return_groups: Whether to return duplicate groups
            time_window: Optional time window for time-based deduplication
            time_column: Column containing timestamps for time-based deduplication
            
        Returns:
            Dictionary with duplicate statistics
        """
        # Reset tracking
        self.duplicates_detected = {}
        self.removed_indices = []
        
        # Check if DataFrame is empty
        if df.empty:
            return {
                "total_duplicates": 0,
                "total_rows": 0,
                "duplicate_rate": 0.0,
                "duplicate_groups": 0
            }
            
        # Select columns to analyze
        if columns is None:
            columns = df.columns.tolist()
            
        # Check if we have columns to analyze
        if not columns:
            return {
                "total_duplicates": 0,
                "total_rows": 0,
                "duplicate_rate": 0.0,
                "duplicate_groups": 0
            }
            
        # Prepare data for duplicate detection
        data = df.copy()
        
        # Handle case sensitivity if needed
        if ignore_case:
            for col in columns:
                if pd.api.types.is_string_dtype(data[col].dtype):
                    data[col] = data[col].str.lower()
                    
        # Handle fuzzy matching if requested
        if fuzzy_match:
            # Fuzzy matching on strings only works on a subset of columns
            string_columns = [col for col in columns if pd.api.types.is_string_dtype(df[col].dtype)]
            if string_columns:
                data = self._apply_fuzzy_hashing(data, string_columns, fuzzy_threshold)
                
        # Handle time-based window if requested
        if time_window and time_column and time_column in df.columns:
            data = self._apply_time_window(data, time_column, time_window, columns)
            
        # Detect duplicates
        duplicated = data.duplicated(subset=columns, keep=False)
        duplicate_indices = data.index[duplicated].tolist()
        
        # Get duplicate groups if requested
        duplicate_groups = []
        if return_groups:
            # Group the duplicated rows
            if duplicate_indices:
                dup_df = data.loc[duplicated, columns]
                grouped = dup_df.groupby(list(columns))
                
                for _, group in grouped:
                    if len(group) > 1:  # Only consider groups with multiple rows
                        duplicate_groups.append(group.index.tolist())
                        
        # Calculate statistics
        total_duplicates = sum(duplicated)
        total_rows = len(df)
        duplicate_rate = total_duplicates / total_rows if total_rows > 0 else 0
        
        # Store for later use
        self.duplicates_detected = {
            "indices": duplicate_indices,
            "mask": duplicated,
            "groups": duplicate_groups if return_groups else []
        }
        
        result = {
            "total_duplicates": int(total_duplicates),
            "total_rows": int(total_rows),
            "duplicate_rate": float(duplicate_rate),
            "duplicate_groups": len(duplicate_groups) if return_groups else sum(duplicated) - len(duplicate_indices)
        }
        
        if return_groups:
            result["groups"] = duplicate_groups
            
        return result
        
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        strategy: str = "keep_first",
        ignore_case: bool = False,
        value_column: Optional[str] = None,
        time_window: Optional[str] = None,
        time_column: Optional[str] = None,
        mark_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Remove or handle duplicates in a DataFrame.
        
        Args:
            df: DataFrame to process
            columns: Columns to consider for duplicate detection (None for all)
            strategy: Strategy for handling duplicates
            ignore_case: Whether to ignore case in string columns
            value_column: Column to use for keep_max/keep_min strategies
            time_window: Optional time window for time-based deduplication
            time_column: Column containing timestamps for time-based deduplication
            mark_column: Column name for marking duplicates
            **kwargs: Additional parameters
            
        Returns:
            Processed DataFrame
        """
        # Reset indices tracking
        self.removed_indices = []
        
        # Make a copy of the DataFrame
        result_df = df.copy()
        
        if result_df.empty:
            return result_df
            
        # Select columns to analyze
        if columns is None:
            columns = result_df.columns.tolist()
            
        # Detect duplicates
        detection_result = self.detect_duplicates(
            result_df,
            columns=columns,
            ignore_case=ignore_case,
            fuzzy_match=kwargs.get("fuzzy_match", False),
            fuzzy_threshold=kwargs.get("fuzzy_threshold", 0.9),
            return_groups=True,
            time_window=time_window,
            time_column=time_column
        )
        
        # Check if there are duplicates to handle
        if detection_result["total_duplicates"] == 0:
            logger.info("No duplicates found, returning original DataFrame")
            return result_df
            
        # Get duplicate groups
        duplicate_groups = detection_result.get("groups", [])
        
        # Apply strategy
        if strategy == "keep_first":
            # Keep first occurrence of each duplicate group
            result_df = result_df.drop_duplicates(subset=columns, keep='first')
            self.removed_indices = df.index[df.duplicated(subset=columns, keep='first')].tolist()
            
        elif strategy == "keep_last":
            # Keep last occurrence of each duplicate group
            result_df = result_df.drop_duplicates(subset=columns, keep='last')
            self.removed_indices = df.index[df.duplicated(subset=columns, keep='last')].tolist()
            
        elif strategy in ["keep_max", "keep_min"] and value_column:
            # Keep row with max/min value in each duplicate group
            if value_column not in result_df.columns:
                logger.warning(f"Value column '{value_column}' not in DataFrame, falling back to keep_first")
                result_df = result_df.drop_duplicates(subset=columns, keep='first')
                self.removed_indices = df.index[df.duplicated(subset=columns, keep='first')].tolist()
            else:
                # Process each duplicate group
                indices_to_remove = []
                
                for group in duplicate_groups:
                    if strategy == "keep_max":
                        # Find index with maximum value
                        keep_idx = result_df.loc[group, value_column].idxmax()
                    else:  # keep_min
                        # Find index with minimum value
                        keep_idx = result_df.loc[group, value_column].idxmin()
                        
                    # Mark other indices for removal
                    indices_to_remove.extend([idx for idx in group if idx != keep_idx])
                    
                # Remove duplicates
                result_df = result_df.drop(indices_to_remove)
                self.removed_indices = indices_to_remove
                
        elif strategy in ["aggregate_mean", "aggregate_sum"]:
            # Aggregate duplicate groups
            numeric_columns = result_df.select_dtypes(include=np.number).columns.tolist()
            aggregation_columns = [col for col in numeric_columns if col not in columns]
            
            if not aggregation_columns:
                logger.warning(f"No numeric columns available for aggregation, falling back to keep_first")
                result_df = result_df.drop_duplicates(subset=columns, keep='first')
                self.removed_indices = df.index[df.duplicated(subset=columns, keep='first')].tolist()
            else:
                # Create aggregation dictionary
                agg_dict = {}
                for col in aggregation_columns:
                    if strategy == "aggregate_mean":
                        agg_dict[col] = 'mean'
                    else:  # aggregate_sum
                        agg_dict[col] = 'sum'
                        
                # For non-numeric columns, keep first value
                for col in result_df.columns:
                    if col not in aggregation_columns and col not in columns:
                        agg_dict[col] = 'first'
                        
                # Aggregate
                result_df = result_df.groupby(columns, as_index=False).agg(agg_dict)
                
                # Track removed indices (all duplicates except the kept ones)
                duplicated_mask = df.duplicated(subset=columns, keep='first')
                self.removed_indices = df.index[duplicated_mask].tolist()
                
        elif strategy == "mark" and mark_column:
            # Mark duplicates instead of removing them
            if mark_column not in result_df.columns:
                result_df[mark_column] = False
                
            # Set mark for duplicates
            duplicated_mask = result_df.duplicated(subset=columns, keep='first')
            result_df.loc[duplicated_mask, mark_column] = True
            
            # No rows removed in this case
            self.removed_indices = []
            
        else:
            logger.warning(f"Unknown strategy '{strategy}', falling back to keep_first")
            result_df = result_df.drop_duplicates(subset=columns, keep='first')
            self.removed_indices = df.index[df.duplicated(subset=columns, keep='first')].tolist()
            
        # Log statistics
        removed_count = len(self.removed_indices)
        logger.info(f"Removed {removed_count} duplicate rows using strategy '{strategy}'")
        
        return result_df
        
    def _apply_fuzzy_hashing(
        self,
        df: pd.DataFrame,
        string_columns: List[str],
        threshold: float = 0.9
    ) -> pd.DataFrame:
        """
        Apply fuzzy hashing to string columns for fuzzy duplicate detection.
        
        Args:
            df: DataFrame to process
            string_columns: String columns to process
            threshold: Similarity threshold
            
        Returns:
            Processed DataFrame
        """
        try:
            from rapidfuzz import fuzz
            
            # Create a copy of the DataFrame
            result = df.copy()
            
            # Create a new column for fuzzy hashing
            fuzzy_col = "_fuzzy_hash"
            
            # Generate simhash for string columns
            result[fuzzy_col] = result.apply(
                lambda row: self._generate_simhash(row, string_columns),
                axis=1
            )
            
            # Group by simhash
            groups = []
            processed = set()
            
            for idx, row in result.iterrows():
                if idx in processed:
                    continue
                    
                # Find similar rows
                current_hash = row[fuzzy_col]
                group = [idx]
                
                for other_idx, other_row in result.iterrows():
                    if other_idx != idx and other_idx not in processed:
                        other_hash = other_row[fuzzy_col]
                        
                        # Calculate similarity
                        similarity = self._calculate_hash_similarity(current_hash, other_hash)
                        
                        if similarity >= threshold:
                            group.append(other_idx)
                            processed.add(other_idx)
                            
                if len(group) > 1:
                    groups.append(group)
                    
                processed.add(idx)
                
            # Drop fuzzy hash column
            result = result.drop(columns=[fuzzy_col])
            
            return result
            
        except ImportError:
            logger.warning("rapidfuzz not available, falling back to exact matching")
            return df
            
    def _generate_simhash(
        self,
        row: pd.Series,
        string_columns: List[str]
    ) -> str:
        """
        Generate a simhash for a row based on string columns.
        
        Args:
            row: DataFrame row
            string_columns: String columns to include
            
        Returns:
            Simhash as a string
        """
        # Concatenate string values
        text = " ".join(str(row[col]) for col in string_columns if not pd.isna(row[col]))
        
        # Generate hash
        return hashlib.md5(text.encode()).hexdigest()
        
    def _calculate_hash_similarity(
        self,
        hash1: str,
        hash2: str
    ) -> float:
        """
        Calculate similarity between two hashes.
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Similarity score (0-1)
        """
        # Convert to binary
        bin1 = bin(int(hash1, 16))[2:].zfill(128)
        bin2 = bin(int(hash2, 16))[2:].zfill(128)
        
        # Calculate Hamming distance
        distance = sum(b1 != b2 for b1, b2 in zip(bin1, bin2))
        
        # Convert to similarity
        return 1 - (distance / len(bin1))
        
    def _apply_time_window(
        self,
        df: pd.DataFrame,
        time_column: str,
        time_window: str,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply time window for time-based deduplication.
        
        Args:
            df: DataFrame to process
            time_column: Column containing timestamps
            time_window: Time window specification
            columns: Columns to consider for duplicate detection
            
        Returns:
            Processed DataFrame
        """
        try:
            # Ensure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
                
            # Sort by time column
            sorted_df = df.sort_values(by=time_column)
            
            # Parse time window
            window_size = pd.Timedelta(time_window)
            
            # Group records within time window
            groups = []
            current_group = []
            
            for i in range(len(sorted_df)):
                row = sorted_df.iloc[i]
                
                if not current_group:
                    current_group.append(row)
                    continue
                    
                prev_row = current_group[-1]
                time_diff = row[time_column] - prev_row[time_column]
                
                # Check if row is within time window and has matching columns
                if time_diff <= window_size:
                    # Check if columns match
                    match = all(row[col] == prev_row[col] for col in columns if col != time_column)
                    if match:
                        current_group.append(row)
                    else:
                        # Start new group if values don't match
                        if len(current_group) > 1:
                            groups.append(current_group)
                        current_group = [row]
                else:
                    # Start new group if outside time window
                    if len(current_group) > 1:
                        groups.append(current_group)
                    current_group = [row]
                    
            # Add last group if it contains duplicates
            if len(current_group) > 1:
                groups.append(current_group)
                
            # Create result DataFrame with group information
            return sorted_df
            
        except Exception as e:
            logger.warning(f"Error applying time window: {str(e)}")
            return df
            
    def get_removed_indices(self) -> List[Any]:
        """
        Get indices of removed duplicate rows.
        
        Returns:
            List of removed indices
        """
        return self.removed_indices