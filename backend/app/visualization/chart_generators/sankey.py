"""
Heatmap generator module for the supply chain LLM platform.

This module provides functionality to generate heatmaps from
supply chain data for various analytics use cases.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class HeatmapGenerator:
    """
    Generates heatmaps for supply chain analytics.
    
    This class provides methods to create different types of heatmaps,
    including standard heatmaps, correlation matrices, and calendar heatmaps
    from supply chain data.
    """
    
    def __init__(self):
        """Initialize the heatmap generator."""
        # Set default style
        sns.set_style("white")
        self.default_cmap = "YlOrRd"  # Yellow-Orange-Red
        self.default_figsize = (12, 8)
        
    def generate_heatmap(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        x_column: str,
        y_column: str,
        value_column: str,
        title: str = "Heatmap",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        cmap: str = None,
        figsize: Optional[Tuple[int, int]] = None,
        annotate: bool = True,
        normalize: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a standard heatmap.
        
        Args:
            data: DataFrame or list of dicts containing the data
            x_column: Column name for x-axis categories
            y_column: Column name for y-axis categories
            value_column: Column name for cell values
            title: Chart title
            x_label: X-axis label (defaults to x_column)
            y_label: Y-axis label (defaults to y_column)
            cmap: Colormap name
            figsize: Figure size as (width, height) tuple
            annotate: Whether to show values in cells
            normalize: Whether to normalize values (0-1 scale)
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing chart data and metadata
        """
        try:
            # Convert to DataFrame if list
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            # Pivot data into heatmap format
            pivot_data = df.pivot_table(
                index=y_column, 
                columns=x_column, 
                values=value_column, 
                aggfunc='sum'
            )
            
            # Normalize if requested
            if normalize:
                max_value = pivot_data.max().max()
                min_value = pivot_data.min().min()
                if max_value > min_value:  # Avoid division by zero
                    pivot_data = (pivot_data - min_value) / (max_value - min_value)
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Generate heatmap
            heatmap = sns.heatmap(
                pivot_data,
                annot=annotate,
                fmt=".2f" if normalize else ".1f",
                cmap=cmap or self.default_cmap,
                linewidths=0.5,
                ax=ax,
                **kwargs
            )
            
            # Set labels
            ax.set_xlabel(x_label or x_column)
            ax.set_ylabel(y_label or y_column)
            
            # Rotate x labels if there are many categories
            plt.xticks(rotation=45, ha="right")
            
            # Set title
            ax.set_title(title)
            
            # Tight layout
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Convert pivot back to regular dataframe for result
            result_data = pivot_data.reset_index().melt(
                id_vars=y_column,
                var_name=x_column,
                value_name=value_column
            )
            
            # Prepare result
            result = {
                "type": "heatmap",
                "title": title,
                "x_column": x_column,
                "y_column": y_column,
                "value_column": value_column,
                "normalized": normalize,
                "data": result_data.to_dict(orient="records"),
                "x_categories": pivot_data.columns.tolist(),
                "y_categories": pivot_data.index.tolist(),
                "min_value": float(pivot_data.min().min()),
                "max_value": float(pivot_data.max().max()),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating heatmap: {str(e)}")
            raise
    
    def generate_correlation_heatmap(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        columns: Optional[List[str]] = None,
        title: str = "Correlation Matrix",
        cmap: str = "coolwarm",
        figsize: Optional[Tuple[int, int]] = None,
        annotate: bool = True,
        mask_upper: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a correlation matrix heatmap.
        
        Args:
            data: DataFrame or list of dicts containing the data
            columns: List of columns to include in correlation
            title: Chart title
            cmap: Colormap name (default is coolwarm for correlation)
            figsize: Figure size as (width, height) tuple
            annotate: Whether to show values in cells
            mask_upper: Whether to mask the upper triangle
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing chart data and metadata
        """
        try:
            # Convert to DataFrame if list
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            # Filter columns if specified
            if columns:
                df = df[columns]
            else:
                # Only include numeric columns
                df = df.select_dtypes(include=['number'])
                
            # Compute correlation matrix
            corr_matrix = df.corr()
            
            # Create mask for upper triangle if requested
            mask = None
            if mask_upper:
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Generate heatmap
            heatmap = sns.heatmap(
                corr_matrix,
                annot=annotate,
                fmt=".2f",
                cmap=cmap,
                linewidths=0.5,
                mask=mask,
                vmin=-1,
                vmax=1,
                center=0,
                ax=ax,
                **kwargs
            )
            
            # Set title
            ax.set_title(title)
            
            # Make y labels more readable
            plt.yticks(rotation=0)
            
            # Tight layout
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare result data
            result_data = corr_matrix.reset_index()
            result_data.columns.name = None
            result_data = result_data.rename(columns={'index': 'variable_1'})
            result_data = result_data.melt(
                id_vars='variable_1',
                var_name='variable_2',
                value_name='correlation'
            )
            
            # Prepare result
            result = {
                "type": "correlation_heatmap",
                "title": title,
                "columns": df.columns.tolist(),
                "data": result_data.to_dict(orient="records"),
                "correlation_matrix": corr_matrix.to_dict(),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating correlation heatmap: {str(e)}")
            raise
    
    def generate_calendar_heatmap(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        date_column: str,
        value_column: str,
        title: str = "Calendar Heatmap",
        cmap: str = None,
        figsize: Optional[Tuple[int, int]] = None,
        year: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a calendar heatmap (values by day).
        
        Args:
            data: DataFrame or list of dicts containing the data
            date_column: Column name containing dates
            value_column: Column name containing values
            title: Chart title
            cmap: Colormap name
            figsize: Figure size as (width, height) tuple
            year: Specific year to visualize (defaults to most recent)
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing chart data and metadata
        """
        try:
            # Convert to DataFrame if list
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            # Convert date column to datetime
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Filter by year if specified
            if year:
                df = df[df[date_column].dt.year == year]
            else:
                # Default to most recent year
                year = df[date_column].dt.year.max()
                df = df[df[date_column].dt.year == year]
                
            if df.empty:
                raise ValueError(f"No data available for year {year}")
                
            # Create date index with all days of the year
            start_date = pd.Timestamp(f"{year}-01-01")
            end_date = pd.Timestamp(f"{year}-12-31")
            date_index = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Create a DataFrame with all days
            calendar_df = pd.DataFrame(index=date_index)
            
            # Aggregate data by day and merge with calendar
            daily_data = df.groupby(df[date_column].dt.date)[value_column].sum()
            calendar_df['value'] = daily_data
            
            # Fill missing values with 0
            calendar_df['value'] = calendar_df['value'].fillna(0)
            
            # Extract month and day of week
            calendar_df['month'] = calendar_df.index.month
            calendar_df['day_of_week'] = calendar_df.index.dayofweek
            calendar_df['day'] = calendar_df.index.day
            
            # Create month labels
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_days = [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30, 
                        31, 31, 30, 31, 30, 31]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or (16, 8))
            
            # Create pivot data for heatmap
            pivot_data = calendar_df.pivot_table(
                index='day_of_week', 
                columns='month', 
                values='value', 
                aggfunc='sum'
            )
            
            # Generate heatmap
            heatmap = sns.heatmap(
                pivot_data,
                cmap=cmap or self.default_cmap,
                linewidths=1,
                linecolor='white',
                cbar_kws={'label': value_column},
                **kwargs
            )
            
            # Set y labels (days of week)
            ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            
            # Set x labels (months)
            ax.set_xticks(np.arange(len(month_labels)) + 0.5)
            ax.set_xticklabels(month_labels)
            
            # Set title
            ax.set_title(f"{title} ({year})")
            
            # Tight layout
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Create a daily result dataset
            result_data = calendar_df.reset_index()
            result_data = result_data.rename(columns={'index': 'date'})
            
            # Prepare result
            result = {
                "type": "calendar_heatmap",
                "title": title,
                "date_column": date_column,
                "value_column": value_column,
                "year": year,
                "data": result_data.to_dict(orient="records"),
                "monthly_totals": calendar_df.groupby('month')['value'].sum().to_dict(),
                "day_of_week_totals": calendar_df.groupby('day_of_week')['value'].sum().to_dict(),
                "min_value": float(calendar_df['value'].min()),
                "max_value": float(calendar_df['value'].max()),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating calendar heatmap: {str(e)}")
            raise
    
    def generate_matrix_heatmap(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        row_column: str,
        column_column: str,
        value_column: str,
        title: str = "Matrix Heatmap",
        cmap: str = None,
        figsize: Optional[Tuple[int, int]] = None,
        annotate: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a matrix heatmap for comparing categories.
        
        Args:
            data: DataFrame or list of dicts containing the data
            row_column: Column name for matrix rows
            column_column: Column name for matrix columns
            value_column: Column name for cell values
            title: Chart title
            cmap: Colormap name
            figsize: Figure size as (width, height) tuple
            annotate: Whether to show values in cells
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing chart data and metadata
        """
        try:
            # Convert to DataFrame if list
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            # Create pivot table
            pivot_data = df.pivot_table(
                index=row_column,
                columns=column_column,
                values=value_column,
                aggfunc='mean',
                fill_value=0
            )
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Generate heatmap
            heatmap = sns.heatmap(
                pivot_data,
                annot=annotate,
                fmt=".2f",
                cmap=cmap or self.default_cmap,
                linewidths=0.5,
                ax=ax,
                **kwargs
            )
            
            # Set labels
            ax.set_xlabel(column_column)
            ax.set_ylabel(row_column)
            
            # Set title
            ax.set_title(title)
            
            # Rotate axis labels if needed
            if len(pivot_data.columns) > 5:
                plt.xticks(rotation=45, ha="right")
                
            if len(pivot_data.index) > 5:
                plt.yticks(rotation=0)
            
            # Tight layout
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Convert pivot back to regular dataframe for result
            result_data = pivot_data.reset_index().melt(
                id_vars=row_column,
                var_name=column_column,
                value_name=value_column
            )
            
            # Prepare result
            result = {
                "type": "matrix_heatmap",
                "title": title,
                "row_column": row_column,
                "column_column": column_column,
                "value_column": value_column,
                "data": result_data.to_dict(orient="records"),
                "row_categories": pivot_data.index.tolist(),
                "column_categories": pivot_data.columns.tolist(),
                "min_value": float(pivot_data.min().min()),
                "max_value": float(pivot_data.max().max()),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating matrix heatmap: {str(e)}")
            raise