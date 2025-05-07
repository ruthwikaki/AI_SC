"""
Line chart generator module for the supply chain LLM platform.

This module provides functionality to generate line charts from
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

class LineChartGenerator:
    """
    Generates line charts for supply chain analytics.
    
    This class provides methods to create different types of line charts,
    including simple lines, multi-line charts, and area charts from
    supply chain data.
    """
    
    def __init__(self):
        """Initialize the line chart generator."""
        # Set default style
        sns.set_style("whitegrid")
        self.default_colors = plt.cm.tab10.colors
        self.default_figsize = (10, 6)
        
    def generate_line_chart(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        x_column: str,
        y_column: str,
        title: str = "Line Chart",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        color: Optional[str] = None,
        marker: Optional[str] = 'o',
        figsize: Optional[Tuple[int, int]] = None,
        time_series: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a simple line chart.
        
        Args:
            data: DataFrame or list of dicts containing the data
            x_column: Column name for x-axis (typically time periods)
            y_column: Column name for y-axis values
            title: Chart title
            x_label: X-axis label (defaults to x_column)
            y_label: Y-axis label (defaults to y_column)
            color: Line color
            marker: Marker style (None for no markers)
            figsize: Figure size as (width, height) tuple
            time_series: Whether to treat x as datetime
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
                
            # Sort data by x values to ensure proper line
            df = df.sort_values(by=x_column)
                
            # Convert to datetime if time_series
            if time_series and not pd.api.types.is_datetime64_any_dtype(df[x_column]):
                try:
                    df[x_column] = pd.to_datetime(df[x_column])
                except Exception as e:
                    logger.warning(f"Could not convert {x_column} to datetime: {str(e)}")
                    time_series = False
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Plot line
            line = ax.plot(
                df[x_column], 
                df[y_column],
                marker=marker,
                color=color or self.default_colors[0],
                **kwargs
            )
            
            # Set labels
            ax.set_xlabel(x_label or x_column)
            ax.set_ylabel(y_label or y_column)
            
            # Format x-axis if time series
            if time_series:
                fig.autofmt_xdate()
                
            # Rotate x labels if there are many points
            if len(df) > 10 and not time_series:
                plt.xticks(rotation=45, ha="right")
                
            # Set title and layout
            ax.set_title(title)
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare result
            result = {
                "type": "line",
                "title": title,
                "x_column": x_column,
                "y_column": y_column,
                "time_series": time_series,
                "data": df.to_dict(orient="records"),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating line chart: {str(e)}")
            raise
            
    def generate_multi_line_chart(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        x_column: str,
        y_columns: List[str],
        title: str = "Multi-Line Chart",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        colors: Optional[List[str]] = None,
        markers: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        time_series: bool = False,
        include_legend: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a multi-line chart with multiple series.
        
        Args:
            data: DataFrame or list of dicts containing the data
            x_column: Column name for x-axis (typically time periods)
            y_columns: List of column names for different lines
            title: Chart title
            x_label: X-axis label (defaults to x_column)
            y_label: Y-axis label
            colors: List of line colors
            markers: List of marker styles
            figsize: Figure size as (width, height) tuple
            time_series: Whether to treat x as datetime
            include_legend: Whether to include legend
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
                
            # Sort data by x values to ensure proper lines
            df = df.sort_values(by=x_column)
                
            # Convert to datetime if time_series
            if time_series and not pd.api.types.is_datetime64_any_dtype(df[x_column]):
                try:
                    df[x_column] = pd.to_datetime(df[x_column])
                except Exception as e:
                    logger.warning(f"Could not convert {x_column} to datetime: {str(e)}")
                    time_series = False
                    
            # Set up colors and markers
            if not colors:
                colors = self.default_colors[:len(y_columns)]
                
            if not markers:
                markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+']
                markers = markers[:len(y_columns)]
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Plot lines for each y column
            for i, y_col in enumerate(y_columns):
                marker = markers[i % len(markers)] if markers else None
                color = colors[i % len(colors)]
                
                ax.plot(
                    df[x_column], 
                    df[y_col],
                    marker=marker,
                    color=color,
                    label=y_col,
                    **kwargs
                )
            
            # Set labels
            ax.set_xlabel(x_label or x_column)
            ax.set_ylabel(y_label or "Value")
            
            # Add legend if requested
            if include_legend:
                ax.legend()
                
            # Format x-axis if time series
            if time_series:
                fig.autofmt_xdate()
                
            # Rotate x labels if there are many points
            if len(df) > 10 and not time_series:
                plt.xticks(rotation=45, ha="right")
                
            # Set title and layout
            ax.set_title(title)
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare result
            result = {
                "type": "multi_line",
                "title": title,
                "x_column": x_column,
                "y_columns": y_columns,
                "time_series": time_series,
                "data": df.to_dict(orient="records"),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating multi-line chart: {str(e)}")
            raise
            
    def generate_area_chart(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        x_column: str,
        y_column: str,
        title: str = "Area Chart",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 0.5,
        figsize: Optional[Tuple[int, int]] = None,
        time_series: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a simple area chart.
        
        Args:
            data: DataFrame or list of dicts containing the data
            x_column: Column name for x-axis (typically time periods)
            y_column: Column name for y-axis values
            title: Chart title
            x_label: X-axis label (defaults to x_column)
            y_label: Y-axis label (defaults to y_column)
            color: Fill color
            alpha: Transparency of fill
            figsize: Figure size as (width, height) tuple
            time_series: Whether to treat x as datetime
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
                
            # Sort data by x values to ensure proper area
            df = df.sort_values(by=x_column)
                
            # Convert to datetime if time_series
            if time_series and not pd.api.types.is_datetime64_any_dtype(df[x_column]):
                try:
                    df[x_column] = pd.to_datetime(df[x_column])
                except Exception as e:
                    logger.warning(f"Could not convert {x_column} to datetime: {str(e)}")
                    time_series = False
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Plot area
            ax.fill_between(
                df[x_column], 
                df[y_column],
                color=color or self.default_colors[0],
                alpha=alpha,
                **kwargs
            )
            
            # Add line on top of area
            ax.plot(
                df[x_column],
                df[y_column],
                color=color or self.default_colors[0],
                linewidth=2
            )
            
            # Set labels
            ax.set_xlabel(x_label or x_column)
            ax.set_ylabel(y_label or y_column)
            
            # Format x-axis if time series
            if time_series:
                fig.autofmt_xdate()
                
            # Rotate x labels if there are many points
            if len(df) > 10 and not time_series:
                plt.xticks(rotation=45, ha="right")
                
            # Set title and layout
            ax.set_title(title)
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare result
            result = {
                "type": "area",
                "title": title,
                "x_column": x_column,
                "y_column": y_column,
                "time_series": time_series,
                "data": df.to_dict(orient="records"),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating area chart: {str(e)}")
            raise
            
    def generate_stacked_area_chart(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        x_column: str,
        y_columns: List[str],
        title: str = "Stacked Area Chart",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        colors: Optional[List[str]] = None,
        alpha: float = 0.7,
        figsize: Optional[Tuple[int, int]] = None,
        time_series: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a stacked area chart.
        
        Args:
            data: DataFrame or list of dicts containing the data
            x_column: Column name for x-axis (typically time periods)
            y_columns: List of column names for different areas
            title: Chart title
            x_label: X-axis label (defaults to x_column)
            y_label: Y-axis label
            colors: List of fill colors
            alpha: Transparency of fills
            figsize: Figure size as (width, height) tuple
            time_series: Whether to treat x as datetime
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
                
            # Sort data by x values to ensure proper areas
            df = df.sort_values(by=x_column)
                
            # Convert to datetime if time_series
            if time_series and not pd.api.types.is_datetime64_any_dtype(df[x_column]):
                try:
                    df[x_column] = pd.to_datetime(df[x_column])
                except Exception as e:
                    logger.warning(f"Could not convert {x_column} to datetime: {str(e)}")
                    time_series = False
                    
            # Set up colors
            if not colors:
                colors = self.default_colors[:len(y_columns)]
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Plot stacked area
            ax.stackplot(
                df[x_column],
                [df[col] for col in y_columns],
                labels=y_columns,
                colors=colors,
                alpha=alpha,
                **kwargs
            )
            
            # Set labels
            ax.set_xlabel(x_label or x_column)
            ax.set_ylabel(y_label or "Value")
            
            # Add legend
            ax.legend(loc="upper left")
            
            # Format x-axis if time series
            if time_series:
                fig.autofmt_xdate()
                
            # Rotate x labels if there are many points
            if len(df) > 10 and not time_series:
                plt.xticks(rotation=45, ha="right")
                
            # Set title and layout
            ax.set_title(title)
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Add total values to dataset
            df['total'] = df[y_columns].sum(axis=1)
            
            # Prepare result
            result = {
                "type": "stacked_area",
                "title": title,
                "x_column": x_column,
                "y_columns": y_columns,
                "time_series": time_series,
                "data": df.to_dict(orient="records"),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating stacked area chart: {str(e)}")
            raise
            
    def generate_trend_line_chart(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        x_column: str,
        y_column: str,
        title: str = "Trend Line Chart",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        color: Optional[str] = None,
        trend_color: Optional[str] = "red",
        marker: Optional[str] = 'o',
        figsize: Optional[Tuple[int, int]] = None,
        time_series: bool = False,
        trend_type: str = "linear",  # "linear", "polynomial", "moving_avg"
        polynomial_degree: int = 2,
        moving_avg_window: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a line chart with trend line.
        
        Args:
            data: DataFrame or list of dicts containing the data
            x_column: Column name for x-axis (typically time periods)
            y_column: Column name for y-axis values
            title: Chart title
            x_label: X-axis label (defaults to x_column)
            y_label: Y-axis label (defaults to y_column)
            color: Line color
            trend_color: Trend line color
            marker: Marker style
            figsize: Figure size as (width, height) tuple
            time_series: Whether to treat x as datetime
            trend_type: Type of trend ("linear", "polynomial", "moving_avg")
            polynomial_degree: Degree for polynomial trend
            moving_avg_window: Window size for moving average
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
                
            # Sort data by x values to ensure proper line
            df = df.sort_values(by=x_column)
                
            # Convert to datetime if time_series
            original_x = df[x_column].copy()
            if time_series and not pd.api.types.is_datetime64_any_dtype(df[x_column]):
                try:
                    df[x_column] = pd.to_datetime(df[x_column])
                    original_x = df[x_column].copy()
                    # For trend calculation, convert datetime to ordinal numbers
                    df['x_numeric'] = pd.to_datetime(df[x_column]).map(datetime.toordinal)
                    x_for_trend = df['x_numeric']
                except Exception as e:
                    logger.warning(f"Could not convert {x_column} to datetime: {str(e)}")
                    time_series = False
                    x_for_trend = df[x_column]
            else:
                # For non-datetime x values
                try:
                    # Try to convert to numeric for trend calculation
                    x_for_trend = pd.to_numeric(df[x_column])
                except:
                    # If conversion fails, use array indices
                    x_for_trend = np.arange(len(df))
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Plot actual data
            ax.plot(
                df[x_column], 
                df[y_column],
                marker=marker,
                color=color or self.default_colors[0],
                label="Actual",
                **kwargs
            )
            
            # Calculate and plot trend line
            y_values = df[y_column].values
            
            if trend_type == "linear":
                # Linear trend
                z = np.polyfit(range(len(x_for_trend)), y_values, 1)
                trend_fn = np.poly1d(z)
                trend_values = trend_fn(range(len(x_for_trend)))
                trend_label = f"Linear Trend (y = {z[0]:.2f}x + {z[1]:.2f})"
                
            elif trend_type == "polynomial":
                # Polynomial trend
                z = np.polyfit(range(len(x_for_trend)), y_values, polynomial_degree)
                trend_fn = np.poly1d(z)
                trend_values = trend_fn(range(len(x_for_trend)))
                trend_label = f"Polynomial Trend (degree {polynomial_degree})"
                
            elif trend_type == "moving_avg":
                # Moving average
                trend_values = df[y_column].rolling(window=moving_avg_window, min_periods=1).mean().values
                trend_label = f"Moving Average (window={moving_avg_window})"
                
            else:
                raise ValueError(f"Unsupported trend type: {trend_type}")
                
            # Plot trend line
            ax.plot(
                df[x_column], 
                trend_values,
                color=trend_color,
                linestyle='--',
                linewidth=2,
                label=trend_label
            )
            
            # Set labels
            ax.set_xlabel(x_label or x_column)
            ax.set_ylabel(y_label or y_column)
            
            # Add legend
            ax.legend()
            
            # Format x-axis if time series
            if time_series:
                fig.autofmt_xdate()
                
            # Rotate x labels if there are many points
            if len(df) > 10 and not time_series:
                plt.xticks(rotation=45, ha="right")
                
            # Set title and layout
            ax.set_title(title)
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Add trend values to dataset
            df['trend'] = trend_values
            
            # If we converted datetime to numeric, restore original values
            if time_series and 'x_numeric' in df.columns:
                df = df.drop('x_numeric', axis=1)
                
            # Prepare result
            result = {
                "type": "trend_line",
                "title": title,
                "x_column": x_column,
                "y_column": y_column,
                "trend_type": trend_type,
                "time_series": time_series,
                "data": df.to_dict(orient="records"),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating trend line chart: {str(e)}")
            raise