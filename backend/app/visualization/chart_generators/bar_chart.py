"""
Bar chart generator module for the supply chain LLM platform.

This module provides functionality to generate bar charts from
supply chain data for various analytics use cases.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class BarChartGenerator:
    """
    Generates bar charts for supply chain analytics.
    
    This class provides methods to create different types of bar charts,
    including simple bars, grouped bars, and stacked bars from
    supply chain data.
    """
    
    def __init__(self):
        """Initialize the bar chart generator."""
        # Set default style
        sns.set_style("whitegrid")
        self.default_colors = plt.cm.tab10.colors
        self.default_figsize = (10, 6)
        
    def generate_bar_chart(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        x_column: str,
        y_column: str,
        title: str = "Bar Chart",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        orientation: str = "vertical",
        color: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        sort_values: bool = False,
        limit: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a simple bar chart.
        
        Args:
            data: DataFrame or list of dicts containing the data
            x_column: Column name for x-axis categories
            y_column: Column name for y-axis values
            title: Chart title
            x_label: X-axis label (defaults to x_column)
            y_label: Y-axis label (defaults to y_column)
            orientation: "vertical" or "horizontal"
            color: Bar color
            figsize: Figure size as (width, height) tuple
            sort_values: Whether to sort by values
            limit: Limit the number of bars shown
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
                
            # Apply limit if specified
            if limit and len(df) > limit:
                if sort_values:
                    df = df.sort_values(by=y_column, ascending=False).head(limit)
                else:
                    df = df.head(limit)
            elif sort_values:
                df = df.sort_values(by=y_column, ascending=False)
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Generate bar chart based on orientation
            if orientation == "horizontal":
                bars = ax.barh(
                    df[x_column], 
                    df[y_column],
                    color=color or self.default_colors[0],
                    **kwargs
                )
                # Set labels
                ax.set_xlabel(y_label or y_column)
                ax.set_ylabel(x_label or x_column)
            else:  # vertical
                bars = ax.bar(
                    df[x_column], 
                    df[y_column],
                    color=color or self.default_colors[0],
                    **kwargs
                )
                # Set labels
                ax.set_xlabel(x_label or x_column)
                ax.set_ylabel(y_label or y_column)
                # Rotate x labels if there are many categories
                if len(df) > 5:
                    plt.xticks(rotation=45, ha="right")
                    
            # Add data labels on bars
            for bar in bars:
                height = bar.get_height() if orientation == "vertical" else bar.get_width()
                text_x = bar.get_x() + bar.get_width()/2 if orientation == "vertical" else height + 0.1
                text_y = height + 0.1 if orientation == "vertical" else bar.get_y() + bar.get_height()/2
                ha = "center" if orientation == "vertical" else "left"
                va = "bottom" if orientation == "vertical" else "center"
                ax.text(text_x, text_y, f"{height:.1f}", ha=ha, va=va, fontsize=9)
                
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
                "type": "bar",
                "orientation": orientation,
                "title": title,
                "x_column": x_column,
                "y_column": y_column,
                "data": df.to_dict(orient="records"),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating bar chart: {str(e)}")
            raise
    
    def generate_grouped_bar_chart(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        x_column: str,
        y_columns: List[str],
        title: str = "Grouped Bar Chart",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        orientation: str = "vertical",
        colors: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        sort_values: bool = False,
        limit: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a grouped bar chart with multiple series.
        
        Args:
            data: DataFrame or list of dicts containing the data
            x_column: Column name for x-axis categories
            y_columns: List of column names for different bar groups
            title: Chart title
            x_label: X-axis label (defaults to x_column)
            y_label: Y-axis label
            orientation: "vertical" or "horizontal"
            colors: List of colors for different groups
            figsize: Figure size as (width, height) tuple
            sort_values: Whether to sort by sum of values
            limit: Limit the number of groups shown
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
                
            # Apply sorting and limit if specified
            if sort_values:
                df['_sort_value'] = df[y_columns].sum(axis=1)
                df = df.sort_values(by='_sort_value', ascending=False)
                if '_sort_value' in df.columns:
                    df = df.drop('_sort_value', axis=1)
                    
            if limit and len(df) > limit:
                df = df.head(limit)
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Set up colors
            if not colors:
                colors = self.default_colors[:len(y_columns)]
                
            # Bar width and positions
            width = 0.8 / len(y_columns)
            x = np.arange(len(df))
            
            # Generate grouped bars
            bars = []
            if orientation == "horizontal":
                for i, y_col in enumerate(y_columns):
                    y_pos = x - width * (len(y_columns) - 1) / 2 + i * width
                    bar = ax.barh(y_pos, df[y_col], width, 
                                  label=y_col, color=colors[i % len(colors)], **kwargs)
                    bars.append(bar)
                ax.set_yticks(x)
                ax.set_yticklabels(df[x_column])
                ax.set_xlabel(y_label or "Value")
                ax.set_ylabel(x_label or x_column)
            else:  # vertical
                for i, y_col in enumerate(y_columns):
                    x_pos = x - width * (len(y_columns) - 1) / 2 + i * width
                    bar = ax.bar(x_pos, df[y_col], width, 
                                 label=y_col, color=colors[i % len(colors)], **kwargs)
                    bars.append(bar)
                ax.set_xticks(x)
                ax.set_xticklabels(df[x_column], rotation=45, ha="right")
                ax.set_xlabel(x_label or x_column)
                ax.set_ylabel(y_label or "Value")
                
            # Add legend
            ax.legend()
            
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
                "type": "grouped_bar",
                "orientation": orientation,
                "title": title,
                "x_column": x_column,
                "y_columns": y_columns,
                "data": df.to_dict(orient="records"),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating grouped bar chart: {str(e)}")
            raise
    
    def generate_stacked_bar_chart(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        x_column: str,
        y_columns: List[str],
        title: str = "Stacked Bar Chart",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        orientation: str = "vertical",
        colors: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        sort_values: bool = False,
        limit: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a stacked bar chart.
        
        Args:
            data: DataFrame or list of dicts containing the data
            x_column: Column name for x-axis categories
            y_columns: List of column names for different stacked segments
            title: Chart title
            x_label: X-axis label (defaults to x_column)
            y_label: Y-axis label
            orientation: "vertical" or "horizontal"
            colors: List of colors for different segments
            figsize: Figure size as (width, height) tuple
            sort_values: Whether to sort by total values
            limit: Limit the number of bars shown
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
                
            # Apply sorting and limit if specified
            if sort_values:
                df['_sort_value'] = df[y_columns].sum(axis=1)
                df = df.sort_values(by='_sort_value', ascending=False)
                if '_sort_value' in df.columns:
                    df = df.drop('_sort_value', axis=1)
                    
            if limit and len(df) > limit:
                df = df.head(limit)
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Set up colors
            if not colors:
                colors = self.default_colors[:len(y_columns)]
                
            # Generate stacked bars
            if orientation == "horizontal":
                df_plot = df.set_index(x_column)
                df_plot[y_columns].plot.barh(stacked=True, ax=ax, color=colors, **kwargs)
                ax.set_xlabel(y_label or "Value")
                ax.set_ylabel(x_label or x_column)
            else:  # vertical
                df_plot = df.set_index(x_column)
                df_plot[y_columns].plot.bar(stacked=True, ax=ax, color=colors, **kwargs)
                ax.set_xlabel(x_label or x_column)
                ax.set_ylabel(y_label or "Value")
                plt.xticks(rotation=45, ha="right")
                
            # Add legend
            ax.legend(title="Categories")
            
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
                "type": "stacked_bar",
                "orientation": orientation,
                "title": title,
                "x_column": x_column,
                "y_columns": y_columns,
                "data": df.to_dict(orient="records"),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating stacked bar chart: {str(e)}")
            raise
    
    def generate_histogram(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        value_column: str,
        bins: int = 10,
        title: str = "Histogram",
        x_label: Optional[str] = None,
        y_label: str = "Frequency",
        color: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a histogram.
        
        Args:
            data: DataFrame or list of dicts containing the data
            value_column: Column name containing values to distribute
            bins: Number of bins for histogram
            title: Chart title
            x_label: X-axis label (defaults to value_column)
            y_label: Y-axis label
            color: Bar color
            figsize: Figure size as (width, height) tuple
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
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Generate histogram
            n, bins, patches = ax.hist(
                df[value_column], 
                bins=bins,
                color=color or self.default_colors[0],
                **kwargs
            )
            
            # Set labels
            ax.set_xlabel(x_label or value_column)
            ax.set_ylabel(y_label)
            
            # Set title and layout
            ax.set_title(title)
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare histogram data
            hist_data = []
            for i in range(len(n)):
                hist_data.append({
                    "bin_start": bins[i],
                    "bin_end": bins[i+1],
                    "frequency": n[i]
                })
            
            # Prepare result
            result = {
                "type": "histogram",
                "title": title,
                "value_column": value_column,
                "bins": bins,
                "histogram_data": hist_data,
                "data_summary": {
                    "mean": float(df[value_column].mean()),
                    "median": float(df[value_column].median()),
                    "std_dev": float(df[value_column].std()),
                    "min": float(df[value_column].min()),
                    "max": float(df[value_column].max())
                },
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating histogram: {str(e)}")
            raise
    
    def generate_pareto_chart(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        category_column: str,
        value_column: str,
        title: str = "Pareto Chart",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        bar_color: str = "#1f77b4",
        line_color: str = "#ff7f0e",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a Pareto chart (sorted bar chart with cumulative line).
        
        Args:
            data: DataFrame or list of dicts containing the data
            category_column: Column name for categories
            value_column: Column name for values
            title: Chart title
            x_label: X-axis label (defaults to category_column)
            y_label: Y-axis label (defaults to value_column)
            figsize: Figure size as (width, height) tuple
            bar_color: Color for bars
            line_color: Color for cumulative line
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
                
            # Sort values in descending order
            df = df.sort_values(by=value_column, ascending=False)
            
            # Calculate cumulative percentage
            total = df[value_column].sum()
            df['cumulative'] = df[value_column].cumsum() / total * 100
            
            # Create figure with two y-axes
            fig, ax1 = plt.subplots(figsize=figsize or self.default_figsize)
            ax2 = ax1.twinx()
            
            # Plot bars on first y-axis
            bars = ax1.bar(
                df[category_column], 
                df[value_column],
                color=bar_color,
                **kwargs
            )
            
            # Plot cumulative line on second y-axis
            ax2.plot(
                df[category_column], 
                df['cumulative'], 
                color=line_color, 
                marker='o',
                linewidth=2
            )
            
            # Add horizontal line at 80% (Pareto principle)
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7)
            
            # Set axes ranges and labels
            ax1.set_ylim(0, df[value_column].max() * 1.1)
            ax2.set_ylim(0, 105)
            
            ax1.set_xlabel(x_label or category_column)
            ax1.set_ylabel(y_label or value_column)
            ax2.set_ylabel('Cumulative Percentage (%)')
            
            # Rotate x labels if there are many categories
            if len(df) > 5:
                plt.xticks(rotation=45, ha="right")
                
            # Set title and layout
            ax1.set_title(title)
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare result
            result = {
                "type": "pareto",
                "title": title,
                "category_column": category_column,
                "value_column": value_column,
                "data": df.to_dict(orient="records"),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating Pareto chart: {str(e)}")
            raise

# Make sure to import numpy
import numpy as np