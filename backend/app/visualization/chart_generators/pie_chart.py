"""
Pie chart generator module for the supply chain LLM platform.

This module provides functionality to generate pie and donut charts from
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

class PieChartGenerator:
    """
    Generates pie and donut charts for supply chain analytics.
    
    This class provides methods to create different types of pie charts,
    including standard pies, donut charts, and exploded pie charts from
    supply chain data.
    """
    
    def __init__(self):
        """Initialize the pie chart generator."""
        # Set default style
        sns.set_style("whitegrid")
        self.default_colors = plt.cm.tab10.colors
        self.default_figsize = (10, 6)
        
    def generate_pie_chart(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        category_column: str,
        value_column: str,
        title: str = "Pie Chart",
        colors: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        sort_values: bool = True,
        limit: Optional[int] = None,
        explode_largest: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a pie chart.
        
        Args:
            data: DataFrame or list of dicts containing the data
            category_column: Column name for pie slice categories
            value_column: Column name for pie slice values
            title: Chart title
            colors: List of colors for pie slices
            figsize: Figure size as (width, height) tuple
            sort_values: Whether to sort by values (descending)
            limit: Limit the number of categories shown (others will be grouped)
            explode_largest: Whether to explode (offset) the largest slice
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
                
            # Sort values if requested
            if sort_values:
                df = df.sort_values(by=value_column, ascending=False)
                
            # Apply limit if specified
            original_categories = len(df)
            if limit and len(df) > limit:
                # Group small categories into "Other"
                top_categories = df.head(limit - 1)
                other_sum = df.iloc[limit - 1:][value_column].sum()
                
                # Only include "Other" if it has a value
                if other_sum > 0:
                    other_row = pd.DataFrame({
                        category_column: ["Other"],
                        value_column: [other_sum]
                    })
                    df = pd.concat([top_categories, other_row], ignore_index=True)
                else:
                    df = top_categories
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Set up colors
            if not colors:
                colors = self.default_colors[:len(df)]
                
            # Set up explode array if requested
            explode = None
            if explode_largest:
                explode = [0] * len(df)
                largest_idx = df[value_column].argmax()
                explode[largest_idx] = 0.1
                
            # Generate pie chart
            wedges, texts, autotexts = ax.pie(
                df[value_column],
                labels=df[category_column],
                autopct='%1.1f%%',
                explode=explode,
                colors=colors,
                shadow=False,
                startangle=90,
                **kwargs
            )
            
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
                
            # Set title
            ax.set_title(title)
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Calculate percentages for result data
            total = df[value_column].sum()
            df['percentage'] = df[value_column] / total * 100
            
            # Prepare result
            result = {
                "type": "pie",
                "title": title,
                "category_column": category_column,
                "value_column": value_column,
                "data": df.to_dict(orient="records"),
                "total": float(total),
                "original_categories": original_categories,
                "displayed_categories": len(df),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating pie chart: {str(e)}")
            raise
    
    def generate_donut_chart(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        category_column: str,
        value_column: str,
        title: str = "Donut Chart",
        colors: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        sort_values: bool = True,
        limit: Optional[int] = None,
        center_text: Optional[str] = None,
        center_value: Optional[Union[float, int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a donut chart.
        
        Args:
            data: DataFrame or list of dicts containing the data
            category_column: Column name for donut slice categories
            value_column: Column name for donut slice values
            title: Chart title
            colors: List of colors for donut slices
            figsize: Figure size as (width, height) tuple
            sort_values: Whether to sort by values (descending)
            limit: Limit the number of categories shown (others will be grouped)
            center_text: Optional text to display in center of donut
            center_value: Optional value to display in center of donut (below text)
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
                
            # Sort values if requested
            if sort_values:
                df = df.sort_values(by=value_column, ascending=False)
                
            # Apply limit if specified
            original_categories = len(df)
            if limit and len(df) > limit:
                # Group small categories into "Other"
                top_categories = df.head(limit - 1)
                other_sum = df.iloc[limit - 1:][value_column].sum()
                
                # Only include "Other" if it has a value
                if other_sum > 0:
                    other_row = pd.DataFrame({
                        category_column: ["Other"],
                        value_column: [other_sum]
                    })
                    df = pd.concat([top_categories, other_row], ignore_index=True)
                else:
                    df = top_categories
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Set up colors
            if not colors:
                colors = self.default_colors[:len(df)]
                
            # Generate donut chart by creating a pie with a hole
            wedges, texts, autotexts = ax.pie(
                df[value_column],
                labels=df[category_column],
                autopct='%1.1f%%',
                colors=colors,
                shadow=False,
                startangle=90,
                wedgeprops=dict(width=0.5),  # This creates the hole
                **kwargs
            )
            
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
                
            # Add center text if provided
            if center_text or center_value is not None:
                # Create a white circle in the center for the text background
                center_circle = plt.Circle((0, 0), 0.25, fc='white')
                ax.add_artist(center_circle)
                
                # Add center text
                if center_text:
                    ax.text(0, 0.1, center_text, ha='center', va='center', fontsize=12)
                    
                # Add center value
                if center_value is not None:
                    # Format value: show whole number if integer, 1 decimal place if float
                    if isinstance(center_value, int) or center_value.is_integer():
                        value_text = f"{int(center_value)}"
                    else:
                        value_text = f"{center_value:.1f}"
                        
                    ax.text(0, -0.1, value_text, ha='center', va='center', fontsize=16, fontweight='bold')
                
            # Set title
            ax.set_title(title)
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Calculate percentages for result data
            total = df[value_column].sum()
            df['percentage'] = df[value_column] / total * 100
            
            # Prepare result
            result = {
                "type": "donut",
                "title": title,
                "category_column": category_column,
                "value_column": value_column,
                "data": df.to_dict(orient="records"),
                "total": float(total),
                "original_categories": original_categories,
                "displayed_categories": len(df),
                "center_text": center_text,
                "center_value": center_value,
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating donut chart: {str(e)}")
            raise
    
    def generate_nested_pie_chart(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        inner_category_column: str,
        outer_category_column: str,
        value_column: str,
        title: str = "Nested Pie Chart",
        inner_colors: Optional[List[str]] = None,
        outer_colors: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        sort_values: bool = True,
        inner_limit: Optional[int] = None,
        outer_limit: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a nested pie chart (pie chart with two levels).
        
        Args:
            data: DataFrame or list of dicts containing the data
            inner_category_column: Column name for inner pie categories
            outer_category_column: Column name for outer pie categories
            value_column: Column name for slice values
            title: Chart title
            inner_colors: List of colors for inner pie slices
            outer_colors: List of colors for outer pie slices
            figsize: Figure size as (width, height) tuple
            sort_values: Whether to sort by values (descending)
            inner_limit: Limit the number of inner categories shown
            outer_limit: Limit the number of outer categories per inner category
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
            
            # Aggregate data for inner and outer rings
            # For inner ring, group by inner_category_column
            inner_data = df.groupby(inner_category_column)[value_column].sum().reset_index()
            
            # Sort inner data if requested
            if sort_values:
                inner_data = inner_data.sort_values(by=value_column, ascending=False)
                
            # Apply limit to inner categories if specified
            original_inner_categories = len(inner_data)
            if inner_limit and len(inner_data) > inner_limit:
                # Group small categories into "Other"
                top_categories = inner_data.head(inner_limit - 1)
                other_sum = inner_data.iloc[inner_limit - 1:][value_column].sum()
                
                if other_sum > 0:
                    other_row = pd.DataFrame({
                        inner_category_column: ["Other"],
                        value_column: [other_sum]
                    })
                    inner_data = pd.concat([top_categories, other_row], ignore_index=True)
                else:
                    inner_data = top_categories
                    
                # Also filter the original dataframe to only include the top inner categories
                # plus re-assign the "Other" outer categories
                top_inner_categories = top_categories[inner_category_column].tolist()
                
                # Create a new dataframe with only top inner categories
                top_df = df[df[inner_category_column].isin(top_inner_categories)]
                
                # Create an "Other" outer dataset
                other_df = df[~df[inner_category_column].isin(top_inner_categories)].copy()
                if not other_df.empty:
                    other_df[inner_category_column] = "Other"
                    
                # Combine the datasets
                df = pd.concat([top_df, other_df], ignore_index=True)
            
            # For outer ring, group by both inner and outer category columns
            outer_data = df.groupby([inner_category_column, outer_category_column])[value_column].sum().reset_index()
            
            # Sort outer data if requested
            if sort_values:
                outer_data = outer_data.sort_values(by=[inner_category_column, value_column], ascending=[True, False])
            
            # Apply limit to outer categories if specified
            original_outer_categories = len(outer_data)
            if outer_limit:
                new_outer_data = []
                for inner_cat in inner_data[inner_category_column]:
                    # Get data for this inner category
                    cat_data = outer_data[outer_data[inner_category_column] == inner_cat]
                    
                    if len(cat_data) > outer_limit:
                        # Take the top N-1 categories
                        top_cats = cat_data.head(outer_limit - 1)
                        # Sum the rest as "Other"
                        other_sum = cat_data.iloc[outer_limit - 1:][value_column].sum()
                        
                        if other_sum > 0:
                            other_row = pd.DataFrame({
                                inner_category_column: [inner_cat],
                                outer_category_column: [f"Other ({inner_cat})"],
                                value_column: [other_sum]
                            })
                            cat_data = pd.concat([top_cats, other_row], ignore_index=True)
                        else:
                            cat_data = top_cats
                    
                    new_outer_data.append(cat_data)
                
                outer_data = pd.concat(new_outer_data, ignore_index=True)
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
            
            # Set up colors
            if not inner_colors:
                inner_colors = plt.cm.Set3.colors[:len(inner_data)]
                
            if not outer_colors:
                outer_colors = plt.cm.tab20.colors[:len(outer_data)]
                
            # Create inner pie chart
            inner_wedges, inner_texts = ax.pie(
                inner_data[value_column],
                labels=inner_data[inner_category_column],
                colors=inner_colors,
                radius=0.7,    # Smaller radius for inner pie
                wedgeprops=dict(width=0.4, edgecolor='w'),  # Set width and white edge
                startangle=90,
                **kwargs
            )
            
            # Create outer pie chart
            # We need to calculate the sizes and positions
            outer_sizes = outer_data[value_column].values
            
            # Calculate angles for outer pie sections
            outer_angles = []
            inner_angles = []
            
            start_angle = 90
            for inner_value in inner_data[value_column]:
                # Calculate angle for this inner category
                angle = 360 * inner_value / inner_data[value_column].sum()
                inner_angles.append((start_angle, angle))
                start_angle -= angle
            
            # Create outer pie sections for each inner category
            for i, inner_cat in enumerate(inner_data[inner_category_column]):
                # Get data for this inner category
                cat_data = outer_data[outer_data[inner_category_column] == inner_cat]
                
                if not cat_data.empty:
                    start_angle, angle = inner_angles[i]
                    end_angle = start_angle - angle
                    
                    # Calculate proportional angles for outer sections
                    cat_total = cat_data[value_column].sum()
                    cat_angles = [360 * val / cat_total * (angle / 360) for val in cat_data[value_column]]
                    
                    # Create outer pie sections
                    for j, (_, row) in enumerate(cat_data.iterrows()):
                        # Calculate angles for this section
                        section_angle = cat_angles[j]
                        section_start = start_angle - sum(cat_angles[:j])
                        section_end = section_start - section_angle
                        
                        # Create wedge
                        wedge = matplotlib.patches.Wedge(
                            center=(0, 0),
                            r=1.0,             # Outer radius
                            theta1=section_start,
                            theta2=section_end,
                            width=0.3,         # Width of outer ring
                            color=outer_colors[j % len(outer_colors)],
                            edgecolor='w'      # White edge
                        )
                        ax.add_patch(wedge)
            
            # Create legend for inner pie
            inner_legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=inner_colors[i % len(inner_colors)], 
                          markersize=10, label=cat)
                for i, cat in enumerate(inner_data[inner_category_column])
            ]
            
            # Create legend for outer pie (showing only a few to avoid clutter)
            outer_categories = outer_data[outer_category_column].unique()
            outer_legend_count = min(5, len(outer_categories))
            outer_legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=outer_colors[i % len(outer_colors)], 
                          markersize=10, label=cat if len(cat) < 15 else cat[:12] + '...')
                for i, cat in enumerate(outer_categories[:outer_legend_count])
            ]
            
            # Display legends
            inner_legend = ax.legend(
                handles=inner_legend_elements,
                loc='upper left',
                bbox_to_anchor=(1.0, 1.0),
                title=inner_category_column
            )
            
            # Add second legend
            ax.add_artist(inner_legend)
            if outer_legend_count > 0:
                ax.legend(
                    handles=outer_legend_elements,
                    loc='upper left',
                    bbox_to_anchor=(1.0, 0.7),
                    title=outer_category_column
                )
            
            # Set title
            ax.set_title(title)
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight')
            plt.close(fig)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare result data
            # Calculate percentages for result data
            inner_total = inner_data[value_column].sum()
            inner_data['percentage'] = inner_data[value_column] / inner_total * 100
            
            outer_data['percentage'] = outer_data.apply(
                lambda row: (row[value_column] / outer_data[outer_data[inner_category_column] == row[inner_category_column]][value_column].sum()) * 100,
                axis=1
            )
            
            # Prepare result
            result = {
                "type": "nested_pie",
                "title": title,
                "inner_category_column": inner_category_column,
                "outer_category_column": outer_category_column,
                "value_column": value_column,
                "inner_data": inner_data.to_dict(orient="records"),
                "outer_data": outer_data.to_dict(orient="records"),
                "total": float(inner_total),
                "original_inner_categories": original_inner_categories,
                "displayed_inner_categories": len(inner_data),
                "original_outer_categories": original_outer_categories,
                "displayed_outer_categories": len(outer_data),
                "image": f"data:image/png;base64,{image_base64}",
                "generated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating nested pie chart: {str(e)}")
            raise

# Import matplotlib patches for custom wedges
import matplotlib.patches as matplotlib_patches