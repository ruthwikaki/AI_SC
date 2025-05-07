from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Query
from fastapi.responses import FileResponse
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
from datetime import datetime
import json
import uuid
from io import BytesIO

from app.visualization.chart_generators.bar_chart import generate_bar_chart
from app.visualization.chart_generators.line_chart import generate_line_chart
from app.visualization.chart_generators.pie_chart import generate_pie_chart
from app.visualization.chart_generators.heatmap import generate_heatmap
from app.visualization.chart_generators.sankey import generate_sankey_diagram
from app.visualization.chart_generators.network_graph import generate_network_graph
from app.visualization.recommendation_engine import recommend_chart_type
from app.visualization.export_manager import export_chart
from app.utils.logger import get_logger
from app.db.interfaces.user_interface import User

from app.api.routes.auth import get_current_active_user
from app.security.rbac_manager import check_permission

# Initialize logger
logger = get_logger(__name__)

# Router
router = APIRouter(
    prefix="/visualizations",
    tags=["visualizations"],
    dependencies=[Depends(get_current_active_user)],
    responses={401: {"description": "Unauthorized"}}
)

# Models
class DataPoint(BaseModel):
    """Generic data point for chart data"""
    x: Union[str, int, float]
    y: Union[int, float]
    category: Optional[str] = None
    series: Optional[str] = None
    tooltip: Optional[str] = None

class ChartData(BaseModel):
    """Data for chart generation"""
    type: str
    data: List[Dict[str, Any]]  # Generic format to accommodate various chart types
    title: str
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    categories: Optional[List[str]] = None
    series: Optional[List[str]] = None
    color_scale: Optional[List[str]] = None
    width: Optional[int] = 800
    height: Optional[int] = 500
    show_legend: bool = True
    animation: bool = True
    interactive: bool = True
    theme: str = "light"
    additional_options: Optional[Dict[str, Any]] = None
    client_id: Optional[str] = None

class ChartResponse(BaseModel):
    """Response containing chart data and metadata"""
    chart_id: str
    chart_url: str
    chart_type: str
    created_at: datetime
    created_by: str
    title: str
    width: int
    height: int
    format: str = "svg"
    data_summary: Dict[str, Any]
    client_id: str

class SavedChart(BaseModel):
    """Saved chart configuration"""
    id: str
    name: str
    description: Optional[str] = None
    chart_config: ChartData
    created_by: str
    created_at: datetime
    last_viewed: Optional[datetime] = None
    view_count: int = 0
    is_public: bool = False
    tags: List[str] = []
    client_id: str

class Dashboard(BaseModel):
    """Dashboard configuration"""
    id: str
    name: str
    description: Optional[str] = None
    layout: List[Dict[str, Any]]  # Dashboard layout configuration
    charts: List[str]  # List of chart IDs
    created_by: str
    created_at: datetime
    last_viewed: Optional[datetime] = None
    is_public: bool = False
    tags: List[str] = []
    client_id: str

# Helper functions
def get_chart_generator(chart_type: str):
    """Get the appropriate chart generator function based on chart type"""
    chart_generators = {
        "bar": generate_bar_chart,
        "line": generate_line_chart,
        "pie": generate_pie_chart,
        "heatmap": generate_heatmap,
        "sankey": generate_sankey_diagram,
        "network": generate_network_graph
    }
    
    if chart_type not in chart_generators:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    return chart_generators[chart_type]

# Routes
@router.post("/generate", response_model=ChartResponse)
async def generate_chart(
    chart_data: ChartData,
    current_user: User = Depends(get_current_active_user)
):
    """Generate a chart from provided data"""
    # Check user has permission
    check_permission(current_user.role, "visualizations:generate")
    
    # Ensure client_id is set
    client_id = chart_data.client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get appropriate chart generator
        chart_gen = get_chart_generator(chart_data.type)
        
        # Generate chart ID
        chart_id = f"chart-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        
        # Generate chart
        chart_path = await chart_gen(
            data=chart_data.data,
            title=chart_data.title,
            x_axis_label=chart_data.x_axis_label,
            y_axis_label=chart_data.y_axis_label,
            categories=chart_data.categories,
            series=chart_data.series,
            color_scale=chart_data.color_scale,
            width=chart_data.width or 800,
            height=chart_data.height or 500,
            show_legend=chart_data.show_legend,
            animation=chart_data.animation,
            theme=chart_data.theme,
            additional_options=chart_data.additional_options,
            chart_id=chart_id,
            client_id=client_id
        )
        
        # Calculate data summary
        data_summary = {
            "record_count": len(chart_data.data),
            "series_count": len(chart_data.series) if chart_data.series else 1,
            "min_value": min([p.get('y', 0) for p in chart_data.data if 'y' in p], default=0),
            "max_value": max([p.get('y', 0) for p in chart_data.data if 'y' in p], default=0),
            "avg_value": sum([p.get('y', 0) for p in chart_data.data if 'y' in p], default=0) / 
                        len([p.get('y', 0) for p in chart_data.data if 'y' in p]) if len(chart_data.data) > 0 else 0
        }
        
        # Create response
        response = ChartResponse(
            chart_id=chart_id,
            chart_url=f"/api/visualizations/charts/{chart_id}",
            chart_type=chart_data.type,
            created_at=datetime.now(),
            created_by=current_user.id,
            title=chart_data.title,
            width=chart_data.width or 800,
            height=chart_data.height or 500,
            data_summary=data_summary,
            client_id=client_id
        )
        
        # Save chart metadata to database
        # In a real implementation, this would store the chart metadata for future retrieval
        
        logger.info(f"Generated {chart_data.type} chart: {chart_data.title}")
        return response
    
    except ValueError as ve:
        logger.error(f"Chart generation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Chart generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating chart: {str(e)}"
        )

@router.get("/recommend", response_model=Dict[str, Any])
async def recommend_visualization(
    query_results: Optional[str] = None,  # JSON string of query results
    query_text: Optional[str] = None,     # Natural language query text
    current_user: User = Depends(get_current_active_user)
):
    """Recommend visualization types based on the data structure and query"""
    # Check user has permission
    check_permission(current_user.role, "visualizations:recommend")
    
    try:
        # Parse the query results if provided
        data = None
        if query_results:
            data = json.loads(query_results)
        
        # Use recommendation engine to suggest chart types
        from app.visualization.recommendation_engine import recommend_chart_type
        recommendations = recommend_chart_type(query_text, data)
        
        logger.info(f"Generated visualization recommendations")
        return {"recommendations": recommendations}
    
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON data"
        )
    except Exception as e:
        logger.error(f"Error recommending visualizations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recommending visualizations: {str(e)}"
        )

@router.get("/charts/{chart_id}")
async def get_chart(
    chart_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific chart by ID"""
    # Check user has permission
    check_permission(current_user.role, "visualizations:view")
    
    try:
        # Get chart metadata from database
        # In a real implementation, you would retrieve chart details
        # and check if user has access to this chart
        
        # Get chart file
        from app.visualization.chart_generators.chart_storage import get_chart_file
        file_path, format = await get_chart_file(chart_id)
        
        if not file_path:
            logger.warning(f"Chart not found: {chart_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chart not found"
            )
        
        # Set appropriate media type based on format
        media_types = {
            "svg": "image/svg+xml",
            "png": "image/png",
            "json": "application/json"
        }
        
        return FileResponse(file_path, media_type=media_types.get(format, "application/octet-stream"))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chart: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chart: {str(e)}"
        )

@router.post("/export/{chart_id}")
async def export_chart_endpoint(
    chart_id: str,
    format: str = Query(..., description="Export format (png, svg, pdf, csv, excel)"),
    current_user: User = Depends(get_current_active_user)
):
    """Export a chart to a specific format"""
    # Check user has permission
    check_permission(current_user.role, "visualizations:export")
    
    valid_formats = ["png", "svg", "pdf", "csv", "excel"]
    if format not in valid_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format. Supported formats: {', '.join(valid_formats)}"
        )
    
    try:
        # Export chart to requested format
        from app.visualization.export_manager import export_chart
        file_path = await export_chart(chart_id, format)
        
        if not file_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chart not found"
            )
        
        # Generate download file name
        file_name = f"{chart_id}.{format}"
        
        # Set appropriate media type
        media_types = {
            "png": "image/png",
            "svg": "image/svg+xml",
            "pdf": "application/pdf",
            "csv": "text/csv",
            "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
        
        logger.info(f"Exported chart {chart_id} to {format}")
        return FileResponse(
            path=file_path,
            filename=file_name,
            media_type=media_types.get(format)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting chart: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error exporting chart: {str(e)}"
        )

@router.get("/saved", response_model=List[SavedChart])
async def get_saved_charts(
    client_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get all saved charts for the current user"""
    # Check user has permission
    check_permission(current_user.role, "visualizations:view")
    
    # Use provided client_id or user's client_id
    client_id = client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get saved charts from database
        # In a real implementation, this would query your database
        # For now, we'll return mock data
        
        # Create mock chart config
        mock_chart_data = ChartData(
            type="bar",
            data=[{"x": "Category A", "y": 10}, {"x": "Category B", "y": 20}],
            title="Sample Chart",
            x_axis_label="Category",
            y_axis_label="Value",
            client_id=client_id
        )
        
        # Create mock saved charts
        mock_saved_charts = [
            SavedChart(
                id=f"sc-{uuid.uuid4().hex[:8]}",
                name="Inventory Levels",
                description="Current inventory levels by product category",
                chart_config=mock_chart_data,
                created_by=current_user.id,
                created_at=datetime.now(),
                last_viewed=datetime.now(),
                view_count=5,
                is_public=True,
                tags=["inventory", "product"],
                client_id=client_id
            ),
            SavedChart(
                id=f"sc-{uuid.uuid4().hex[:8]}",
                name="Supplier Performance",
                description="On-time delivery performance by supplier",
                chart_config=mock_chart_data,
                created_by=current_user.id,
                created_at=datetime.now(),
                last_viewed=datetime.now(),
                view_count=3,
                is_public=False,
                tags=["supplier", "performance"],
                client_id=client_id
            )
        ]
        
        logger.info(f"Retrieved saved charts for user: {current_user.username}")
        return mock_saved_charts
    
    except Exception as e:
        logger.error(f"Error retrieving saved charts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving saved charts: {str(e)}"
        )

@router.get("/dashboards", response_model=List[Dashboard])
async def get_dashboards(
    client_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get all dashboards for the current user"""
    # Check user has permission
    check_permission(current_user.role, "visualizations:view")
    
    # Use provided client_id or user's client_id
    client_id = client_id or current_user.client_id
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get dashboards from database
        # In a real implementation, this would query your database
        # For now, we'll return mock data
        
        # Create mock dashboards
        mock_dashboards = [
            Dashboard(
                id=f"db-{uuid.uuid4().hex[:8]}",
                name="Supply Chain Overview",
                description="Key supply chain metrics and KPIs",
                layout=[
                    {"i": "chart1", "x": 0, "y": 0, "w": 6, "h": 4},
                    {"i": "chart2", "x": 6, "y": 0, "w": 6, "h": 4},
                    {"i": "chart3", "x": 0, "y": 4, "w": 12, "h": 4}
                ],
                charts=[f"sc-{uuid.uuid4().hex[:8]}" for _ in range(3)],
                created_by=current_user.id,
                created_at=datetime.now(),
                last_viewed=datetime.now(),
                is_public=True,
                tags=["overview", "kpi"],
                client_id=client_id
            ),
            Dashboard(
                id=f"db-{uuid.uuid4().hex[:8]}",
                name="Supplier Analysis",
                description="Supplier performance and risk analysis",
                layout=[
                    {"i": "chart1", "x": 0, "y": 0, "w": 12, "h": 4},
                    {"i": "chart2", "x": 0, "y": 4, "w": 6, "h": 4},
                    {"i": "chart3", "x": 6, "y": 4, "w": 6, "h": 4}
                ],
                charts=[f"sc-{uuid.uuid4().hex[:8]}" for _ in range(3)],
                created_by=current_user.id,
                created_at=datetime.now(),
                last_viewed=datetime.now(),
                is_public=False,
                tags=["supplier", "performance", "risk"],
                client_id=client_id
            )
        ]
        
        logger.info(f"Retrieved dashboards for user: {current_user.username}")
        return mock_dashboards
    
    except Exception as e:
        logger.error(f"Error retrieving dashboards: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving dashboards: {str(e)}"
        )

@router.post("/save-dashboard", response_model=Dashboard)
async def save_dashboard(
    dashboard: Dashboard,
    current_user: User = Depends(get_current_active_user)
):
    """Save a dashboard configuration"""
    # Check user has permission
    check_permission(current_user.role, "visualizations:save")
    
    # Ensure client_id is set
    if not dashboard.client_id:
        dashboard.client_id = current_user.client_id
        
    if not dashboard.client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Generate ID if not provided
        if not dashboard.id:
            dashboard.id = f"db-{uuid.uuid4().hex}"
        
        # Set created_by and created_at if not provided
        dashboard.created_by = dashboard.created_by or current_user.id
        dashboard.created_at = dashboard.created_at or datetime.now()
        
        # Save dashboard to database
        # In a real implementation, this would save to your database
        
        logger.info(f"Dashboard saved: {dashboard.name}")
        return dashboard
    
    except Exception as e:
        logger.error(f"Error saving dashboard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving dashboard: {str(e)}"
        )

@router.delete("/dashboards/{dashboard_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dashboard(
    dashboard_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a dashboard"""
    # Check user has permission
    check_permission(current_user.role, "visualizations:delete")
    
    try:
        # Delete dashboard from database
        # In a real implementation, this would delete from your database
        # and check if the user has permission to delete this dashboard
        
        logger.info(f"Dashboard deleted: {dashboard_id}")
        return None
    
    except Exception as e:
        logger.error(f"Error deleting dashboard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting dashboard: {str(e)}"
        )