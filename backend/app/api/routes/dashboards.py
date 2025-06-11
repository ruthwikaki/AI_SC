# backend/app/api/routes/dashboards.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
import json

from app.db.database import get_db
from app.api.routes.auth import get_current_active_user
from app.models.user import User
from app.models.visualization import Dashboard, DashboardWidget
from app.models.extended_models import WidgetType
from app.db.repositories.dashboard_repository import DashboardRepository
from app.schemas.visualization import (
    DashboardCreateRequest,
    DashboardUpdateRequest,
    DashboardResponse,
    WidgetCreateRequest,
    WidgetResponse
)

router = APIRouter(prefix="/api/dashboards", tags=["dashboards"])

@router.get("/", response_model=List[DashboardResponse])
async def get_dashboards(
    shared: Optional[bool] = Query(None, description="Filter by shared status"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get user's dashboards"""
    try:
        repo = DashboardRepository(db)
        dashboards = repo.get_user_dashboards(current_user.id, shared)
        
        return [
            DashboardResponse(
                id=d.id,
                name=d.name,
                description=d.description,
                is_public=d.is_public,
                is_default=d.is_default,
                layout_config=json.loads(d.layout_config) if d.layout_config else {},
                created_at=d.created_at,
                updated_at=d.updated_at,
                widgets_count=len(d.widgets)
            )
            for d in dashboards
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=DashboardResponse)
async def create_dashboard(
    dashboard: DashboardCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new dashboard"""
    try:
        repo = DashboardRepository(db)
        
        # Create dashboard
        new_dashboard = repo.create_dashboard(
            user_id=current_user.id,
            name=dashboard.name,
            description=dashboard.description,
            is_public=dashboard.is_public,
            layout_config=dashboard.layout_config
        )
        
        # Add widgets if provided
        if dashboard.widgets:
            for widget in dashboard.widgets:
                repo.add_widget_to_dashboard(
                    dashboard_id=new_dashboard.id,
                    widget_type=widget.type,
                    config=widget.config,
                    position=widget.position
                )
        
        db.commit()
        db.refresh(new_dashboard)
        
        return DashboardResponse(
            id=new_dashboard.id,
            name=new_dashboard.name,
            description=new_dashboard.description,
            is_public=new_dashboard.is_public,
            is_default=new_dashboard.is_default,
            layout_config=json.loads(new_dashboard.layout_config) if new_dashboard.layout_config else {},
            created_at=new_dashboard.created_at,
            updated_at=new_dashboard.updated_at,
            widgets_count=len(new_dashboard.widgets)
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dashboard_id}", response_model=Dict[str, Any])
async def get_dashboard_details(
    dashboard_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get dashboard with all widgets"""
    try:
        repo = DashboardRepository(db)
        dashboard = repo.get_dashboard_by_id(dashboard_id)
        
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        # Check access
        if dashboard.user_id != current_user.id and not dashboard.is_public:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get widgets with data
        widgets = []
        for widget in dashboard.widgets:
            widget_data = await _get_widget_data(widget, db)
            widgets.append({
                "id": widget.id,
                "type": widget.widget_type,
                "title": widget.title,
                "config": json.loads(widget.config) if widget.config else {},
                "position": json.loads(widget.position) if widget.position else {},
                "data": widget_data,
                "last_updated": widget.updated_at
            })
        
        return {
            "dashboard": {
                "id": dashboard.id,
                "name": dashboard.name,
                "description": dashboard.description,
                "is_public": dashboard.is_public,
                "is_default": dashboard.is_default,
                "layout_config": json.loads(dashboard.layout_config) if dashboard.layout_config else {},
                "created_at": dashboard.created_at,
                "updated_at": dashboard.updated_at
            },
            "widgets": widgets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{dashboard_id}", response_model=DashboardResponse)
async def update_dashboard(
    dashboard_id: int,
    update_data: DashboardUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update dashboard configuration"""
    try:
        repo = DashboardRepository(db)
        dashboard = repo.get_dashboard_by_id(dashboard_id)
        
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        if dashboard.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update dashboard
        if update_data.name:
            dashboard.name = update_data.name
        if update_data.description is not None:
            dashboard.description = update_data.description
        if update_data.is_public is not None:
            dashboard.is_public = update_data.is_public
        if update_data.layout_config:
            dashboard.layout_config = json.dumps(update_data.layout_config)
        
        dashboard.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(dashboard)
        
        return DashboardResponse(
            id=dashboard.id,
            name=dashboard.name,
            description=dashboard.description,
            is_public=dashboard.is_public,
            is_default=dashboard.is_default,
            layout_config=json.loads(dashboard.layout_config) if dashboard.layout_config else {},
            created_at=dashboard.created_at,
            updated_at=dashboard.updated_at,
            widgets_count=len(dashboard.widgets)
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{dashboard_id}")
async def delete_dashboard(
    dashboard_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete dashboard"""
    try:
        repo = DashboardRepository(db)
        dashboard = repo.get_dashboard_by_id(dashboard_id)
        
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        if dashboard.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if dashboard.is_default:
            raise HTTPException(status_code=400, detail="Cannot delete default dashboard")
        
        db.delete(dashboard)
        db.commit()
        
        return {"message": "Dashboard deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{dashboard_id}/widgets", response_model=WidgetResponse)
async def add_widget(
    dashboard_id: int,
    widget: WidgetCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Add widget to dashboard"""
    try:
        repo = DashboardRepository(db)
        dashboard = repo.get_dashboard_by_id(dashboard_id)
        
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        if dashboard.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create widget
        new_widget = DashboardWidget(
            dashboard_id=dashboard_id,
            widget_type=widget.type,
            title=widget.title,
            config=json.dumps(widget.config) if widget.config else '{}',
            position=json.dumps(widget.position) if widget.position else '{}',
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(new_widget)
        db.commit()
        db.refresh(new_widget)
        
        return WidgetResponse(
            id=new_widget.id,
            type=new_widget.widget_type,
            title=new_widget.title,
            config=json.loads(new_widget.config) if new_widget.config else {},
            position=json.loads(new_widget.position) if new_widget.position else {},
            created_at=new_widget.created_at
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{dashboard_id}/widgets/{widget_id}")
async def update_widget(
    dashboard_id: int,
    widget_id: int,
    widget_update: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update widget configuration"""
    try:
        # Verify dashboard ownership
        dashboard = db.query(Dashboard).filter(
            Dashboard.id == dashboard_id,
            Dashboard.user_id == current_user.id
        ).first()
        
        if not dashboard:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update widget
        widget = db.query(DashboardWidget).filter(
            DashboardWidget.id == widget_id,
            DashboardWidget.dashboard_id == dashboard_id
        ).first()
        
        if not widget:
            raise HTTPException(status_code=404, detail="Widget not found")
        
        if 'title' in widget_update:
            widget.title = widget_update['title']
        if 'config' in widget_update:
            widget.config = json.dumps(widget_update['config'])
        if 'position' in widget_update:
            widget.position = json.dumps(widget_update['position'])
        
        widget.updated_at = datetime.utcnow()
        db.commit()
        
        return {"message": "Widget updated successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{dashboard_id}/widgets/{widget_id}")
async def remove_widget(
    dashboard_id: int,
    widget_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Remove widget from dashboard"""
    try:
        # Verify dashboard ownership
        dashboard = db.query(Dashboard).filter(
            Dashboard.id == dashboard_id,
            Dashboard.user_id == current_user.id
        ).first()
        
        if not dashboard:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete widget
        widget = db.query(DashboardWidget).filter(
            DashboardWidget.id == widget_id,
            DashboardWidget.dashboard_id == dashboard_id
        ).first()
        
        if not widget:
            raise HTTPException(status_code=404, detail="Widget not found")
        
        db.delete(widget)
        db.commit()
        
        return {"message": "Widget removed successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/widget-types", response_model=List[Dict[str, Any]])
async def get_available_widget_types(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get available widget types"""
    try:
        widget_types = db.query(WidgetType).filter(WidgetType.is_active == True).all()
        
        return [
            {
                "id": wt.id,
                "name": wt.name,
                "display_name": wt.display_name,
                "category": wt.category,
                "description": wt.description,
                "default_config": json.loads(wt.default_config) if wt.default_config else {},
                "preview_image": wt.preview_image
            }
            for wt in widget_types
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _get_widget_data(widget: DashboardWidget, db: Session) -> Dict[str, Any]:
    """Fetch data for a specific widget based on its type and configuration"""
    config = json.loads(widget.config) if widget.config else {}
    
    # Implementation would fetch actual data based on widget type
    # This is a placeholder that returns sample data structure
    return {
        "values": [],
        "last_updated": datetime.utcnow()
    }