"""
Dashboard repository for dashboard management operations
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID
import logging

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, or_, and_, desc
from sqlalchemy.exc import IntegrityError

from app.models import Dashboard, DashboardChart, Chart, User

logger = logging.getLogger(__name__)


class DashboardRepository:
    """Repository for dashboard-related database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # =====================================================
    # Dashboard CRUD Operations
    # =====================================================
    
    def create_dashboard(
        self,
        name: str,
        created_by: UUID,
        description: Optional[str] = None,
        layout_config: Optional[Dict[str, Any]] = None,
        theme: str = 'default',
        refresh_interval: Optional[int] = None,
        is_public: bool = False,
        tags: Optional[List[str]] = None
    ) -> Dashboard:
        """Create a new dashboard"""
        dashboard = Dashboard(
            name=name,
            description=description,
            layout_config=layout_config or {"grid": {"cols": 12, "rowHeight": 100}},
            theme=theme,
            refresh_interval=refresh_interval,
            is_public=is_public,
            tags=tags or [],
            created_by=created_by
        )
        
        self.db.add(dashboard)
        self.db.commit()
        self.db.refresh(dashboard)
        return dashboard
    
    def get_dashboard_by_id(
        self,
        dashboard_id: UUID,
        include_charts: bool = True
    ) -> Optional[Dashboard]:
        """Get dashboard by ID"""
        query = self.db.query(Dashboard)
        
        if include_charts:
            query = query.options(
                joinedload(Dashboard.charts).joinedload(DashboardChart.chart).joinedload(Chart.chart_type),
                joinedload(Dashboard.created_by_user)
            )
        
        return query.filter(Dashboard.id == dashboard_id).first()
    
    def get_dashboards(
        self,
        user_id: Optional[UUID] = None,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public_only: bool = False
    ) -> Tuple[List[Dashboard], int]:
        """Get dashboards with filters"""
        query = self.db.query(Dashboard)
        
        # Filter by access
        if is_public_only:
            query = query.filter(Dashboard.is_public == True)
        elif user_id:
            query = query.filter(
                or_(
                    Dashboard.created_by == user_id,
                    Dashboard.is_public == True
                )
            )
        
        # Apply filters
        if search:
            search_filter = f"%{search}%"
            query = query.filter(
                or_(
                    Dashboard.name.ilike(search_filter),
                    Dashboard.description.ilike(search_filter)
                )
            )
        
        if tags:
            query = query.filter(Dashboard.tags.overlap(tags))
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        dashboards = query.order_by(
            desc(Dashboard.created_at)
        ).offset(skip).limit(limit).all()
        
        return dashboards, total
    
    def update_dashboard(
        self,
        dashboard_id: UUID,
        user_id: UUID,
        update_data: Dict[str, Any]
    ) -> Optional[Dashboard]:
        """Update dashboard configuration"""
        dashboard = self.get_dashboard_by_id(dashboard_id, include_charts=False)
        if not dashboard or dashboard.created_by != user_id:
            return None
        
        # Update allowed fields
        allowed_fields = [
            'name', 'description', 'layout_config', 'theme',
            'refresh_interval', 'is_public', 'tags'
        ]
        
        for key, value in update_data.items():
            if key in allowed_fields and hasattr(dashboard, key):
                setattr(dashboard, key, value)
        
        dashboard.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(dashboard)
        return dashboard
    
    def delete_dashboard(self, dashboard_id: UUID, user_id: UUID) -> bool:
        """Delete dashboard"""
        dashboard = self.get_dashboard_by_id(dashboard_id, include_charts=False)
        if not dashboard or dashboard.created_by != user_id:
            return False
        
        self.db.delete(dashboard)
        self.db.commit()
        return True
    
    def duplicate_dashboard(
        self,
        dashboard_id: UUID,
        user_id: UUID,
        new_name: Optional[str] = None
    ) -> Optional[Dashboard]:
        """Create a copy of an existing dashboard"""
        original = self.get_dashboard_by_id(dashboard_id)
        if not original or (original.created_by != user_id and not original.is_public):
            return None
        
        # Create new dashboard
        new_dashboard = Dashboard(
            name=new_name or f"{original.name} (Copy)",
            description=original.description,
            layout_config=original.layout_config.copy() if original.layout_config else {},
            theme=original.theme,
            refresh_interval=original.refresh_interval,
            is_public=False,  # Copies are private by default
            tags=original.tags.copy() if original.tags else [],
            created_by=user_id
        )
        
        self.db.add(new_dashboard)
        self.db.flush()  # Get ID without committing
        
        # Copy charts
        for dc in original.charts:
            new_dc = DashboardChart(
                dashboard_id=new_dashboard.id,
                chart_id=dc.chart_id,
                position=dc.position.copy() if dc.position else {},
                config_overrides=dc.config_overrides.copy() if dc.config_overrides else {},
                display_order=dc.display_order
            )
            self.db.add(new_dc)
        
        self.db.commit()
        self.db.refresh(new_dashboard)
        return new_dashboard
    
    # =====================================================
    # Dashboard Chart Operations
    # =====================================================
    
    def add_chart_to_dashboard(
        self,
        dashboard_id: UUID,
        chart_id: UUID,
        position: Dict[str, int],
        user_id: UUID,
        config_overrides: Optional[Dict[str, Any]] = None,
        display_order: Optional[int] = None
    ) -> Optional[DashboardChart]:
        """Add chart to dashboard"""
        # Verify dashboard ownership
        dashboard = self.get_dashboard_by_id(dashboard_id, include_charts=False)
        if not dashboard or dashboard.created_by != user_id:
            return None
        
        # Verify chart exists and is accessible
        chart = self.db.query(Chart).filter(
            Chart.id == chart_id,
            or_(
                Chart.created_by == user_id,
                Chart.is_public == True
            )
        ).first()
        
        if not chart:
            return None
        
        # Check if chart already in dashboard
        existing = self.db.query(DashboardChart).filter(
            DashboardChart.dashboard_id == dashboard_id,
            DashboardChart.chart_id == chart_id
        ).first()
        
        if existing:
            # Update position
            existing.position = position
            if config_overrides is not None:
                existing.config_overrides = config_overrides
            if display_order is not None:
                existing.display_order = display_order
            dashboard_chart = existing
        else:
            # Get next display order if not provided
            if display_order is None:
                max_order = self.db.query(
                    func.max(DashboardChart.display_order)
                ).filter(
                    DashboardChart.dashboard_id == dashboard_id
                ).scalar() or 0
                display_order = max_order + 1
            
            dashboard_chart = DashboardChart(
                dashboard_id=dashboard_id,
                chart_id=chart_id,
                position=position,
                config_overrides=config_overrides or {},
                display_order=display_order
            )
            self.db.add(dashboard_chart)
        
        dashboard.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(dashboard_chart)
        return dashboard_chart
    
    def update_chart_position(
        self,
        dashboard_id: UUID,
        chart_id: UUID,
        new_position: Dict[str, int],
        user_id: UUID
    ) -> bool:
        """Update chart position in dashboard"""
        # Verify dashboard ownership
        dashboard = self.get_dashboard_by_id(dashboard_id, include_charts=False)
        if not dashboard or dashboard.created_by != user_id:
            return False
        
        dashboard_chart = self.db.query(DashboardChart).filter(
            DashboardChart.dashboard_id == dashboard_id,
            DashboardChart.chart_id == chart_id
        ).first()
        
        if not dashboard_chart:
            return False
        
        dashboard_chart.position = new_position
        dashboard.updated_at = datetime.utcnow()
        self.db.commit()
        return True
    
    def update_dashboard_layout(
        self,
        dashboard_id: UUID,
        chart_positions: List[Dict[str, Any]],
        user_id: UUID
    ) -> bool:
        """Update positions of multiple charts in dashboard"""
        # Verify dashboard ownership
        dashboard = self.get_dashboard_by_id(dashboard_id, include_charts=False)
        if not dashboard or dashboard.created_by != user_id:
            return False
        
        # Update each chart position
        for pos_data in chart_positions:
            chart_id = pos_data.get('chart_id')
            position = pos_data.get('position')
            
            if chart_id and position:
                dashboard_chart = self.db.query(DashboardChart).filter(
                    DashboardChart.dashboard_id == dashboard_id,
                    DashboardChart.chart_id == chart_id
                ).first()
                
                if dashboard_chart:
                    dashboard_chart.position = position
        
        dashboard.updated_at = datetime.utcnow()
        self.db.commit()
        return True
    
    def remove_chart_from_dashboard(
        self,
        dashboard_id: UUID,
        chart_id: UUID,
        user_id: UUID
    ) -> bool:
        """Remove chart from dashboard"""
        # Verify dashboard ownership
        dashboard = self.get_dashboard_by_id(dashboard_id, include_charts=False)
        if not dashboard or dashboard.created_by != user_id:
            return False
        
        dashboard_chart = self.db.query(DashboardChart).filter(
            DashboardChart.dashboard_id == dashboard_id,
            DashboardChart.chart_id == chart_id
        ).first()
        
        if not dashboard_chart:
            return False
        
        self.db.delete(dashboard_chart)
        dashboard.updated_at = datetime.utcnow()
        self.db.commit()
        return True
    
    def reorder_dashboard_charts(
        self,
        dashboard_id: UUID,
        chart_order: List[UUID],
        user_id: UUID
    ) -> bool:
        """Reorder charts in dashboard"""
        # Verify dashboard ownership
        dashboard = self.get_dashboard_by_id(dashboard_id, include_charts=False)
        if not dashboard or dashboard.created_by != user_id:
            return False
        
        # Update display order for each chart
        for idx, chart_id in enumerate(chart_order):
            dashboard_chart = self.db.query(DashboardChart).filter(
                DashboardChart.dashboard_id == dashboard_id,
                DashboardChart.chart_id == chart_id
            ).first()
            
            if dashboard_chart:
                dashboard_chart.display_order = idx
        
        dashboard.updated_at = datetime.utcnow()
        self.db.commit()
        return True
    
    # =====================================================
    # Dashboard Sharing and Access
    # =====================================================
    
    def set_default_dashboard(
        self,
        dashboard_id: UUID,
        user_id: UUID
    ) -> bool:
        """Set dashboard as default for user"""
        # First, unset any existing default
        self.db.query(Dashboard).filter(
            Dashboard.created_by == user_id,
            Dashboard.is_default == True
        ).update({"is_default": False})
        
        # Set new default
        dashboard = self.get_dashboard_by_id(dashboard_id, include_charts=False)
        if not dashboard or dashboard.created_by != user_id:
            return False
        
        dashboard.is_default = True
        self.db.commit()
        return True
    
    def get_default_dashboard(self, user_id: UUID) -> Optional[Dashboard]:
        """Get user's default dashboard"""
        return self.db.query(Dashboard).filter(
            Dashboard.created_by == user_id,
            Dashboard.is_default == True
        ).first()
    
    def share_dashboard(
        self,
        dashboard_id: UUID,
        user_id: UUID,
        make_public: bool = True
    ) -> bool:
        """Share dashboard (make public)"""
        dashboard = self.get_dashboard_by_id(dashboard_id, include_charts=False)
        if not dashboard or dashboard.created_by != user_id:
            return False
        
        dashboard.is_public = make_public
        dashboard.updated_at = datetime.utcnow()
        self.db.commit()
        return True
    
    # =====================================================
    # Dashboard Analytics
    # =====================================================
    
    def get_dashboard_statistics(
        self,
        user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Get dashboard usage statistics"""
        query = self.db.query(Dashboard)
        if user_id:
            query = query.filter(Dashboard.created_by == user_id)
        
        total_dashboards = query.count()
        
        # Public vs private
        public_dashboards = query.filter(Dashboard.is_public == True).count()
        
        # Average charts per dashboard
        avg_charts = self.db.query(
            func.avg(
                self.db.query(func.count(DashboardChart.id))
                .filter(DashboardChart.dashboard_id == Dashboard.id)
                .scalar_subquery()
            )
        ).scalar() or 0
        
        # Dashboards by theme
        themes = dict(
            self.db.query(
                Dashboard.theme,
                func.count(Dashboard.id)
            ).group_by(Dashboard.theme).all()
        )
        
        # Recent activity
        recent_dashboards = query.filter(
            Dashboard.updated_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
        ).count()
        
        return {
            "total_dashboards": total_dashboards,
            "public_dashboards": public_dashboards,
            "private_dashboards": total_dashboards - public_dashboards,
            "average_charts_per_dashboard": float(avg_charts),
            "themes_distribution": themes,
            "dashboards_updated_today": recent_dashboards
        }
    
    def get_popular_dashboards(
        self,
        limit: int = 10,
        days: int = 30
    ) -> List[Dashboard]:
        """Get most viewed/accessed dashboards"""
        # In a full implementation, this would track actual views
        # For now, return most recently updated public dashboards
        since_date = datetime.utcnow() - timedelta(days=days)
        
        return self.db.query(Dashboard).filter(
            Dashboard.is_public == True,
            Dashboard.updated_at >= since_date
        ).order_by(
            desc(Dashboard.updated_at)
        ).limit(limit).all()
    
    def get_dashboard_chart_types(
        self,
        dashboard_id: UUID
    ) -> Dict[str, int]:
        """Get distribution of chart types in a dashboard"""
        from app.models import ChartType
        
        chart_types = self.db.query(
            ChartType.name,
            func.count(DashboardChart.id)
        ).join(
            Chart, Chart.chart_type_id == ChartType.id
        ).join(
            DashboardChart, DashboardChart.chart_id == Chart.id
        ).filter(
            DashboardChart.dashboard_id == dashboard_id
        ).group_by(
            ChartType.name
        ).all()
        
        return dict(chart_types)