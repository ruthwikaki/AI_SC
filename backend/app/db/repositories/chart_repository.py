# Models should be imported inside methods to avoid circular imports
# Example:
# def get_something(self):
#     from app.models import User  # Import here, not at module level
#     return User.query.all()
"""
Chart repository for visualization management operations
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID
import logging

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, or_, and_, desc
from sqlalchemy.exc import IntegrityError

# MOVED TO METHOD LEVEL: from app.models import (
    Chart, ChartType, SavedChart, Dashboard, DashboardChart,
    NaturalLanguageQuery, User
)

logger = logging.getLogger(__name__)


class ChartRepository:
    """Repository for chart and visualization operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # =====================================================
    # Chart Type Operations
    # =====================================================
    
    def get_chart_types(self, is_active: bool = True) -> List[ChartType]:
        """Get available chart types"""
        query = self.db.query(ChartType)
        if is_active is not None:
            query = query.filter(ChartType.is_active == is_active)
        return query.order_by(ChartType.name).all()
    
    def get_chart_type_by_name(self, name: str) -> Optional[ChartType]:
        """Get chart type by name"""
        return self.db.query(ChartType).filter(
            ChartType.name == name
        ).first()
    
    def get_recommended_chart_types(
        self,
        data_type: str,
        data_points: int
    ) -> List[ChartType]:
        """Get recommended chart types based on data characteristics"""
        query = self.db.query(ChartType).filter(
            ChartType.is_active == True
        )
        
        # Filter by data type support
        if data_type:
            query = query.filter(
                ChartType.supported_data_types.any(data_type)
            )
        
        # Filter by data point limits
        query = query.filter(
            or_(
                ChartType.min_data_points.is_(None),
                ChartType.min_data_points <= data_points
            ),
            or_(
                ChartType.max_data_points.is_(None),
                ChartType.max_data_points >= data_points
            )
        )
        
        return query.all()
    
    # =====================================================
    # Chart CRUD Operations
    # =====================================================
    
    def create_chart(
        self,
        title: str,
        chart_type_id: UUID,
        data_source: Dict[str, Any],
        created_by: UUID,
        description: Optional[str] = None,
        query_id: Optional[UUID] = None,
        config: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        is_public: bool = False
    ) -> Chart:
        """Create a new chart"""
        chart = Chart(
            title=title,
            description=description,
            chart_type_id=chart_type_id,
            query_id=query_id,
            data_source=data_source,
            config=config or {},
            filters=filters or {},
            is_public=is_public,
            created_by=created_by
        )
        
        self.db.add(chart)
        self.db.commit()
        self.db.refresh(chart)
        return chart
    
    def get_chart_by_id(self, chart_id: UUID) -> Optional[Chart]:
        """Get chart by ID"""
        return self.db.query(Chart).options(
            joinedload(Chart.chart_type),
            joinedload(Chart.query),
            joinedload(Chart.created_by_user)
        ).filter(Chart.id == chart_id).first()
    
    def get_charts(
        self,
        user_id: Optional[UUID] = None,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        chart_type: Optional[str] = None,
        is_public_only: bool = False
    ) -> Tuple[List[Chart], int]:
        """Get charts with filters"""
        query = self.db.query(Chart).options(
            joinedload(Chart.chart_type)
        )
        
        # Filter by access
        if is_public_only:
            query = query.filter(Chart.is_public == True)
        elif user_id:
            query = query.filter(
                or_(
                    Chart.created_by == user_id,
                    Chart.is_public == True
                )
            )
        
        # Apply filters
        if search:
            search_filter = f"%{search}%"
            query = query.filter(
                or_(
                    Chart.title.ilike(search_filter),
                    Chart.description.ilike(search_filter)
                )
            )
        
        if chart_type:
            query = query.join(ChartType).filter(
                ChartType.name == chart_type
            )
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        charts = query.order_by(
            desc(Chart.created_at)
        ).offset(skip).limit(limit).all()
        
        return charts, total
    
    def update_chart(
        self,
        chart_id: UUID,
        user_id: UUID,
        update_data: Dict[str, Any]
    ) -> Optional[Chart]:
        """Update chart configuration"""
        chart = self.get_chart_by_id(chart_id)
        if not chart or (chart.created_by != user_id and not chart.is_public):
            return None
        
        # Update allowed fields
        allowed_fields = ['title', 'description', 'config', 'filters', 'is_public']
        for key, value in update_data.items():
            if key in allowed_fields and hasattr(chart, key):
                setattr(chart, key, value)
        
        chart.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(chart)
        return chart
    
    def update_chart_data(
        self,
        chart_id: UUID,
        new_data_source: Dict[str, Any]
    ) -> Optional[Chart]:
        """Update chart data source"""
        chart = self.get_chart_by_id(chart_id)
        if not chart:
            return None
        
        chart.data_source = new_data_source
        chart.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(chart)
        return chart
    
    def delete_chart(self, chart_id: UUID, user_id: UUID) -> bool:
        """Delete chart"""
        chart = self.get_chart_by_id(chart_id)
        if not chart or chart.created_by != user_id:
            return False
        
        # Check if chart is used in dashboards
        dashboard_usage = self.db.query(DashboardChart).filter(
            DashboardChart.chart_id == chart_id
        ).count()
        
        if dashboard_usage > 0:
            raise ValueError("Chart is used in dashboards and cannot be deleted")
        
        self.db.delete(chart)
        self.db.commit()
        return True
    
    def duplicate_chart(
        self,
        chart_id: UUID,
        user_id: UUID,
        new_title: Optional[str] = None
    ) -> Optional[Chart]:
        """Create a copy of an existing chart"""
        original = self.get_chart_by_id(chart_id)
        if not original or (original.created_by != user_id and not original.is_public):
            return None
        
        # Create new chart with same configuration
        new_chart = Chart(
            title=new_title or f"{original.title} (Copy)",
            description=original.description,
            chart_type_id=original.chart_type_id,
            data_source=original.data_source.copy() if original.data_source else {},
            config=original.config.copy() if original.config else {},
            filters=original.filters.copy() if original.filters else {},
            is_public=False,  # Copies are private by default
            created_by=user_id
        )
        
        self.db.add(new_chart)
        self.db.commit()
        self.db.refresh(new_chart)
        return new_chart
    
    # =====================================================
    # Saved Chart Operations
    # =====================================================
    
    def save_chart_for_user(
        self,
        user_id: UUID,
        chart_id: UUID,
        name: Optional[str] = None,
        is_favorite: bool = False,
        tags: Optional[List[str]] = None
    ) -> SavedChart:
        """Save chart to user's collection"""
        # Check if already saved
        existing = self.db.query(SavedChart).filter(
            SavedChart.user_id == user_id,
            SavedChart.chart_id == chart_id
        ).first()
        
        if existing:
            # Update existing
            if name:
                existing.name = name
            existing.is_favorite = is_favorite
            if tags is not None:
                existing.tags = tags
            saved_chart = existing
        else:
            # Create new
            saved_chart = SavedChart(
                user_id=user_id,
                chart_id=chart_id,
                name=name,
                is_favorite=is_favorite,
                tags=tags or []
            )
            self.db.add(saved_chart)
        
        self.db.commit()
        self.db.refresh(saved_chart)
        return saved_chart
    
    def get_user_saved_charts(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 50,
        is_favorite: Optional[bool] = None,
        tags: Optional[List[str]] = None
    ) -> Tuple[List[SavedChart], int]:
        """Get user's saved charts"""
        query = self.db.query(SavedChart).options(
            joinedload(SavedChart.chart).joinedload(Chart.chart_type)
        ).filter(
            SavedChart.user_id == user_id
        )
        
        if is_favorite is not None:
            query = query.filter(SavedChart.is_favorite == is_favorite)
        
        if tags:
            query = query.filter(SavedChart.tags.overlap(tags))
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        saved_charts = query.order_by(
            desc(SavedChart.is_favorite),
            desc(SavedChart.saved_at)
        ).offset(skip).limit(limit).all()
        
        return saved_charts, total
    
    def unsave_chart(self, user_id: UUID, chart_id: UUID) -> bool:
        """Remove chart from user's saved collection"""
        saved_chart = self.db.query(SavedChart).filter(
            SavedChart.user_id == user_id,
            SavedChart.chart_id == chart_id
        ).first()
        
        if not saved_chart:
            return False
        
        self.db.delete(saved_chart)
        self.db.commit()
        return True
    
    # =====================================================
    # Chart Generation from Queries
    # =====================================================
    
    def create_chart_from_query(
        self,
        query_id: UUID,
        chart_type_name: str,
        title: str,
        user_id: UUID,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Chart]:
        """Create chart from query results"""
        # Get query
        query = self.db.query(NaturalLanguageQuery).filter(
            NaturalLanguageQuery.id == query_id
        ).first()
        
        if not query or query.status != 'completed':
            return None
        
        # Get chart type
        chart_type = self.get_chart_type_by_name(chart_type_name)
        if not chart_type:
            return None
        
        # Extract data from query results (simplified)
        # In production, this would parse the actual query results
        data_source = {
            "type": "query",
            "query_id": str(query_id),
            "sql": query.generated_sql,
            "data": []  # Would be populated from query results
        }
        
        return self.create_chart(
            title=title,
            chart_type_id=chart_type.id,
            data_source=data_source,
            created_by=user_id,
            query_id=query_id,
            config=config
        )
    
    # =====================================================
    # Chart Analytics
    # =====================================================
    
    def get_popular_charts(
        self,
        limit: int = 10,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get most saved/viewed charts"""
        # Get charts by save count
        popular = self.db.query(
            Chart,
            func.count(SavedChart.id).label('save_count')
        ).outerjoin(
            SavedChart
        ).group_by(
            Chart.id
        ).order_by(
            desc('save_count')
        ).limit(limit).all()
        
        return [
            {
                "chart": chart,
                "save_count": save_count
            }
            for chart, save_count in popular
        ]
    
    def get_chart_statistics(self, user_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Get chart usage statistics"""
        query = self.db.query(Chart)
        if user_id:
            query = query.filter(Chart.created_by == user_id)
        
        total_charts = query.count()
        
        # Charts by type
        charts_by_type = dict(
            self.db.query(
                ChartType.name,
                func.count(Chart.id)
            ).join(
                Chart
            ).group_by(ChartType.name).all()
        )
        
        # Public vs private
        public_charts = query.filter(Chart.is_public == True).count()
        
        # Recent charts
        recent_charts = query.filter(
            Chart.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
        ).count()
        
        return {
            "total_charts": total_charts,
            "charts_by_type": charts_by_type,
            "public_charts": public_charts,
            "private_charts": total_charts - public_charts,
            "charts_created_today": recent_charts
        }
