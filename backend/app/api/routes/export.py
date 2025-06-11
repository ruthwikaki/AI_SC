# backend/app/api/routes/export.py
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
import json
import io
import os

from app.db.database import get_db
from app.api.routes.auth import get_current_active_user
from app.models.user import User
from app.models.analytics import ExportJob
from app.visualization.export_manager import ExportManager
from app.schemas.visualization import ExportRequest, ExportJobResponse
from app.utils.logger import get_logger


# Initialize logger
logger = get_logger(__name__)

router = APIRouter(prefix="/api/export", tags=["export"])

@router.post("/create", response_model=ExportJobResponse)
async def create_export(
    export_request: ExportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new export job"""
    try:
        # Create export job record
        export_job = ExportJob(
            user_id=current_user.id,
            export_type=export_request.export_type,
            format=export_request.format,
            parameters=json.dumps(export_request.parameters),
            status='pending',
            created_at=datetime.utcnow()
        )
        db.add(export_job)
        db.commit()
        db.refresh(export_job)
        
        # Process export in background
        background_tasks.add_task(
            _process_export,
            export_job_id=export_job.id,
            export_request=export_request,
            db=db
        )
        
        return ExportJobResponse(
            id=export_job.id,
            status=export_job.status,
            export_type=export_job.export_type,
            format=export_job.format,
            created_at=export_job.created_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs", response_model=List[ExportJobResponse])
async def get_export_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, le=100),
    offset: int = Query(0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get user's export jobs"""
    try:
        query = db.query(ExportJob).filter(ExportJob.user_id == current_user.id)
        
        if status:
            query = query.filter(ExportJob.status == status)
        
        query = query.order_by(ExportJob.created_at.desc())
        jobs = query.limit(limit).offset(offset).all()
        
        return [
            ExportJobResponse(
                id=job.id,
                status=job.status,
                export_type=job.export_type,
                format=job.format,
                created_at=job.created_at,
                completed_at=job.completed_at,
                file_size=job.file_size,
                error_message=job.error_message
            )
            for job in jobs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{export_id}")
async def download_export(
    export_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Download exported file"""
    try:
        export_job = db.query(ExportJob).filter(
            ExportJob.id == export_id,
            ExportJob.user_id == current_user.id
        ).first()
        
        if not export_job:
            raise HTTPException(status_code=404, detail="Export not found")
        
        if export_job.status != 'completed':
            raise HTTPException(status_code=400, detail="Export not ready")
        
        if not os.path.exists(export_job.file_path):
            raise HTTPException(status_code=404, detail="Export file not found")
        
        return FileResponse(
            path=export_job.file_path,
            filename=f"export_{export_job.id}.{export_job.format}",
            media_type=_get_mime_type(export_job.format)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quick-export")
async def quick_export(
    export_data: Dict[str, Any],
    format: str = Query("csv", description="Export format"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Quick export without creating a job (for small datasets)"""
    try:
        export_manager = ExportManager()
        
        # Validate format
        if format not in ['csv', 'xlsx', 'json', 'pdf']:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
        # Get data based on export type
        data = _fetch_export_data(export_data, db, current_user)
        
        # Generate export
        if format == 'csv':
            output = export_manager.export_to_csv(data)
            return StreamingResponse(
                io.StringIO(output),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"}
            )
        elif format == 'json':
            output = export_manager.export_to_json(data)
            return StreamingResponse(
                io.StringIO(output),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"}
            )
        elif format == 'xlsx':
            output = export_manager.export_to_excel(data)
            return StreamingResponse(
                io.BytesIO(output),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename=export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"}
            )
        else:
            raise HTTPException(status_code=400, detail="Format not supported for quick export")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def get_export_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get available export templates"""
    try:
        templates = [
            {
                "id": "inventory_report",
                "name": "Inventory Report",
                "description": "Complete inventory status with stock levels and values",
                "category": "inventory",
                "formats": ["csv", "xlsx", "pdf"],
                "parameters": ["include_inactive", "warehouse_filter", "category_filter"]
            },
            {
                "id": "order_history",
                "name": "Order History",
                "description": "Detailed order history with customer information",
                "category": "orders",
                "formats": ["csv", "xlsx", "pdf"],
                "parameters": ["date_range", "status_filter", "customer_filter"]
            },
            {
                "id": "supplier_performance",
                "name": "Supplier Performance Report",
                "description": "Comprehensive supplier metrics and ratings",
                "category": "suppliers",
                "formats": ["xlsx", "pdf"],
                "parameters": ["date_range", "supplier_filter", "metric_selection"]
            },
            {
                "id": "financial_summary",
                "name": "Financial Summary",
                "description": "Revenue, costs, and profit analysis",
                "category": "financial",
                "formats": ["xlsx", "pdf"],
                "parameters": ["period", "department_filter", "include_projections"]
            },
            {
                "id": "custom_query",
                "name": "Custom Query Export",
                "description": "Export results from any custom query",
                "category": "custom",
                "formats": ["csv", "xlsx", "json"],
                "parameters": ["query_id", "include_metadata"]
            }
        ]
        
        if category:
            templates = [t for t in templates if t['category'] == category]
        
        return {"templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/jobs/{export_id}")
async def delete_export_job(
    export_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete an export job and its file"""
    try:
        export_job = db.query(ExportJob).filter(
            ExportJob.id == export_id,
            ExportJob.user_id == current_user.id
        ).first()
        
        if not export_job:
            raise HTTPException(status_code=404, detail="Export not found")
        
        # Delete file if exists
        if export_job.file_path and os.path.exists(export_job.file_path):
            os.remove(export_job.file_path)
        
        # Delete database record
        db.delete(export_job)
        db.commit()
        
        return {"message": "Export deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def _process_export(export_job_id: int, export_request: ExportRequest, db: Session):
    """Background task to process export"""
    try:
        export_job = db.query(ExportJob).filter(ExportJob.id == export_job_id).first()
        if not export_job:
            return
        
        # Update status
        export_job.status = 'processing'
        db.commit()
        
        # Get export manager
        export_manager = ExportManager()
        
        # Fetch data based on export type
        data = _fetch_export_data(export_request.parameters, db, None)
        
        # Generate export file
        file_path = export_manager.generate_export_file(
            data=data,
            format=export_request.format,
            export_type=export_request.export_type,
            parameters=export_request.parameters
        )
        
        # Update job record
        export_job.status = 'completed'
        export_job.file_path = file_path
        export_job.file_size = os.path.getsize(file_path)
        export_job.completed_at = datetime.utcnow()
        db.commit()
        
    except Exception as e:
        logger.error(f"Error processing export {export_job_id}: {str(e)}")
        export_job = db.query(ExportJob).filter(ExportJob.id == export_job_id).first()
        if export_job:
            export_job.status = 'failed'
            export_job.error_message = str(e)
            db.commit()

def _fetch_export_data(parameters: Dict, db: Session, user: Optional[User]) -> Dict:
    """Fetch data based on export parameters"""
    # This would contain logic to fetch data from various tables
    # based on the export type and parameters
    return {
        "data": [],
        "metadata": {
            "exported_at": datetime.utcnow(),
            "record_count": 0
        }
    }

def _get_mime_type(format: str) -> str:
    """Get MIME type for export format"""
    mime_types = {
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "json": "application/json",
        "pdf": "application/pdf",
        "xml": "application/xml"
    }
    return mime_types.get(format, "application/octet-stream")