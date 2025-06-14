from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import json
import uuid

from app.db.database import get_db
from app.api.routes.auth import get_current_active_user
from app.models.user import User
from app.models.extended_models import Report, ReportTemplate, ScheduledReport
from app.models.extended_models import ReportTemplate, ScheduledReport
from app.schemas.reports import (
    ReportRequest,
    ReportResponse,
    ReportTemplateRequest,
    ReportTemplateResponse,
    ScheduledReportRequest,
    ScheduledReportResponse,
    ReportListResponse,
    ReportTemplateListResponse,
    ScheduledReportListResponse
)
from app.utils.logger import get_logger
from app.api.middleware.client_context import get_client_context

# Initialize logger
logger = get_logger(__name__)

# Router
router = APIRouter(
    prefix="/reports",
    tags=["reports"],
    dependencies=[Depends(get_current_active_user)],
    responses={401: {"description": "Unauthorized"}}
)


@router.get("/templates", response_model=List[ReportTemplateResponse])
async def get_report_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get available report templates"""
    try:
        query = db.query(ReportTemplate).filter(ReportTemplate.is_active == True)
        
        if category:
            query = query.filter(ReportTemplate.category == category)
        
        templates = query.all()
        
        return [
            ReportTemplateResponse(
                id=template.id,
                name=template.name,
                description=template.description,
                category=template.category,
                parameters=json.loads(template.parameters) if template.parameters else {},
                preview_image=template.preview_image,
                estimated_time=template.estimated_generation_time
            )
            for template in templates
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    report_request: ReportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Generate a new report"""
    try:
        # Get template
        template = db.query(ReportTemplate).filter(
            ReportTemplate.id == report_request.template_id
        ).first()
        
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Create report record
        report = Report(
            name=report_request.name or f"{template.name} - {datetime.utcnow().strftime('%Y-%m-%d')}",
            template_id=template.id,
            user_id=current_user.id,
            parameters=json.dumps(report_request.parameters),
            status='generating',
            created_at=datetime.utcnow()
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        
        # Generate report in background
        background_tasks.add_task(
            _generate_report_content,
            report_id=report.id,
            template=template,
            parameters=report_request.parameters,
            db=db
        )
        
        return ReportResponse(
            id=report.id,
            name=report.name,
            status=report.status,
            template_name=template.name,
            created_at=report.created_at,
            parameters=report_request.parameters
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list", response_model=List[ReportResponse])
async def list_reports(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, le=100),
    offset: int = Query(0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List user's reports"""
    try:
        query = db.query(Report).filter(Report.user_id == current_user.id)
        
        if status:
            query = query.filter(Report.status == status)
        
        query = query.order_by(Report.created_at.desc())
        reports = query.limit(limit).offset(offset).all()
        
        return [
            ReportResponse(
                id=report.id,
                name=report.name,
                status=report.status,
                template_name=report.template.name if report.template else "Custom",
                created_at=report.created_at,
                completed_at=report.completed_at,
                file_path=report.file_path,
                error_message=report.error_message
            )
            for report in reports
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{report_id}/download")
async def download_report(
    report_id: int,
    format: str = Query("pdf", description="Export format"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Download generated report"""
    try:
        report = db.query(Report).filter(
            Report.id == report_id,
            Report.user_id == current_user.id
        ).first()
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        if report.status != 'completed':
            raise HTTPException(status_code=400, detail="Report not ready")
        
        # Get file path based on format
        export_manager = ExportManager()
        file_path = export_manager.get_report_file(report.id, format)
        
        if not os.path.exists(file_path):
            # Generate export in requested format
            file_path = export_manager.export_report(report, format)
        
        return FileResponse(
            path=file_path,
            filename=f"{report.name}.{format}",
            media_type=export_manager.get_mime_type(format)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/schedule", response_model=Dict[str, Any])
async def schedule_report(
    schedule_request: ScheduledReportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Schedule recurring report generation"""
    try:
        scheduled_report = ScheduledReport(
            name=schedule_request.name,
            template_id=schedule_request.template_id,
            user_id=current_user.id,
            schedule_type=schedule_request.schedule_type,
            schedule_config=json.dumps(schedule_request.schedule_config),
            parameters=json.dumps(schedule_request.parameters),
            recipients=json.dumps(schedule_request.recipients),
            is_active=True,
            created_at=datetime.utcnow(),
            next_run=_calculate_next_run(schedule_request.schedule_type, schedule_request.schedule_config)
        )
        
        db.add(scheduled_report)
        db.commit()
        db.refresh(scheduled_report)
        
        return {
            "id": scheduled_report.id,
            "name": scheduled_report.name,
            "schedule_type": scheduled_report.schedule_type,
            "next_run": scheduled_report.next_run,
            "status": "scheduled"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scheduled", response_model=List[Dict[str, Any]])
async def get_scheduled_reports(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get user's scheduled reports"""
    try:
        scheduled_reports = db.query(ScheduledReport).filter(
            ScheduledReport.user_id == current_user.id,
            ScheduledReport.is_active == True
        ).all()
        
        return [
            {
                "id": sr.id,
                "name": sr.name,
                "template_name": sr.template.name if sr.template else "Custom",
                "schedule_type": sr.schedule_type,
                "next_run": sr.next_run,
                "last_run": sr.last_run,
                "recipients": json.loads(sr.recipients) if sr.recipients else []
            }
            for sr in scheduled_reports
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _generate_report_content(report_id: int, template: ReportTemplate, parameters: Dict, db: Session):
    """Background task to generate report content"""
    try:
        report = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            return
        
        # Generate report based on template type
        export_manager = ExportManager()
        
        # Get data based on template requirements
        report_data = _fetch_report_data(template, parameters, db)
        
        # Generate report file
        file_path = export_manager.generate_report(
            template=template,
            data=report_data,
            parameters=parameters
        )
        
        # Update report record
        report.status = 'completed'
        report.file_path = file_path
        report.completed_at = datetime.utcnow()
        db.commit()
        
    except Exception as e:
        logger.error(f"Error generating report {report_id}: {str(e)}")
        report = db.query(Report).filter(Report.id == report_id).first()
        if report:
            report.status = 'failed'
            report.error_message = str(e)
            db.commit()

def _fetch_report_data(template: ReportTemplate, parameters: Dict, db: Session) -> Dict:
    """Fetch data required for report generation"""
    # Implementation depends on template type
    # This would query various tables based on template requirements
    return {
        "template": template.name,
        "generated_at": datetime.utcnow(),
        "data": {}  # Actual data fetching logic here
    }

def _calculate_next_run(schedule_type: str, config: Dict) -> datetime:
    """Calculate next run time based on schedule configuration"""
    now = datetime.utcnow()
    
    if schedule_type == 'daily':
        return now + timedelta(days=1)
    elif schedule_type == 'weekly':
        return now + timedelta(weeks=1)
    elif schedule_type == 'monthly':
        return now + timedelta(days=30)
    else:
        return now