from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any
# REMOVED CIRCULAR IMPORT: from app.api.routes.auth import get_current_user
from app.models.user import User
from ..deps import get_db, get_current_user, get_current_active_user, get_admin_user

router = APIRouter(prefix="/api/export", tags=["export"])

@router.post("/create")
async def create_export(export_data: Dict[str, Any], current_user: User = Depends(get_current_user)):
    return {"export_id": "exp-123", "status": "processing"}

@router.get("/jobs")
async def get_export_jobs(current_user: User = Depends(get_current_user)):
    return {"jobs": [], "total": 0}

@router.post("/quick-export")
async def quick_export(export_data: Dict[str, Any], format: str = Query("csv"), current_user: User = Depends(get_current_user)):
    return {"message": "Export completed", "format": format}

