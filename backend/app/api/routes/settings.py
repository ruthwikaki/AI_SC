# backend/app/api/routes/settings.py
from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime
import json

from app.db.database import get_db
from app.api.routes.auth import get_current_active_user
from app.models.user import User, UserPreference
from app.models.system import SystemSetting
from app.models.extended_models import NotificationSetting
from app.schemas.auth import UserPreferencesUpdate, NotificationSettingsUpdate
from app.utils.logger import get_logger


# Initialize logger
logger = get_logger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

@router.get("/preferences")
async def get_user_preferences(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get current user preferences"""
    try:
        preferences = db.query(UserPreference).filter(
            UserPreference.user_id == current_user.id
        ).all()
        
        # Convert to dict
        pref_dict = {
            pref.key: json.loads(pref.value) if pref.value_type == 'json' else pref.value
            for pref in preferences
        }
        
        # Add default preferences if missing
        defaults = {
            "theme": "light",
            "language": "en",
            "timezone": "UTC",
            "date_format": "YYYY-MM-DD",
            "number_format": "1,234.56",
            "page_size": 20,
            "auto_refresh": False,
            "refresh_interval": 300,
            "compact_view": False,
            "show_tooltips": True
        }
        
        for key, value in defaults.items():
            if key not in pref_dict:
                pref_dict[key] = value
        
        return {
            "preferences": pref_dict,
            "last_updated": max([p.updated_at for p in preferences]) if preferences else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/preferences")
async def update_user_preferences(
    preferences: UserPreferencesUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update user preferences"""
    try:
        updated_count = 0
        
        for key, value in preferences.preferences.items():
            # Check if preference exists
            pref = db.query(UserPreference).filter(
                UserPreference.user_id == current_user.id,
                UserPreference.key == key
            ).first()
            
            # Determine value type
            value_type = 'json' if isinstance(value, (dict, list)) else 'string'
            value_str = json.dumps(value) if value_type == 'json' else str(value)
            
            if pref:
                pref.value = value_str
                pref.value_type = value_type
                pref.updated_at = datetime.utcnow()
            else:
                pref = UserPreference(
                    user_id=current_user.id,
                    key=key,
                    value=value_str,
                    value_type=value_type,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(pref)
            
            updated_count += 1
        
        db.commit()
        
        return {
            "message": f"Updated {updated_count} preferences",
            "updated_at": datetime.utcnow()
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/notifications")
async def get_notification_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get user notification settings"""
    try:
        settings = db.query(NotificationSetting).filter(
            NotificationSetting.user_id == current_user.id
        ).first()
        
        if not settings:
            # Create default settings
            settings = NotificationSetting(
                user_id=current_user.id,
                email_enabled=True,
                push_enabled=False,
                sms_enabled=False,
                notification_types=json.dumps({
                    "order_updates": True,
                    "inventory_alerts": True,
                    "supplier_issues": True,
                    "report_completion": True,
                    "system_updates": False,
                    "performance_alerts": True
                }),
                quiet_hours_start=None,
                quiet_hours_end=None,
                created_at=datetime.utcnow()
            )
            db.add(settings)
            db.commit()
            db.refresh(settings)
        
        return {
            "email_enabled": settings.email_enabled,
            "push_enabled": settings.push_enabled,
            "sms_enabled": settings.sms_enabled,
            "notification_types": json.loads(settings.notification_types),
            "quiet_hours": {
                "enabled": settings.quiet_hours_start is not None,
                "start": settings.quiet_hours_start,
                "end": settings.quiet_hours_end
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/notifications")
async def update_notification_settings(
    settings: NotificationSettingsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update notification settings"""
    try:
        notification_settings = db.query(NotificationSetting).filter(
            NotificationSetting.user_id == current_user.id
        ).first()
        
        if not notification_settings:
            notification_settings = NotificationSetting(user_id=current_user.id)
            db.add(notification_settings)
        
        # Update settings
        if settings.email_enabled is not None:
            notification_settings.email_enabled = settings.email_enabled
        if settings.push_enabled is not None:
            notification_settings.push_enabled = settings.push_enabled
        if settings.sms_enabled is not None:
            notification_settings.sms_enabled = settings.sms_enabled
        if settings.notification_types:
            notification_settings.notification_types = json.dumps(settings.notification_types)
        if settings.quiet_hours:
            notification_settings.quiet_hours_start = settings.quiet_hours.get('start')
            notification_settings.quiet_hours_end = settings.quiet_hours.get('end')
        
        notification_settings.updated_at = datetime.utcnow()
        db.commit()
        
        return {"message": "Notification settings updated successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system")
async def get_system_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get system-wide settings (admin only)"""
    try:
        if not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        settings = db.query(SystemSetting).all()
        
        return {
            setting.key: {
                "value": json.loads(setting.value) if setting.value_type == 'json' else setting.value,
                "description": setting.description,
                "category": setting.category,
                "is_public": setting.is_public
            }
            for setting in settings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/system/{key}")
async def update_system_setting(
    key: str,
    value: Any = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update system setting (admin only)"""
    try:
        if not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
        
        if not setting:
            raise HTTPException(status_code=404, detail="Setting not found")
        
        # Update value
        value_type = 'json' if isinstance(value, (dict, list)) else 'string'
        value_str = json.dumps(value) if value_type == 'json' else str(value)
        
        setting.value = value_str
        setting.value_type = value_type
        setting.updated_at = datetime.utcnow()
        setting.updated_by = current_user.id
        
        db.commit()
        
        # Log the change
        logger.info(f"System setting '{key}' updated by user {current_user.email}")
        
        return {
            "key": key,
            "value": value,
            "updated_at": setting.updated_at
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard-layout")
async def get_dashboard_layout(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get user's dashboard layout configuration"""
    try:
        layout_pref = db.query(UserPreference).filter(
            UserPreference.user_id == current_user.id,
            UserPreference.key == 'dashboard_layout'
        ).first()
        
        if layout_pref:
            return json.loads(layout_pref.value)
        else:
            # Return default layout
            return {
                "widgets": [
                    {"id": "inventory_overview", "position": {"x": 0, "y": 0, "w": 6, "h": 4}},
                    {"id": "order_status", "position": {"x": 6, "y": 0, "w": 6, "h": 4}},
                    {"id": "supplier_performance", "position": {"x": 0, "y": 4, "w": 12, "h": 4}}
                ],
                "theme": "default"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/dashboard-layout")
async def update_dashboard_layout(
    layout: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update user's dashboard layout"""
    try:
        layout_pref = db.query(UserPreference).filter(
            UserPreference.user_id == current_user.id,
            UserPreference.key == 'dashboard_layout'
        ).first()
        
        layout_str = json.dumps(layout)
        
        if layout_pref:
            layout_pref.value = layout_str
            layout_pref.updated_at = datetime.utcnow()
        else:
            layout_pref = UserPreference(
                user_id=current_user.id,
                key='dashboard_layout',
                value=layout_str,
                value_type='json',
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(layout_pref)
        
        db.commit()
        
        return {"message": "Dashboard layout updated successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))