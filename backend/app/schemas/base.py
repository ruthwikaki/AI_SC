"""
Base schemas for Pydantic models
"""
from pydantic import BaseModel as PydanticBaseModel, ConfigDict
from typing import Any, Optional
from datetime import datetime
from uuid import UUID


class BaseModel(PydanticBaseModel):
    """Base Pydantic model with common configuration"""
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    )


class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None


class BaseRequest(BaseModel):
    """Base request model"""
    pass