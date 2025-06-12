#!/usr/bin/env python3
"""
Main entry point for the Supply Chain AI Backend
"""

import uvicorn
from app.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "app.api.server:api_app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        workers=settings.uvicorn_workers if settings.environment == "production" else 1,
        log_level=settings.log_level.lower()
    )
