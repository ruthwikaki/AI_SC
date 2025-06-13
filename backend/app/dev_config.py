"""Development configuration overrides"""
import os
from typing import List

class Settings:
    # CORS - Allow everything in development
    cors_origins: List[str] = ["*"]
    
    # Disable rate limiting in development
    rate_limit_enabled: bool = False
    
    # Other settings
    environment: str = "development"
    debug: bool = True

# Apply to FastAPI app
def apply_dev_settings(app):
    from fastapi.middleware.cors import CORSMiddleware
    
    # Remove existing CORS middleware
    app.middleware_stack = None
    app.middleware("http")(app.add_middleware)
    
    # Add permissive CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
    
    print("✓ Development CORS settings applied - allowing all origins")
    return app
