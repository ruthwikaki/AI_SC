#!/usr/bin/env python
"""
Main entry point for the AI-Powered Supply Chain Backend
This runs the complete server with all features
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the FastAPI app from server.py
from app.api.server import api_app as app

# Run with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    
    print("\n" + "="*60)
    print("🚀 Starting AI-Powered Supply Chain Backend")
    print("="*60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"Docs: http://localhost:{port}/api/docs")
    print("="*60 + "\n")
    
    # Run the server
    uvicorn.run(
        "main:app",  # Reference to app in this file
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )