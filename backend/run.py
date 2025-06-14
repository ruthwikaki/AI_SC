#!/usr/bin/env python3
"""
Simple script to run the application
"""
import subprocess
import sys

def main():
    """Run the application"""
    print("Starting Supply Chain AI Backend...")
    print("="*60)
    
    # Run uvicorn
    cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
