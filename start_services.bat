@echo off
echo Starting Supply Chain LLM Services...
echo.

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    pause
    exit /b 1
)

REM Install psutil if needed
echo Checking Python dependencies...
pip show psutil >nul 2>&1
if errorlevel 1 (
    echo Installing psutil...
    pip install psutil
)

REM Run the startup script
echo.
echo Starting all services...
python start_all_services.py

pause