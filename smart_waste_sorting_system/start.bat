@echo off
echo Smart Waste Sorting System - Windows Startup
echo ===========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "app.py" (
    echo Error: app.py not found
    echo Please run this script from the smart_waste_sorting_system directory
    pause
    exit /b 1
)

REM Install dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo Installing Python dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Warning: Some dependencies may not have installed correctly
        echo Continuing anyway...
    )
    echo.
)

REM Start the application
echo Starting Smart Waste Sorting System...
echo.
python start.py

pause
