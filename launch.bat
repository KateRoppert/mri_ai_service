@echo off
REM MRI AI Service Docker Launcher for Windows
REM This script launches the Docker container with automatic path mapping

echo MRI AI Service Docker Launcher
echo ==============================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running or not installed.
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.7+ and try again.
    pause
    exit /b 1
)

REM Install required Python packages if needed
pip show pyyaml >nul 2>&1
if errorlevel 1 (
    echo Installing required Python packages...
    pip install pyyaml
)

REM Run the launcher
python launch_docker.py %*

pause