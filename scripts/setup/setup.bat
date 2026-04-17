@echo off
REM Qallow Project Setup Script for Windows
REM This script installs all Python dependencies

echo.
echo ========================================================================
echo                    QALLOW PROJECT SETUP (Windows)
echo             Quantum-Photonic Computing Platform
echo ========================================================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.10+ from https://www.python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%

REM Check pip
echo Checking pip installation...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip not found!
    echo Please install pip by running: python -m ensurepip --upgrade
    pause
    exit /b 1
)
echo [OK] pip is installed

REM Create virtual environment
if exist venv (
    echo Virtual environment already exists
    set /p RECREATE="Remove and recreate? (y/n): "
    if /i "%RECREATE%"=="y" (
        echo Removing old virtual environment...
        rmdir /s /q venv
    ) else (
        goto :install_packages
    )
)

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created

REM Activate virtual environment
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

REM Upgrade pip
echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip
)

:install_packages
REM Install core requirements
if exist requirements.txt (
    echo Installing core requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo WARNING: Some packages may not have installed correctly
    ) else (
        echo [OK] Core requirements installed
    )
) else (
    echo WARNING: requirements.txt not found
)

REM Ask for optional packages
setlocal enabledelayedexpansion
set /p DEV_TOOLS="Install development tools? (y/n): "
if /i "!DEV_TOOLS!"=="y" (
    if exist requirements-dev.txt (
        echo Installing development tools...
        pip install -r requirements-dev.txt
        echo [OK] Development tools installed
    )
)

set /p WEB_TOOLS="Install web framework? (y/n): "
if /i "!WEB_TOOLS!"=="y" (
    if exist requirements-web.txt (
        echo Installing web framework...
        pip install -r requirements-web.txt
        echo [OK] Web framework installed
    )
)

set /p GPU_TOOLS="Install GPU support (requires CUDA 12.0+)? (y/n): "
if /i "!GPU_TOOLS!"=="y" (
    if exist requirements-gpu.txt (
        echo Installing GPU support...
        pip install -r requirements-gpu.txt
        echo [OK] GPU support installed
    )
)

REM Create data directories
if not exist data mkdir data
if not exist data\logs mkdir data\logs
if not exist data\quantum_results mkdir data\quantum_results
if not exist data\telemetry mkdir data\telemetry
echo [OK] Directories created

REM Verify installation
echo.
echo Verifying installation...
python --version
pip --version

python -c "import numpy, scipy, pandas" 2>nul
if errorlevel 1 (
    echo WARNING: Some core packages not yet installed
) else (
    echo [OK] Core packages verified
)

echo.
echo ========================================================================
echo                      SETUP COMPLETED!
echo ========================================================================
echo.
echo Next steps:
echo   1. Virtual environment is activated
echo   2. Run project overview: python run_qallow.py
echo   3. Run tests: python test_quantum_complete.py
echo   4. Read documentation: type README.md
echo.
echo For more information, see SETUP_GUIDE.md
echo.
pause
