@echo off
REM Build script for Qallow Phase IV with Multi-Pocket and Chronometric modules
REM Handles CUDA compilation with Visual Studio 2022 BuildTools

setlocal enabledelayedexpansion

REM Set up Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM CUDA paths
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set NVCC=%CUDA_PATH%\bin\nvcc.exe

REM Check if NVCC exists
if not exist "%NVCC%" (
    echo Error: NVCC not found at %NVCC%
    exit /b 1
)

REM Directories
set CPU_DIR=backend\cpu
set CUDA_DIR=backend\cuda
set INC_DIR=core\include

REM Clean if requested
if "%1"=="clean" (
    echo Cleaning build artifacts...
    del /q "%CPU_DIR%\*.obj" 2>nul
    del /q "%CUDA_DIR%\*.obj" 2>nul
    del /q "qallow.exe" 2>nul
    del /q "*.csv" 2>nul
    del /q "*_summary.txt" 2>nul
    echo Clean complete.
    exit /b 0
)

echo ================================
echo Building Qallow Phase IV
echo ================================

REM Compile CPU implementation files
echo.
echo [1/3] Compiling CPU modules...
echo --------------------------------

cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\qallow_kernel.c" "/Fo%CPU_DIR%\qallow_kernel.obj"
if errorlevel 1 exit /b 1

cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\overlay.c" "/Fo%CPU_DIR%\overlay.obj"
if errorlevel 1 exit /b 1

cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\ppai.c" "/Fo%CPU_DIR%\ppai.obj"
if errorlevel 1 exit /b 1

cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\qcp.c" "/Fo%CPU_DIR%\qcp.obj"
if errorlevel 1 exit /b 1

cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\ethics.c" "/Fo%CPU_DIR%\ethics.obj"
if errorlevel 1 exit /b 1

cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\pocket_dimension.c" "/Fo%CPU_DIR%\pocket_dimension.obj"
if errorlevel 1 exit /b 1

REM Phase IV modules
echo.
echo [1/3] Compiling Phase IV CPU modules...
cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\multi_pocket.c" "/Fo%CPU_DIR%\multi_pocket.obj"
if errorlevel 1 exit /b 1

cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\chronometric.c" "/Fo%CPU_DIR%\chronometric.obj"
if errorlevel 1 exit /b 1

REM Compile CUDA kernel files
echo.
echo [2/3] Compiling CUDA kernels (sm_89 for RTX 5080)...
echo --------------------------------

"%NVCC%" -O2 -arch=sm_89 "-I%INC_DIR%" -DCUDA_ENABLED=1 -c "%CUDA_DIR%\ppai.cu" -o "%CUDA_DIR%\ppai.obj"
if errorlevel 1 exit /b 1

"%NVCC%" -O2 -arch=sm_89 "-I%INC_DIR%" -DCUDA_ENABLED=1 -c "%CUDA_DIR%\qcp.cu" -o "%CUDA_DIR%\qcp.obj"
if errorlevel 1 exit /b 1

"%NVCC%" -O2 -arch=sm_89 "-I%INC_DIR%" -DCUDA_ENABLED=1 -c "%CUDA_DIR%\photonic.cu" -o "%CUDA_DIR%\photonic.obj"
if errorlevel 1 exit /b 1

"%NVCC%" -O2 -arch=sm_89 "-I%INC_DIR%" -DCUDA_ENABLED=1 -c "%CUDA_DIR%\quantum.cu" -o "%CUDA_DIR%\quantum.obj"
if errorlevel 1 exit /b 1

REM Link everything
echo.
echo [3/3] Linking qallow.exe...
echo --------------------------------

"%NVCC%" -o qallow.exe ^
    "%CPU_DIR%\qallow_kernel.obj" ^
    "%CPU_DIR%\overlay.obj" ^
    "%CPU_DIR%\ppai.obj" ^
    "%CPU_DIR%\qcp.obj" ^
    "%CPU_DIR%\ethics.obj" ^
    "%CPU_DIR%\pocket_dimension.obj" ^
    "%CPU_DIR%\multi_pocket.obj" ^
    "%CPU_DIR%\chronometric.obj" ^
    "%CUDA_DIR%\ppai.obj" ^
    "%CUDA_DIR%\qcp.obj" ^
    "%CUDA_DIR%\photonic.obj" ^
    "%CUDA_DIR%\quantum.obj" ^
    -lcurand

if errorlevel 1 (
    echo.
    echo ================================
    echo BUILD FAILED
    echo ================================
    exit /b 1
)

echo.
echo ================================
echo BUILD SUCCESSFUL
echo ================================
echo Executable: qallow.exe
echo.
echo Phase IV Features:
echo   - Multi-Pocket Scheduler (16 parallel worldlines)
echo   - Chronometric Prediction Layer
echo   - Temporal Time Bank
echo   - Drift Detection
echo.
echo Run with: qallow.exe
echo ================================
