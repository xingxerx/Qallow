@echo off
REM Build Phase IV Demo with Multi-Pocket and Chronometric modules

setlocal enabledelayedexpansion

REM Set up Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM CUDA paths
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set NVCC=%CUDA_PATH%\bin\nvcc.exe

REM Directories
set CPU_DIR=backend\cpu
set CUDA_DIR=backend\cuda
set INC_DIR=core\include

if "%1"=="clean" (
    del /q "*.obj" "qallow_phase4.exe" 2>nul
    echo Clean complete.
    exit /b 0
)

echo ================================
echo Building Phase IV Demo
echo ================================

REM Compile demo main
echo Compiling phase4_demo.c...
cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "phase4_demo.c" /Fophase4_demo.obj
if errorlevel 1 exit /b 1

REM Compile CPU modules
echo Compiling CPU modules...
cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\qallow_kernel.c" "/Fo%CPU_DIR%\qallow_kernel.obj"
if errorlevel 1 exit /b 1

cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\overlay.c" "/Fo%CPU_DIR%\overlay.obj"
if errorlevel 1 exit /b 1

cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\ethics.c" "/Fo%CPU_DIR%\ethics.obj"
if errorlevel 1 exit /b 1

cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\multi_pocket.c" "/Fo%CPU_DIR%\multi_pocket.obj"
if errorlevel 1 exit /b 1

cl /O2 /W4 "/I%INC_DIR%" /DCUDA_ENABLED=1 /c "%CPU_DIR%\chronometric.c" "/Fo%CPU_DIR%\chronometric.obj"
if errorlevel 1 exit /b 1

REM Link
echo Linking qallow_phase4.exe...
"%NVCC%" -o qallow_phase4.exe ^
    phase4_demo.obj ^
    "%CPU_DIR%\qallow_kernel.obj" ^
    "%CPU_DIR%\overlay.obj" ^
    "%CPU_DIR%\ethics.obj" ^
    "%CPU_DIR%\multi_pocket.obj" ^
    "%CPU_DIR%\chronometric.obj" ^
    -lcurand

if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

echo.
echo ================================
echo BUILD SUCCESSFUL
echo ================================
echo.
echo Executable: qallow_phase4.exe
echo.
echo Usage:
echo   qallow_phase4.exe [num_pockets] [num_ticks]
echo   qallow_phase4.exe 8 100    (8 pockets, 100 ticks each)
echo ================================
