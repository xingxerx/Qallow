@echo off
REM Unified build script for Qallow VM
REM Supports both CPU-only and CUDA-enabled builds

setlocal enabledelayedexpansion

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=cuda

echo ============================================
echo Qallow VM Unified Build System
echo ============================================

REM Check for CUDA availability
nvcc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] CUDA not found, falling back to CPU build
    set BUILD_TYPE=cpu
)

REM Setup Visual Studio environment (using the working path from main build.bat)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Clean previous builds
if exist qallow_vm.exe del qallow_vm.exe
if exist qallow_vm_cpu.exe del qallow_vm_cpu.exe
if exist *.obj del *.obj

echo [BUILD] Building Qallow VM (%BUILD_TYPE% mode)

if "%BUILD_TYPE%"=="cuda" (
    echo [CUDA] Compiling CUDA-enabled version...
    
    REM Compile C files with CUDA support
    cl /c /O2 /DCUDA_ENABLED=1 ^
        /I"qallow_vm\include" ^
        /I"%CUDA_PATH%\include" ^
        qallow_vm\src\main.c ^
        qallow_vm\kernel\qallow_kernel.c ^
        qallow_vm\overlays\overlay.c ^
        qallow_vm\modules\ethics.c ^
        qallow_vm\sandbox\pocket_dimension.c
    
    REM Compile CUDA files
    nvcc -c -O2 -arch=sm_89 ^
        -I"qallow_vm\include" ^
        qallow_vm\modules\ppai.cu ^
        qallow_vm\modules\qcp.cu
    
    REM Link everything together
    nvcc -O2 -arch=sm_89 ^
        *.obj ^
        -L"%CUDA_PATH%\lib\x64" ^
        -lcudart -lcurand ^
        -o qallow_vm.exe
        
    if exist qallow_vm.exe (
        echo [SUCCESS] CUDA build completed: qallow_vm.exe
    ) else (
        echo [ERROR] CUDA build failed
        exit /b 1
    )
) else (
    echo [CPU] Compiling CPU-only version...
    
    REM Compile CPU-only version
    cl /O2 /Fe:qallow_vm_cpu.exe ^
        /I"qallow_vm\include" ^
        qallow_vm\src\main.c ^
        qallow_vm\kernel\qallow_kernel.c ^
        qallow_vm\overlays\overlay.c ^
        qallow_vm\modules\ppai.c ^
        qallow_vm\modules\qcp.c ^
        qallow_vm\modules\ethics.c ^
        qallow_vm\sandbox\pocket_dimension.c
    
    if exist qallow_vm_cpu.exe (
        echo [SUCCESS] CPU build completed: qallow_vm_cpu.exe
    ) else (
        echo [ERROR] CPU build failed
        exit /b 1
    )
)

REM Clean intermediate files
del *.obj >nul 2>&1

echo [BUILD] Build process completed
echo.
echo Usage:
if "%BUILD_TYPE%"=="cuda" (
    echo   .\qallow_vm.exe           - Run CUDA-accelerated version
) else (
    echo   .\qallow_vm_cpu.exe       - Run CPU-only version
)
echo   .\build_unified.bat clean   - Clean build artifacts
echo   .\build_unified.bat cpu     - Force CPU-only build
echo   .\build_unified.bat cuda    - Force CUDA build

if "%1"=="clean" (
    echo [CLEAN] Removing build artifacts...
    del qallow_vm.exe >nul 2>&1
    del qallow_vm_cpu.exe >nul 2>&1
    del *.obj >nul 2>&1
    echo [CLEAN] Clean completed
)

endlocal