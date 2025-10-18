@echo off
REM Build wrapper script for Qallow VM
REM Sets up Visual Studio environment and compiles

setlocal enabledelayedexpansion

set MODE=%1
if "%MODE%"=="" set MODE=CPU

REM Setup Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

set BUILD_DIR=build
set QALLOW_VM_DIR=qallow_vm
set INCLUDE_DIR=%QALLOW_VM_DIR%\include
set SRC_DIR=%QALLOW_VM_DIR%\src
set KERNEL_DIR=%QALLOW_VM_DIR%\kernel
set OVERLAYS_DIR=%QALLOW_VM_DIR%\overlays
set MODULES_DIR=%QALLOW_VM_DIR%\modules
set SANDBOX_DIR=%QALLOW_VM_DIR%\sandbox

if not exist %BUILD_DIR% mkdir %BUILD_DIR%

if "%MODE%"=="CUDA" (
    echo [CUDA] Compiling CUDA-enabled version...

    REM Compile C files
    cl /c /O2 /DCUDA_ENABLED=1 "/I%INCLUDE_DIR%" ^
        "%SRC_DIR%\main.c" ^
        "%KERNEL_DIR%\qallow_kernel.c" ^
        "%OVERLAYS_DIR%\overlay.c" ^
        "%MODULES_DIR%\ethics.c" ^
        "%MODULES_DIR%\ppai.c" ^
        "%MODULES_DIR%\qcp.c" ^
        "%SANDBOX_DIR%\pocket_dimension.c"

    if errorlevel 1 exit /b 1

    REM Move object files to build directory
    if not exist %BUILD_DIR% mkdir %BUILD_DIR%
    move *.obj %BUILD_DIR%\ >nul 2>&1
    
    REM Compile CUDA files
    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
    "%CUDA_PATH%\bin\nvcc.exe" -c -O2 -arch=sm_89 -I%INCLUDE_DIR% ^
        "%MODULES_DIR%\ppai.cu" ^
        "%MODULES_DIR%\qcp.cu"

    if errorlevel 1 exit /b 1

    REM Move CUDA object files to build directory
    move ppai.obj %BUILD_DIR%\ >nul 2>&1
    move qcp.obj %BUILD_DIR%\ >nul 2>&1

    REM Link CUDA executable
    "%CUDA_PATH%\bin\nvcc.exe" -O2 -arch=sm_89 ^
        %BUILD_DIR%\main.obj ^
        %BUILD_DIR%\qallow_kernel.obj ^
        %BUILD_DIR%\overlay.obj ^
        %BUILD_DIR%\ethics.obj ^
        %BUILD_DIR%\ppai.obj ^
        %BUILD_DIR%\qcp.obj ^
        %BUILD_DIR%\pocket_dimension.obj ^
        -L"%CUDA_PATH%\lib\x64" -lcudart -lcurand ^
        -o "%BUILD_DIR%\qallow_cuda.exe"
    
    if errorlevel 1 exit /b 1
    
    echo [SUCCESS] CUDA build completed: %BUILD_DIR%\qallow_cuda.exe
    
) else (
    echo [CPU] Compiling CPU-only version...

    cl /O2 "/I%INCLUDE_DIR%" "/Fe%BUILD_DIR%\qallow.exe" ^
        "%SRC_DIR%\main.c" ^
        "%KERNEL_DIR%\qallow_kernel.c" ^
        "%OVERLAYS_DIR%\overlay.c" ^
        "%MODULES_DIR%\ethics.c" ^
        "%MODULES_DIR%\ppai.c" ^
        "%MODULES_DIR%\qcp.c" ^
        "%SANDBOX_DIR%\pocket_dimension.c"

    if errorlevel 1 exit /b 1

    echo [SUCCESS] CPU build completed: %BUILD_DIR%\qallow.exe
)

REM Clean up object files
del /q "%BUILD_DIR%\*.obj" >nul 2>&1

echo [BUILD] Build process completed successfully
endlocal

