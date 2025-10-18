@echo off
REM Build wrapper script for Qallow VM
REM Sets up Visual Studio environment and compiles

setlocal enabledelayedexpansion

set MODE=%1
if "%MODE%"=="" set MODE=CPU

REM Setup Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

set BUILD_DIR=build
set INCLUDE_DIR=core\include
set BACKEND_CPU=backend\cpu
set BACKEND_CUDA=backend\cuda
set INTERFACE_DIR=interface
set IO_DIR=io\adapters

if not exist %BUILD_DIR% mkdir %BUILD_DIR%

echo [BUILD] Compiling unified launcher and governance core...

if "%MODE%"=="CUDA" (
    echo [CUDA] Compiling CUDA-enabled version...

    REM Compile C files
    cl /c /O2 /DCUDA_ENABLED=1 "/I%INCLUDE_DIR%" ^
        "%INTERFACE_DIR%\launcher.c" ^
        "%INTERFACE_DIR%\main.c" ^
        "%BACKEND_CPU%\qallow_kernel.c" ^
        "%BACKEND_CPU%\overlay.c" ^
        "%BACKEND_CPU%\ethics.c" ^
        "%BACKEND_CPU%\ppai.c" ^
        "%BACKEND_CPU%\qcp.c" ^
        "%BACKEND_CPU%\pocket_dimension.c" ^
        "%BACKEND_CPU%\telemetry.c" ^
        "%BACKEND_CPU%\adaptive.c" ^
        "%BACKEND_CPU%\pocket.c" ^
        "%BACKEND_CPU%\govern.c" ^
        "%BACKEND_CPU%\ingest.c" ^
        "%BACKEND_CPU%\verify.c" ^
        "%BACKEND_CPU%\semantic_memory.c" ^
        "%BACKEND_CPU%\goal_synthesizer.c" ^
        "%BACKEND_CPU%\transfer_engine.c" ^
        "%BACKEND_CPU%\self_reflection.c" ^
        "%BACKEND_CPU%\phase7_core.c" ^
        "%IO_DIR%\net_adapter.c" ^
        "%IO_DIR%\sim_adapter.c"

    if errorlevel 1 exit /b 1

    REM Move object files to build directory
    if not exist %BUILD_DIR% mkdir %BUILD_DIR%
    move *.obj %BUILD_DIR%\ >nul 2>&1

    REM Compile CUDA files (includes ppai_kernels.cu and qcp_kernels.cu)
    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
    "%CUDA_PATH%\bin\nvcc.exe" -c -O2 -arch=sm_89 -I%INCLUDE_DIR% ^
        "%BACKEND_CUDA%\ppai_kernels.cu" ^
        "%BACKEND_CUDA%\qcp_kernels.cu"

    if errorlevel 1 exit /b 1

    REM Move CUDA object files to build directory
    move ppai.obj %BUILD_DIR%\ >nul 2>&1
    move qcp.obj %BUILD_DIR%\ >nul 2>&1

    REM Link CUDA executable
    "%CUDA_PATH%\bin\nvcc.exe" -O2 -arch=sm_89 ^
        %BUILD_DIR%\launcher.obj ^
        %BUILD_DIR%\main.obj ^
        %BUILD_DIR%\qallow_kernel.obj ^
        %BUILD_DIR%\overlay.obj ^
        %BUILD_DIR%\ethics.obj ^
        %BUILD_DIR%\govern.obj ^
        %BUILD_DIR%\ppai.obj ^
        %BUILD_DIR%\qcp.obj ^
        %BUILD_DIR%\pocket_dimension.obj ^
        %BUILD_DIR%\telemetry.obj ^
        %BUILD_DIR%\adaptive.obj ^
        %BUILD_DIR%\pocket.obj ^
        %BUILD_DIR%\ingest.obj ^
        %BUILD_DIR%\verify.obj ^
        %BUILD_DIR%\semantic_memory.obj ^
        %BUILD_DIR%\goal_synthesizer.obj ^
        %BUILD_DIR%\transfer_engine.obj ^
        %BUILD_DIR%\self_reflection.obj ^
        %BUILD_DIR%\phase7_core.obj ^
        %BUILD_DIR%\net_adapter.obj ^
        %BUILD_DIR%\sim_adapter.obj ^
        -L"%CUDA_PATH%\lib\x64" -lcudart -lcurand ^
        -o "%BUILD_DIR%\qallow_cuda.exe"

    if errorlevel 1 exit /b 1

    echo [SUCCESS] CUDA build completed: %BUILD_DIR%\qallow_cuda.exe
    
) else (
    echo [CPU] Compiling CPU-only version...

    cl /O2 "/I%INCLUDE_DIR%" "/Fe%BUILD_DIR%\qallow.exe" ^
        "%INTERFACE_DIR%\launcher.c" ^
        "%INTERFACE_DIR%\main.c" ^
        "%BACKEND_CPU%\qallow_kernel.c" ^
        "%BACKEND_CPU%\overlay.c" ^
        "%BACKEND_CPU%\ethics.c" ^
        "%BACKEND_CPU%\ppai.c" ^
        "%BACKEND_CPU%\qcp.c" ^
        "%BACKEND_CPU%\pocket_dimension.c" ^
        "%BACKEND_CPU%\telemetry.c" ^
        "%BACKEND_CPU%\adaptive.c" ^
        "%BACKEND_CPU%\pocket.c" ^
        "%BACKEND_CPU%\govern.c" ^
        "%BACKEND_CPU%\ingest.c" ^
        "%BACKEND_CPU%\verify.c" ^
        "%BACKEND_CPU%\semantic_memory.c" ^
        "%BACKEND_CPU%\goal_synthesizer.c" ^
        "%BACKEND_CPU%\transfer_engine.c" ^
        "%BACKEND_CPU%\self_reflection.c" ^
        "%BACKEND_CPU%\phase7_core.c" ^
        "%IO_DIR%\net_adapter.c" ^
        "%IO_DIR%\sim_adapter.c"

    if errorlevel 1 exit /b 1

    echo [SUCCESS] CPU build completed: %BUILD_DIR%\qallow.exe
)

REM Clean up object files
del /q "%BUILD_DIR%\*.obj" >nul 2>&1

echo [BUILD] Build process completed successfully
endlocal

