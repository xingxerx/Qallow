@echo off
REM Build script for Qallow native C + CUDA project
REM This script handles the CUDA compilation on Windows with proper environment setup

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
set SRC_DIR=src
set EMU_DIR=emulation
set INC_DIR=include

REM Clean if requested
if "%1"=="clean" (
    echo Cleaning build artifacts...
    del /q "%SRC_DIR%\*.obj" 2>nul
    del /q "%EMU_DIR%\*.obj" 2>nul
    del /q "qallow.exe" 2>nul
    echo Clean complete.
)

REM Compile C files with MSVC
echo Compiling C files...
cl /O2 /W4 "/I%INC_DIR%" /c "%SRC_DIR%\qallow_kernel.c" "/Fo%SRC_DIR%\qallow_kernel.obj"
if errorlevel 1 exit /b 1

cl /O2 /W4 "/I%INC_DIR%" /c "%SRC_DIR%\overlays.c" "/Fo%SRC_DIR%\overlays.obj"
if errorlevel 1 exit /b 1

REM Compile CUDA files
echo Compiling CUDA files...
"%NVCC%" -O2 -arch=sm_89 "-I%INC_DIR%" -c "%EMU_DIR%\photonic.cu" -o "%EMU_DIR%\photonic.obj"
if errorlevel 1 exit /b 1

"%NVCC%" -O2 -arch=sm_89 "-I%INC_DIR%" -c "%EMU_DIR%\quantum.cu" -o "%EMU_DIR%\quantum.obj"
if errorlevel 1 exit /b 1

REM Link
echo Linking...
"%NVCC%" -o qallow.exe "%SRC_DIR%\qallow_kernel.obj" "%SRC_DIR%\overlays.obj" "%EMU_DIR%\photonic.obj" "%EMU_DIR%\quantum.obj" -lcurand
if errorlevel 1 exit /b 1

echo Build successful! Executable: qallow.exe

