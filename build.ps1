# Build script for Qallow native C + CUDA project
# This script handles the CUDA compilation on Windows

param(
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"

# CUDA paths
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$NVCC = "$CUDA_PATH\bin\nvcc.exe"

# Visual Studio paths
$VS_PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$VC_VARS = "$VS_PATH\VC\Auxiliary\Build\vcvars64.bat"

# Check if NVCC exists
if (-not (Test-Path $NVCC)) {
    Write-Error "NVCC not found at $NVCC. Please install CUDA Toolkit."
    exit 1
}

# Check if Visual Studio exists
if (-not (Test-Path $VC_VARS)) {
    Write-Error "Visual Studio Build Tools not found. Please install Visual Studio Build Tools."
    exit 1
}

# Set up Visual Studio environment
Write-Host "Setting up Visual Studio environment..."
$env:Path = "$VS_PATH\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;$env:Path"

# Directories
$SRC_DIR = "src"
$EMU_DIR = "emulation"
$INC_DIR = "include"

# Clean if requested
if ($Clean) {
    Write-Host "Cleaning build artifacts..."
    Remove-Item -Path "$SRC_DIR\*.o" -ErrorAction SilentlyContinue
    Remove-Item -Path "$EMU_DIR\*.o" -ErrorAction SilentlyContinue
    Remove-Item -Path "qallow.exe" -ErrorAction SilentlyContinue
    Write-Host "Clean complete."
}

# Compile C files with MSVC
Write-Host "Compiling C files..."
$CL = "cl.exe"
& $CL /O2 /W4 "/I$INC_DIR" /c "$SRC_DIR\qallow_kernel.c" "/Fo$SRC_DIR\qallow_kernel.o"
if ($LASTEXITCODE -ne 0) { exit 1 }

& $CL /O2 /W4 "/I$INC_DIR" /c "$SRC_DIR\overlays.c" "/Fo$SRC_DIR\overlays.o"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Compiling CUDA files..."
# Compile CUDA files
& $NVCC -O2 -arch=sm_89 "-I$INC_DIR" -c "$EMU_DIR\photonic.cu" -o "$EMU_DIR\photonic.o"
if ($LASTEXITCODE -ne 0) { exit 1 }

& $NVCC -O2 -arch=sm_89 "-I$INC_DIR" -c "$EMU_DIR\quantum.cu" -o "$EMU_DIR\quantum.o"
if ($LASTEXITCODE -ne 0) { exit 1 }

# Link
Write-Host "Linking..."
& $NVCC -o qallow.exe "$SRC_DIR\qallow_kernel.o" "$SRC_DIR\overlays.o" "$EMU_DIR\photonic.o" "$EMU_DIR\quantum.o" -lcurand
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Build successful! Executable: qallow.exe"

