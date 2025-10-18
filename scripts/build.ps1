# Qallow VM Build Script
# Supports CPU-only and CUDA-accelerated builds
# Usage: ./scripts/build.ps1 -Mode CPU
#        ./scripts/build.ps1 -Mode CUDA

param(
    [ValidateSet("CPU", "CUDA")]
    [string]$Mode = "CPU",
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"

# Configuration
$BUILD_DIR = "build"

# Create build directory
if (-not (Test-Path $BUILD_DIR)) {
    New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null
    Write-Host "[BUILD] Created build directory" -ForegroundColor Green
}

# Clean if requested
if ($Clean) {
    Write-Host "[CLEAN] Removing build artifacts..." -ForegroundColor Yellow
    Remove-Item -Path "$BUILD_DIR\*.obj" -ErrorAction SilentlyContinue
    Remove-Item -Path "$BUILD_DIR\*.exe" -ErrorAction SilentlyContinue
    Write-Host "[CLEAN] Clean completed" -ForegroundColor Green
    exit 0
}

# Set up Visual Studio environment
Write-Host "[ENV] Setting up Visual Studio environment..." -ForegroundColor Cyan
$VS_PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$VC_VARS = "$VS_PATH\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $VC_VARS)) {
    Write-Error "Visual Studio Build Tools not found at $VS_PATH"
    exit 1
}

Write-Host "[BUILD] Building Qallow VM in $Mode mode" -ForegroundColor Cyan
Write-Host ""

# Use batch wrapper for compilation
Write-Host "[COMPILE] Invoking build wrapper..." -ForegroundColor Cyan

$wrapper_script = Join-Path (Split-Path $PSCommandPath) "build_wrapper.bat"
if (-not (Test-Path $wrapper_script)) {
    Write-Error "Build wrapper not found: $wrapper_script"
    exit 1
}

# Run the batch wrapper
& cmd.exe /c "$wrapper_script $Mode"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed with exit code $LASTEXITCODE"
    exit 1
}

# Verify output
if ($Mode -eq "CUDA") {
    $exe_path = "$BUILD_DIR\qallow_cuda.exe"
} else {
    $exe_path = "$BUILD_DIR\qallow.exe"
}

if (Test-Path $exe_path) {
    $size = (Get-Item $exe_path).Length / 1KB
    Write-Host "[SUCCESS] Build completed: $exe_path ($([math]::Round($size, 2)) KB)" -ForegroundColor Green
} else {
    Write-Error "Executable not created: $exe_path"
    exit 1
}

Write-Host ""
Write-Host "[BUILD] Build process completed successfully" -ForegroundColor Green

