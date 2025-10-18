#Requires -Version 5.0
<#
.SYNOPSIS
    Qallow Unified Command Interface (Phase V)

.DESCRIPTION
    Master command for building, running, and managing the Qallow VM system.
    Routes to the unified qallow binary.

.EXAMPLE
    ./qallow build
    ./qallow run
    ./qallow bench
    ./qallow govern
    ./qallow help
#>

# Get command from arguments
$Command = if ($args.Count -gt 0) { $args[0] } else { 'help' }
$RemainingArgs = if ($args.Count -gt 1) { $args[1..($args.Count-1)] } else { @() }

# Color definitions
$ColorSuccess = 'Green'
$ColorError = 'Red'
$ColorWarning = 'Yellow'
$ColorInfo = 'Cyan'
$ColorHeader = 'Magenta'

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "[$Text]" -ForegroundColor $ColorHeader
    Write-Host ""
}

function Write-Success {
    param([string]$Text)
    Write-Host "[OK] $Text" -ForegroundColor $ColorSuccess
}

function Write-ErrorCustom {
    param([string]$Text)
    Write-Host "[ERROR] $Text" -ForegroundColor $ColorError
}

function Write-InfoMsg {
    param([string]$Text)
    Write-Host "[INFO] $Text" -ForegroundColor $ColorInfo
}

function Show-Help {
    Write-Header "QALLOW UNIFIED COMMAND (Phase V)"
    Write-Host "Usage: ./qallow [command]" -ForegroundColor $ColorInfo
    Write-Host ""
    Write-Host "COMMANDS:" -ForegroundColor $ColorHeader
    Write-Host ""
    Write-Host "  build      Detect toolchain and compile CPU + CUDA backends" -ForegroundColor White
    Write-Host "  run        Execute current binary (auto-selects CPU/CUDA)" -ForegroundColor White
    Write-Host "  bench      Run HITL benchmark with logging" -ForegroundColor White
    Write-Host "  visual     Open live dashboard" -ForegroundColor White
    Write-Host "  govern     Start autonomous governance and ethics audit loop" -ForegroundColor White
    Write-Host "  help       Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor $ColorHeader
    Write-Host ""
    Write-Host "  ./qallow build      # Build both CPU and CUDA versions" -ForegroundColor Gray
    Write-Host "  ./qallow run        # Run the VM" -ForegroundColor Gray
    Write-Host "  ./qallow bench      # Run benchmark" -ForegroundColor Gray
    Write-Host "  ./qallow govern     # Run governance audit" -ForegroundColor Gray
    Write-Host ""
}

function Invoke-Build {
    Write-Header "BUILDING QALLOW"
    if (-not (Test-Path "build\qallow.exe") -and -not (Test-Path "build\qallow_cuda.exe")) {
        Write-ErrorCustom "Qallow executable not found. Cannot build."
        exit 1
    }
    $exe = if (Test-Path "build\qallow_cuda.exe") { "build\qallow_cuda.exe" } else { "build\qallow.exe" }
    & $exe build @RemainingArgs
}

function Invoke-Run {
    Write-Header "RUNNING QALLOW"
    if (-not (Test-Path "build\qallow.exe") -and -not (Test-Path "build\qallow_cuda.exe")) {
        Write-ErrorCustom "Qallow executable not found. Run './qallow build' first."
        exit 1
    }
    $exe = if (Test-Path "build\qallow_cuda.exe") { "build\qallow_cuda.exe" } else { "build\qallow.exe" }
    & $exe run @RemainingArgs
}

function Invoke-Benchmark {
    Write-Header "RUNNING BENCHMARK"
    if (-not (Test-Path "build\qallow.exe") -and -not (Test-Path "build\qallow_cuda.exe")) {
        Write-ErrorCustom "Qallow executable not found. Run './qallow build' first."
        exit 1
    }
    $exe = if (Test-Path "build\qallow_cuda.exe") { "build\qallow_cuda.exe" } else { "build\qallow.exe" }
    & $exe bench @RemainingArgs
}

function Show-Telemetry {
    Write-Header "TELEMETRY DATA"
    Write-InfoMsg "Stream data: qallow_stream.csv"
    Write-InfoMsg "Benchmark log: qallow_bench.log"
    Write-InfoMsg "Adaptive state: adapt_state.json"
}

function Show-Status {
    Write-Header "QALLOW SYSTEM STATUS"
    Write-Host "Build Status:" -ForegroundColor $ColorHeader
    if (Test-Path 'build\qallow.exe') {
        Write-Success "CPU build ready"
    } else {
        Write-ErrorCustom "CPU build not found"
    }
    if (Test-Path 'build\qallow_cuda.exe') {
        Write-Success "CUDA build ready"
    } else {
        Write-InfoMsg "CUDA build not found"
    }
}

function Invoke-Govern {
    Write-Header "RUNNING GOVERNANCE AUDIT"
    if (-not (Test-Path "build\qallow.exe") -and -not (Test-Path "build\qallow_cuda.exe")) {
        Write-ErrorCustom "Qallow executable not found. Run './qallow build' first."
        exit 1
    }
    $exe = if (Test-Path "build\qallow_cuda.exe") { "build\qallow_cuda.exe" } else { "build\qallow.exe" }
    & $exe govern @RemainingArgs
}

function Invoke-Visual {
    Write-Header "OPENING DASHBOARD"
    if (-not (Test-Path "build\qallow.exe") -and -not (Test-Path "build\qallow_cuda.exe")) {
        Write-ErrorCustom "Qallow executable not found. Run './qallow build' first."
        exit 1
    }
    $exe = if (Test-Path "build\qallow_cuda.exe") { "build\qallow_cuda.exe" } else { "build\qallow.exe" }
    & $exe visual @RemainingArgs
}

# Main command routing
switch ($Command.ToLower()) {
    'build' {
        Invoke-Build
    }
    'run' {
        Invoke-Run
    }
    'bench' {
        Invoke-Benchmark
    }
    'benchmark' {
        Invoke-Benchmark
    }
    'govern' {
        Invoke-Govern
    }
    'visual' {
        Invoke-Visual
    }
    'dashboard' {
        Invoke-Visual
    }
    'status' {
        Show-Status
    }
    'help' {
        Show-Help
    }
    default {
        Show-Help
    }
}

