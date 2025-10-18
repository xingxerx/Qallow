#Requires -Version 5.0
<#
.SYNOPSIS
    Qallow Unified Command Launcher (Phase V)
    
.DESCRIPTION
    Master command interface for the unified Qallow binary.
    Routes commands to the appropriate executable.
    
.EXAMPLE
    ./qallow_launcher.ps1 build
    ./qallow_launcher.ps1 run
    ./qallow_launcher.ps1 bench
    ./qallow_launcher.ps1 govern
    ./qallow_launcher.ps1 visual
#>

param(
    [Parameter(Position=0)]
    [ValidateSet('build', 'run', 'bench', 'benchmark', 'visual', 'dashboard', 'govern', 'help', '')]
    [string]$Command = 'help',
    
    [Parameter(Position=1, ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

# Color output helpers
function Write-Header {
    param([string]$Message)
    Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║ $($Message.PadRight(38)) ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[✓] $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[✗] $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "[*] $Message" -ForegroundColor Yellow
}

# Check if executable exists
function Test-QallowExecutable {
    $cuda_exe = "build\qallow_cuda.exe"
    $cpu_exe = "build\qallow.exe"
    
    if ((Test-Path $cuda_exe) -or (Test-Path $cpu_exe)) {
        return $true
    }
    return $false
}

# Get the appropriate executable
function Get-QallowExecutable {
    $cuda_exe = "build\qallow_cuda.exe"
    $cpu_exe = "build\qallow.exe"
    
    if (Test-Path $cuda_exe) {
        return $cuda_exe
    }
    return $cpu_exe
}

# Main command routing
switch ($Command.ToLower()) {
    'build' {
        Write-Header "BUILDING QALLOW"
        $exe = Get-QallowExecutable
        if (-not (Test-Path $exe)) {
            Write-Error-Custom "Qallow executable not found. Cannot build."
            exit 1
        }
        & $exe build @Args
        exit $LASTEXITCODE
    }
    
    'run' {
        Write-Header "RUNNING QALLOW"
        if (-not (Test-QallowExecutable)) {
            Write-Error-Custom "Qallow executable not found. Run 'qallow build' first."
            exit 1
        }
        $exe = Get-QallowExecutable
        & $exe run @Args
        exit $LASTEXITCODE
    }
    
    'bench' {
        Write-Header "RUNNING BENCHMARK"
        if (-not (Test-QallowExecutable)) {
            Write-Error-Custom "Qallow executable not found. Run 'qallow build' first."
            exit 1
        }
        $exe = Get-QallowExecutable
        & $exe bench @Args
        exit $LASTEXITCODE
    }
    
    'benchmark' {
        Write-Header "RUNNING BENCHMARK"
        if (-not (Test-QallowExecutable)) {
            Write-Error-Custom "Qallow executable not found. Run 'qallow build' first."
            exit 1
        }
        $exe = Get-QallowExecutable
        & $exe bench @Args
        exit $LASTEXITCODE
    }
    
    'visual' {
        Write-Header "OPENING DASHBOARD"
        if (-not (Test-QallowExecutable)) {
            Write-Error-Custom "Qallow executable not found. Run 'qallow build' first."
            exit 1
        }
        $exe = Get-QallowExecutable
        & $exe visual @Args
        exit $LASTEXITCODE
    }
    
    'dashboard' {
        Write-Header "OPENING DASHBOARD"
        if (-not (Test-QallowExecutable)) {
            Write-Error-Custom "Qallow executable not found. Run 'qallow build' first."
            exit 1
        }
        $exe = Get-QallowExecutable
        & $exe visual @Args
        exit $LASTEXITCODE
    }
    
    'govern' {
        Write-Header "AUTONOMOUS GOVERNANCE"
        if (-not (Test-QallowExecutable)) {
            Write-Error-Custom "Qallow executable not found. Run 'qallow build' first."
            exit 1
        }
        $exe = Get-QallowExecutable
        & $exe govern @Args
        exit $LASTEXITCODE
    }
    
    'help' {
        Write-Header "QALLOW UNIFIED COMMAND SYSTEM"
        Write-Host ""
        Write-Host "Usage: qallow [command]" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor Cyan
        Write-Host "  build      Detect toolchain and compile CPU + CUDA backends"
        Write-Host "  run        Execute current binary (auto-selects CPU/CUDA)"
        Write-Host "  bench      Run HITL benchmark with logging"
        Write-Host "  visual     Open live dashboard"
        Write-Host "  govern     Start autonomous governance and ethics audit loop"
        Write-Host "  help       Show this help message"
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Cyan
        Write-Host "  qallow build      # Build both CPU and CUDA versions"
        Write-Host "  qallow run        # Run the VM"
        Write-Host "  qallow bench      # Run benchmark"
        Write-Host "  qallow govern     # Run governance audit"
        Write-Host ""
        exit 0
    }
    
    default {
        Write-Error-Custom "Unknown command: $Command"
        Write-Host ""
        & $PSCommandPath help
        exit 1
    }
}

