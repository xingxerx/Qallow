#Requires -Version 5.0
<#
.SYNOPSIS
    Qallow Unified Command Interface
    
.DESCRIPTION
    Master command for building, running, and managing the Qallow VM system.
    
.EXAMPLE
    ./qallow build cpu
    ./qallow run cuda
    ./qallow bench all
    ./qallow telemetry stream
    ./qallow status
#>

param(
    [Parameter(Position=0)]
    [ValidateSet('build', 'run', 'bench', 'benchmark', 'telemetry', 'status', 'clean', 'help', '')]
    [string]$Command = '',
    
    [Parameter(Position=1)]
    [string]$Target = '',
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

# Color definitions
$ColorSuccess = 'Green'
$ColorError = 'Red'
$ColorWarning = 'Yellow'
$ColorInfo = 'Cyan'
$ColorHeader = 'Magenta'

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "╔════════════════════════════════════════╗" -ForegroundColor $ColorHeader
    Write-Host "║ $($Text.PadRight(38)) ║" -ForegroundColor $ColorHeader
    Write-Host "╚════════════════════════════════════════╝" -ForegroundColor $ColorHeader
    Write-Host ""
}

function Write-Success {
    param([string]$Text)
    Write-Host "✅ $Text" -ForegroundColor $ColorSuccess
}

function Write-ErrorCustom {
    param([string]$Text)
    Write-Host "❌ $Text" -ForegroundColor $ColorError
}

function Write-InfoMsg {
    param([string]$Text)
    Write-Host "ℹ️  $Text" -ForegroundColor $ColorInfo
}

function Show-Help {
    param([string]$Topic = '')

    if ($Topic -eq '' -or $Topic -eq 'help') {
        Write-Header "QALLOW UNIFIED COMMAND"
        Write-Host "Usage: ./qallow <command> [target] [options]" -ForegroundColor $ColorInfo
        Write-Host ""
        Write-Host "COMMANDS:" -ForegroundColor $ColorHeader
        Write-Host ""
        Write-Host "  build [cpu|cuda|all]      Build the Qallow system" -ForegroundColor White
        Write-Host "  run [cpu|cuda]            Run simulation" -ForegroundColor White
        Write-Host "  bench [cpu|cuda|all]      Run benchmarks (3 runs)" -ForegroundColor White
        Write-Host "  telemetry [stream|bench]  View telemetry data" -ForegroundColor White
        Write-Host "  status                    Show system status" -ForegroundColor White
        Write-Host "  clean                     Clean build artifacts" -ForegroundColor White
        Write-Host "  help [command]            Show this help" -ForegroundColor White
        Write-Host ""
        Write-Host "EXAMPLES:" -ForegroundColor $ColorHeader
        Write-Host ""
        Write-Host "  ./qallow build cpu        # Build CPU version" -ForegroundColor Gray
        Write-Host "  ./qallow build all        # Build both CPU and CUDA" -ForegroundColor Gray
        Write-Host "  ./qallow run cuda         # Run CUDA version" -ForegroundColor Gray
        Write-Host "  ./qallow bench all        # Benchmark both versions" -ForegroundColor Gray
        Write-Host "  ./qallow telemetry stream # View real-time data" -ForegroundColor Gray
        Write-Host "  ./qallow status           # Show system status" -ForegroundColor Gray
        Write-Host ""
    }
}

function Invoke-Build {
    param([string]$Mode = 'cpu')
    
    if ($Mode -eq 'all') {
        Write-Header "BUILDING CPU VERSION"
        & powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Mode CPU
        
        Write-Header "BUILDING CUDA VERSION"
        & powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Mode CUDA
    } else {
        Write-Header "BUILDING $($Mode.ToUpper()) VERSION"
        & powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Mode $Mode
    }
}

function Invoke-Run {
    param([string]$Mode = 'cpu')

    $exe = if ($Mode -eq 'cuda') { 'build/qallow_cuda.exe' } else { 'build/qallow.exe' }

    if (-not (Test-Path $exe)) {
        Write-ErrorCustom "Executable not found: $exe"
        Write-InfoMsg "Run './qallow build $Mode' first"
        return
    }

    Write-Header "RUNNING $($Mode.ToUpper()) VERSION"
    & $exe
}

function Invoke-Benchmark {
    param([string]$Mode = 'cpu')
    
    if ($Mode -eq 'all') {
        Write-Header "BENCHMARKING CPU VERSION"
        & powershell -ExecutionPolicy Bypass -File scripts/benchmark.ps1 -Exe .\build\qallow.exe -Runs 3
        
        Write-Header "BENCHMARKING CUDA VERSION"
        & powershell -ExecutionPolicy Bypass -File scripts/benchmark.ps1 -Exe .\build\qallow_cuda.exe -Runs 3
    } else {
        $exe = if ($Mode -eq 'cuda') { '.\build\qallow_cuda.exe' } else { '.\build\qallow.exe' }
        Write-Header "BENCHMARKING $($Mode.ToUpper()) VERSION"
        & powershell -ExecutionPolicy Bypass -File scripts/benchmark.ps1 -Exe $exe -Runs 3
    }
}

function Show-Telemetry {
    param([string]$Type = 'stream')

    Write-Header "TELEMETRY DATA"

    if ($Type -eq 'stream' -or $Type -eq 'all') {
        if (Test-Path 'qallow_stream.csv') {
            Write-Host "📊 Real-time Stream (qallow_stream.csv):" -ForegroundColor $ColorInfo
            Get-Content 'qallow_stream.csv' | Select-Object -Last 10
            Write-Host ""
        } else {
            Write-InfoMsg "No stream data yet. Run './qallow run' first."
        }
    }

    if ($Type -eq 'bench' -or $Type -eq 'all') {
        if (Test-Path 'qallow_bench.log') {
            Write-Host "📈 Benchmark Log (qallow_bench.log):" -ForegroundColor $ColorInfo
            Get-Content 'qallow_bench.log'
            Write-Host ""
        } else {
            Write-InfoMsg "No benchmark data yet. Run './qallow bench' first."
        }
    }

    if ($Type -eq 'adapt' -or $Type -eq 'all') {
        if (Test-Path 'adapt_state.json') {
            Write-Host "🧠 Adaptive State (adapt_state.json):" -ForegroundColor $ColorInfo
            Get-Content 'adapt_state.json' | ConvertFrom-Json | Format-List
            Write-Host ""
        } else {
            Write-InfoMsg "No adaptive state yet. Run './qallow run' first."
        }
    }
}

function Show-Status {
    Write-Header "QALLOW SYSTEM STATUS"

    Write-Host "Build Status:" -ForegroundColor $ColorHeader
    if (Test-Path 'build/qallow.exe') {
        $size = [math]::Round((Get-Item 'build/qallow.exe').Length / 1024, 1)
        Write-Host "✅ CPU build ready ($size KB)" -ForegroundColor $ColorSuccess
    } else {
        Write-Host "❌ CPU build not found" -ForegroundColor $ColorError
    }

    if (Test-Path 'build/qallow_cuda.exe') {
        $size = [math]::Round((Get-Item 'build/qallow_cuda.exe').Length / 1024, 1)
        Write-Host "✅ CUDA build ready ($size KB)" -ForegroundColor $ColorSuccess
    } else {
        Write-Host "❌ CUDA build not found" -ForegroundColor $ColorError
    }

    Write-Host ""
    Write-Host "Telemetry Files:" -ForegroundColor $ColorHeader
    if (Test-Path 'qallow_stream.csv') {
        $lines = (Get-Content 'qallow_stream.csv' | Measure-Object -Line).Lines
        Write-Host "✅ Stream data: $lines lines" -ForegroundColor $ColorSuccess
    } else {
        Write-Host "ℹ️  Stream data: Not generated" -ForegroundColor $ColorInfo
    }

    if (Test-Path 'qallow_bench.log') {
        $lines = (Get-Content 'qallow_bench.log' | Measure-Object -Line).Lines
        Write-Host "✅ Benchmark log: $lines lines" -ForegroundColor $ColorSuccess
    } else {
        Write-Host "ℹ️  Benchmark log: Not generated" -ForegroundColor $ColorInfo
    }

    if (Test-Path 'adapt_state.json') {
        Write-Host "✅ Adaptive state: Configured" -ForegroundColor $ColorSuccess
    } else {
        Write-Host "ℹ️  Adaptive state: Not configured" -ForegroundColor $ColorInfo
    }

    Write-Host ""
    Write-Host "System Info:" -ForegroundColor $ColorHeader
    Write-Host "ℹ️  CPU: $(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name)" -ForegroundColor $ColorInfo
    Write-Host "ℹ️  Cores: $(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty NumberOfCores)" -ForegroundColor $ColorInfo
    Write-Host "ℹ️  RAM: $([math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)) GB" -ForegroundColor $ColorInfo

    $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*NVIDIA*' }
    if ($gpu) {
        Write-Host "✅ GPU: $($gpu.Name)" -ForegroundColor $ColorSuccess
    } else {
        Write-Host "ℹ️  GPU: Not detected" -ForegroundColor $ColorInfo
    }

    Write-Host ""
}

function Invoke-Clean {
    Write-Header "CLEANING BUILD ARTIFACTS"

    if (Test-Path 'build') {
        Remove-Item 'build' -Recurse -Force
        Write-Host "✅ Removed build directory" -ForegroundColor $ColorSuccess
    }

    Write-Host "✅ Clean complete" -ForegroundColor $ColorSuccess
}

# Main command routing
switch ($Command.ToLower()) {
    'build' {
        $mode = if ($Target -eq '') { 'cpu' } else { $Target.ToLower() }
        Invoke-Build $mode
    }
    'run' {
        $mode = if ($Target -eq '') { 'cpu' } else { $Target.ToLower() }
        Invoke-Run $mode
    }
    'bench' {
        $mode = if ($Target -eq '') { 'cpu' } else { $Target.ToLower() }
        Invoke-Benchmark $mode
    }
    'benchmark' {
        $mode = if ($Target -eq '') { 'cpu' } else { $Target.ToLower() }
        Invoke-Benchmark $mode
    }
    'telemetry' {
        $type = if ($Target -eq '') { 'stream' } else { $Target.ToLower() }
        Show-Telemetry $type
    }
    'status' {
        Show-Status
    }
    'clean' {
        Invoke-Clean
    }
    'help' {
        Show-Help $Target
    }
    default {
        Show-Help
    }
}

