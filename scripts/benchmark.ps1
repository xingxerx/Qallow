# Qallow VM Benchmark Script
# Times execution and reports performance metrics
# Usage: ./scripts/benchmark.ps1 -Exe .\build\qallow.exe
#        ./scripts/benchmark.ps1 -Exe .\build\qallow_cuda.exe

param(
    [Parameter(Mandatory=$true)]
    [string]$Exe,
    [int]$Runs = 3
)

$ErrorActionPreference = "Stop"

# Validate executable
if (-not (Test-Path $Exe)) {
    Write-Error "Executable not found: $Exe"
    exit 1
}

$ExeName = Split-Path -Leaf $Exe
$ExecutionMode = if ($ExeName -like "*cuda*") { "CUDA" } else { "CPU" }

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "     QALLOW VM BENCHMARK SUITE" -ForegroundColor Cyan
Write-Host "     Execution Mode: $ExecutionMode" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[BENCHMARK] Executable: $Exe" -ForegroundColor Green
Write-Host "[BENCHMARK] Runs: $Runs" -ForegroundColor Green
Write-Host "[BENCHMARK] Verbose: $Verbose" -ForegroundColor Green
Write-Host ""

# Get system info
Write-Host "[SYSTEM] Gathering system information..." -ForegroundColor Cyan
$OS = Get-CimInstance -ClassName Win32_OperatingSystem
$CPU = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
$RAM = [math]::Round(($OS.TotalVisibleMemorySize / 1MB), 2)

Write-Host "  OS: $($OS.Caption)" -ForegroundColor Gray
Write-Host "  CPU: $($CPU.Name)" -ForegroundColor Gray
Write-Host "  Cores: $($CPU.NumberOfCores)" -ForegroundColor Gray
Write-Host "  RAM: $RAM GB" -ForegroundColor Gray

if ($ExecutionMode -eq "CUDA") {
    $GPU = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
    if ($GPU) {
        Write-Host "  GPU: $GPU" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "[BENCHMARK] Starting $Runs benchmark run(s)..." -ForegroundColor Yellow
Write-Host ""

$times = @()
$outputs = @()

for ($i = 1; $i -le $Runs; $i++) {
    Write-Host "[RUN $i/$Runs] Executing..." -ForegroundColor Cyan
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    try {
        $output = & $Exe 2>&1
        $stopwatch.Stop()
        
        $elapsed = $stopwatch.Elapsed.TotalSeconds
        $times += $elapsed
        $outputs += $output
        
        Write-Host "[RUN $i/$Runs] Completed in $([math]::Round($elapsed, 3)) seconds" -ForegroundColor Green
        
        if ($Verbose) {
            Write-Host ""
            Write-Host "--- Output ---" -ForegroundColor Gray
            $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
            Write-Host "--- End Output ---" -ForegroundColor Gray
            Write-Host ""
        }
    } catch {
        Write-Error "Execution failed: $_"
        exit 1
    }
}

# Calculate statistics
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "     BENCHMARK RESULTS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$min = ($times | Measure-Object -Minimum).Minimum
$max = ($times | Measure-Object -Maximum).Maximum
$avg = ($times | Measure-Object -Average).Average
$stddev = if ($times.Count -gt 1) {
    [math]::Sqrt(($times | ForEach-Object { [math]::Pow($_ - $avg, 2) } | Measure-Object -Sum).Sum / ($times.Count - 1))
} else {
    0
}

Write-Host "[RESULTS] Execution Times:" -ForegroundColor Green
for ($i = 0; $i -lt $times.Count; $i++) {
    Write-Host "  Run $($i+1): $([math]::Round($times[$i], 3)) seconds" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[STATISTICS]" -ForegroundColor Green
Write-Host "  Minimum:  $([math]::Round($min, 3)) seconds" -ForegroundColor Cyan
Write-Host "  Maximum:  $([math]::Round($max, 3)) seconds" -ForegroundColor Cyan
Write-Host "  Average:  $([math]::Round($avg, 3)) seconds" -ForegroundColor Cyan
Write-Host "  Std Dev:  $([math]::Round($stddev, 3)) seconds" -ForegroundColor Cyan

# Extract metrics from output if available
Write-Host ""
Write-Host "[METRICS] Extracting execution metrics..." -ForegroundColor Cyan

$lastOutput = $outputs[-1]
$tickCount = 0

foreach ($line in $lastOutput) {
    if ($line -match "tick (\d+)") {
        $tickCount = [int]$matches[1]
    }
}

if ($tickCount -gt 0) {
    $throughput = [math]::Round($tickCount / $avg, 2)
    Write-Host "  Total Ticks: $tickCount" -ForegroundColor Gray
    Write-Host "  Throughput: $throughput ticks/second" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[BENCHMARK] Benchmark completed successfully" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "     SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Mode:     $ExecutionMode" -ForegroundColor Yellow
Write-Host "Avg Time: $([math]::Round($avg, 3)) seconds" -ForegroundColor Yellow
Write-Host "Runs:     $($times.Count)" -ForegroundColor Yellow
Write-Host ""

