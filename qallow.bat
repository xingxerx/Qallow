@echo off
REM Qallow Unified Command Wrapper (Phase V)
REM Routes commands to the unified qallow binary

setlocal enabledelayedexpansion

set MODE=%1
if "%MODE%"=="" set MODE=help

REM Check if qallow executable exists
if not exist "build\qallow.exe" (
    if not exist "build\qallow_cuda.exe" (
        echo [ERROR] Qallow executable not found. Run 'qallow build' first.
        exit /b 1
    )
)

REM Route to appropriate executable
if "%MODE%"=="build" (
    echo [QALLOW] Building Qallow...
    call build\qallow.exe build %2 %3 %4 %5
    exit /b !errorlevel!
)

if "%MODE%"=="run" (
    echo [QALLOW] Running Qallow...
    call build\qallow.exe run %2 %3 %4 %5
    exit /b !errorlevel!
)

if "%MODE%"=="bench" (
    echo [QALLOW] Running benchmark...
    call build\qallow.exe bench %2 %3 %4 %5
    exit /b !errorlevel!
)

if "%MODE%"=="benchmark" (
    echo [QALLOW] Running benchmark...
    call build\qallow.exe bench %2 %3 %4 %5
    exit /b !errorlevel!
)

if "%MODE%"=="visual" (
    echo [QALLOW] Opening dashboard...
    call build\qallow.exe visual %2 %3 %4 %5
    exit /b !errorlevel!
)

if "%MODE%"=="dashboard" (
    echo [QALLOW] Opening dashboard...
    call build\qallow.exe visual %2 %3 %4 %5
    exit /b !errorlevel!
)

if "%MODE%"=="govern" (
    echo [QALLOW] Running governance audit...
    call build\qallow.exe govern %2 %3 %4 %5
    exit /b !errorlevel!
)

if "%MODE%"=="verify" (
    echo [QALLOW] Running system verification...
    call build\qallow.exe verify %2 %3 %4 %5
    exit /b !errorlevel!
)

if "%MODE%"=="live" (
    echo [QALLOW] Starting Phase 6 live interface...
    call build\qallow.exe live %2 %3 %4 %5
    exit /b !errorlevel!
)

if "%MODE%"=="help" (
    call build\qallow.exe help
    exit /b 0
)

echo [ERROR] Unknown mode: %MODE%
call build\qallow.exe help
exit /b 1

