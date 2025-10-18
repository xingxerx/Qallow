@echo off
REM Qallow Unified Command Wrapper (CMD version)
REM Alternative to qallow.bat for compatibility

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Call the PowerShell script with all arguments
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%qallow_unified.ps1" %*

REM Exit with the same code as PowerShell
exit /b %ERRORLEVEL%

