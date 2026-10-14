@echo off
title Offline Face Finder
cd /d "%~dp0"

echo Launching Offline Face Finder...

:: 1. If compiled executable exists, run it directly (works on ANY PC without Python!)
if exist "%~dp0OfflineFaceFinder.exe" (
    start "" "%~dp0OfflineFaceFinder.exe"
    exit /b 0
)

:: 2. If portable folder executable exists, run it
if exist "%~dp0OfflineFaceFinder_portable\OfflineFaceFinder_portable.exe" (
    start "" "%~dp0OfflineFaceFinder_portable\OfflineFaceFinder_portable.exe"
    exit /b 0
)

:: 3. If local virtual environment exists and is valid on this PC
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0main.py"
    exit /b 0
)

:: 4. Fallback to system Python
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start "" pythonw "%~dp0main.py"
    exit /b 0
)

echo.
echo ============================================================
echo [Error] Could not launch application.
echo Please double-click 'OfflineFaceFinder.exe' directly.
echo ============================================================
pause
exit /b 1
