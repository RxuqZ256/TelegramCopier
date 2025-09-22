@echo off
setlocal
cd /d %~dp0\..
if not exist .venv ( python -m venv .venv )
.\.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
.\.venv\Scripts\python.exe -c "from ui.app import run_app; run_app({})"
echo.
pause
