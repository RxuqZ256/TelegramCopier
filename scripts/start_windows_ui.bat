@echo off
setlocal
cd /d %~dp0\..
if not exist .venv ( python -m venv .venv )
.\.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
rem UI direkt starten, egal was die Umgebung hat
.\.venv\Scripts\python.exe TelegramCopier_Windows.py --ui
echo.
pause
