@echo off
setlocal
cd /d %~dp0\..
if not exist .venv ( python -m venv .venv )
.\.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
rem Onboarding erzwingen, danach normale App starten
.\.venv\Scripts\python.exe TelegramCopier_Windows.py --setup
pause
