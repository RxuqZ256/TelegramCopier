@echo off
setlocal
cd /d %~dp0\..
if not exist .venv (
  python -m venv .venv
)
rem immer die venv-Python verwenden
.\.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
.\.venv\Scripts\python.exe TelegramCopier_Windows.py %*
pause
