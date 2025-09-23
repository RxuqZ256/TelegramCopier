@echo on
setlocal
cd /d %~dp0\..
if not exist .venv ( python -m venv .venv )
.\.venv\Scripts\python.exe -u TelegramCopier_Windows.py
pause
