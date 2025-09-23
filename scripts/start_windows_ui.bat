@echo on
setlocal
cd /d %~dp0\..
if not exist .venv ( python -m venv .venv )
rem immer Wizard zeigen -> .env/Variablen werden im Code resettet
.\.venv\Scripts\python.exe -u -X faulthandler TelegramCopier_Windows.py --always-setup
pause
