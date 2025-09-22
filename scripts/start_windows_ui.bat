@echo on
setlocal
cd /d %~dp0\..
if not exist .venv ( python -m venv .venv )
.\.venv\Scripts\python.exe -u -X faulthandler TelegramCopier_Windows.py --setup
echo ExitCode=%errorlevel%
pause
