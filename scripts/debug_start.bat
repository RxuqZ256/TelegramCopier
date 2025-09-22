@echo on
setlocal
cd /d %~dp0\..
if not exist .venv ( python -m venv .venv )
.\.venv\Scripts\python.exe -V
.\.venv\Scripts\python.exe -c "import tkinter as tk; print('tkinter OK')"
.\.venv\Scripts\python.exe -c "import numpy; print('numpy', numpy.__version__)"
.\.venv\Scripts\python.exe -u -X faulthandler TelegramCopier_Windows.py --setup
if exist logs\last_startup_error.log (
  echo --- logs\last_startup_error.log ---
  type logs\last_startup_error.log
)
pause
