@echo off
setlocal
cd /d %~dp0\..
echo === DEBUG ENV ===
where python
if not exist .venv ( echo (creating venv) & python -m venv .venv )
.\.venv\Scripts\python.exe -V
.\.venv\Scripts\python.exe -c "import sys,platform;print('platform', platform.platform());import tkinter as tk;print('tk OK')"
.\.venv\Scripts\python.exe -c "import numpy,MetaTrader5 as mt5;print('numpy',numpy.__version__);print('mt5 import OK')"
echo === RUN APP ===
.\.venv\Scripts\python.exe TelegramCopier_Windows.py --setup --ui
echo.
pause
