@echo off
setlocal
cd /d %~dp0\..
if not exist .venv ( python -m venv .venv )
.\.venv\Scripts\python.exe -c "import tkinter as tk; print('OK: tkinter available')"
pause
