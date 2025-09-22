@echo off
setlocal
cd /d %~dp0\..
if not exist .venv (
  python -m venv .venv
)
rem immer innerhalb der venv installieren
.\.venv\Scripts\python.exe -m pip install --upgrade pip
rem harte Korrektur: falls NumPy 2.x vorhanden, neu installieren
.\.venv\Scripts\python.exe -m pip uninstall -y numpy 2>nul
.\.venv\Scripts\python.exe -m pip install --no-cache-dir numpy==1.26.4
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.windows.txt
echo.
echo ✅ Abhaengigkeiten installiert (NumPy 1.26.4 / MetaTrader5 5.0.45).
echo Starte App...
call scripts\start_windows.bat %*
