@echo off
setlocal
cd /d %~dp0\..
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
rem harte Korrektur falls NumPy 2.x vorhanden ist
pip uninstall -y numpy >nul 2>&1
pip install --no-cache-dir numpy==1.26.4
pip install --no-cache-dir -r requirements.windows.txt
echo.
echo ✅ Abhaengigkeiten installiert (NumPy 1.26.4 / MetaTrader5 5.0.45).
echo Starte App...
python TelegramCopier_Windows.py
pause
