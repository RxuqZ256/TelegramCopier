@echo off
setlocal enabledelayedexpansion

REM ---------------------------------------------------------------
REM  Build-Skript für TelegramCopier (Windows)
REM  Erstellt eine virtuelle Umgebung und packt die App mit PyInstaller
REM ---------------------------------------------------------------

cd /d "%~dp0"

set VENV_DIR=.venv
if not exist "%VENV_DIR%" (
    echo [INFO] Erstelle virtuelle Umgebung in %VENV_DIR% ...
    py -3 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [FEHLER] Virtuelle Umgebung konnte nicht erstellt werden.
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [FEHLER] Virtuelle Umgebung konnte nicht aktiviert werden.
    exit /b 1
)

echo [INFO] Aktualisiere pip ...
python -m pip install --upgrade pip >nul
if errorlevel 1 (
    echo [FEHLER] pip konnte nicht aktualisiert werden.
    exit /b 1
)

echo [INFO] Installiere Projekt-Abhaengigkeiten ...
pip install -r requirements.windows.txt >nul
if errorlevel 1 (
    echo [FEHLER] Anforderungen konnten nicht installiert werden.
    exit /b 1
)

echo [INFO] Installiere PyInstaller ...
pip install "pyinstaller>=6.3" >nul
if errorlevel 1 (
    echo [FEHLER] PyInstaller konnte nicht installiert werden.
    exit /b 1
)

echo [INFO] Baue TelegramCopier.exe ...
pyinstaller TelegramCopier_Windows.py ^
    --name TelegramCopier ^
    --noconsole ^
    --onedir ^
    --clean ^
    --collect-all telethon ^
    --collect-all MetaTrader5

if errorlevel 1 (
    echo [FEHLER] PyInstaller hat den Build mit einem Fehler beendet.
    exit /b 1
)

echo.
echo [FERTIG] Die Anwendung befindet sich im Ordner dist\TelegramCopier.
echo           Du kannst die EXE-Datei zusammen mit den generierten

echo           Konfigurationsdateien auf einen Zielrechner kopieren.

echo.
endlocal
