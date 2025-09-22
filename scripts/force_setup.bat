@echo off
setlocal
cd /d %~dp0\..
rem env Variablen fuer diesen Prozess leeren, damit Popup NICHT skippt
set TG_API_ID=
set TG_API_HASH=
set TG_TARGET=
set FORWARD_TO=
rem alte .env optional loeschen (auskommentieren, falls behalten gewuenscht)
del /f /q .env 2>nul
call scripts\start_windows.bat --setup
