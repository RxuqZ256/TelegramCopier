@echo on
setlocal
cd /d %~dp0\..
del /f /q tg_session.session 2>nul
del /f /q tg_session.session-journal 2>nul
echo Telegram session deleted.
pause
