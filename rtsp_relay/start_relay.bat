@echo off
title RTSP Relay Microservice
color 0A
echo ============================================================
echo    Smart AI Monitoring — RTSP Relay Microservice
echo ============================================================
echo.

:: Change to this script's directory
cd /d "%~dp0"

:: Use project venv
set VENV_PYTHON=..\\.venv_new\\Scripts\\python.exe

:: Install deps if needed
echo [*] Checking dependencies...
%VENV_PYTHON% -m pip install flask pyngrok opencv-python numpy --quiet

echo.
echo [*] Starting relay server with ngrok tunnel...
echo     Dashboard → http://localhost:6001
echo     Streams   → http://localhost:6001/streams
echo.

%VENV_PYTHON% relay_server.py --ngrok --port 6001

pause
