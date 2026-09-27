@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Create the environment first: python -m venv venv
    echo Then install dependencies: venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
"venv\Scripts\python.exe" web_app.py
pause
