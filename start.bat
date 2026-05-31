@echo off
:: NitroDQN — one-click launcher (Windows)
:: Run this from the nitrogen_app folder

echo ============================================
echo   NitroDQN - DQN + LLM Nitrogen Manager
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python not found. Install Python 3.10+ and try again.
  pause & exit /b 1
)

:: Install deps if needed
echo Installing dependencies...
pip install fastapi uvicorn httpx torch numpy --quiet
if errorlevel 1 (
  echo WARNING: Some packages may not have installed. Continuing...
)

echo.
echo Starting server at http://localhost:8000
echo Press Ctrl+C to stop.
echo.

:: Open browser after 2 second delay
start /b cmd /c "timeout /t 2 >nul && start http://localhost:8000"

:: Start FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
