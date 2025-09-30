@echo off
REM Activation script for face detection module virtual environment (Windows)

echo 🐍 Activating Face Detection Module Virtual Environment...

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found. Creating it now...
    python -m venv .venv
    echo ✅ Virtual environment created
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo ✅ Virtual environment activated
echo 📦 Installing/updating requirements...

REM Install requirements if they exist
if exist "requirements.txt" (
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo ✅ Requirements installed
) else (
    echo ⚠️  No requirements.txt found
)

echo.
echo 🎯 Face Detection Module Environment Ready!
echo To deactivate, run: deactivate
echo.