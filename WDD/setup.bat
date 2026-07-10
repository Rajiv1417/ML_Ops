@echo off
REM Welding Defect Detection - Windows Setup Script
REM Installs all dependencies including CPU-only PyTorch

echo.
echo ========================================
echo Welding Defect Detection - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [1/3] Python found. Installing base dependencies...
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install base dependencies
    pause
    exit /b 1
)

echo.
echo [2/3] Installing PyTorch CPU version...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo WARNING: PyTorch installation had issues, but may have partially succeeded
)

echo.
echo [3/3] Verifying installation...
python -c "import torch; import streamlit; print('[OK] All dependencies installed successfully!')" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Some imports failed, but installation may still work
    echo Trying individual imports...
    python -c "import streamlit" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Streamlit not installed properly
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo [OK] Setup Complete!
echo ========================================
echo.
echo To start the application, run:
echo   streamlit run app.py
echo.
pause
