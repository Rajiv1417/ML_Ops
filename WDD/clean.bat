@echo off
REM Welding Defect Detection - Clean Script
REM Removes all generated files and caches, reverting to original state

echo.
echo ========================================
echo Welding Defect Detection - Clean
echo ========================================
echo.
echo This script will remove:
echo   - Generated synthetic dataset
echo   - Model checkpoints
echo   - Python cache files (__pycache__)
echo   - Streamlit cache files
echo.

REM Confirm before proceeding
set /p confirm="Are you sure you want to clean all generated files? (yes/no): "
if /i not "%confirm%"=="yes" (
    echo Cleanup cancelled.
    pause
    exit /b 0
)

echo.
echo Cleaning generated files...
echo.

REM Remove synthetic dataset
if exist "data\weld_defects" (
    echo [1/4] Removing generated dataset...
    rmdir /s /q "data\weld_defects"
    echo        Removed: data\weld_defects
) else (
    echo [1/4] Dataset not found (already clean)
)

REM Remove checkpoints
if exist "checkpoints" (
    echo [2/4] Removing model checkpoints...
    rmdir /s /q "checkpoints"
    echo        Removed: checkpoints
) else (
    echo [2/4] Checkpoints not found (already clean)
)

REM Remove Python cache files
echo [3/4] Removing Python cache files...
setlocal enabledelayedexpansion
set count=0
for /r . %%d in (__pycache__) do (
    if exist "%%d" (
        rmdir /s /q "%%d"
        set /a count+=1
    )
)
if %count% gtr 0 (
    echo        Removed %count% __pycache__ directories
) else (
    echo        No cache directories found
)
endlocal

REM Remove .pyc files
echo [4/4] Removing .pyc files...
setlocal enabledelayedexpansion
set count=0
for /r . %%f in (*.pyc) do (
    if exist "%%f" (
        del /q "%%f"
        set /a count+=1
    )
)
if %count% gtr 0 (
    echo        Removed %count% .pyc files
) else (
    echo        No .pyc files found
)
endlocal

echo.
echo ========================================
echo [OK] Cleanup Complete!
echo ========================================
echo.
echo The following remain (not cleaned):
echo   - Python packages (pip installations)
echo   - Configuration files
echo   - Source code
echo.
echo To clean Python packages as well, run:
echo   pip install -r requirements.txt --force-reinstall
echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall
echo.
pause
