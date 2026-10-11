@echo off
title Airport Hub Detection - Linear Algebra Project
echo ======================================================================
echo    Starting Airport Hub Detection Linear Algebra Mini-Project
echo ======================================================================
echo.

REM Try python in PATH, then user-level Python installations
set PYTHON_CMD=python
where python >nul 2>nul
if %errorlevel% neq 0 (
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
        set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    ) else (
        set PYTHON_CMD=py
    )
)

"%PYTHON_CMD%" "%~dp0main.py"

echo.
echo Launching Interactive Visual Dashboard in default browser...
start "" "%~dp0figures\airport_hub_interactive_dashboard.html"

echo.
echo ======================================================================
echo Project executed successfully! Press any key to exit.
echo ======================================================================
pause
