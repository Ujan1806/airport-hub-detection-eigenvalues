@echo off
title Airport Hub Detection - Linear Algebra Project
echo ======================================================================
echo    Starting Airport Hub Detection Linear Algebra Mini-Project
echo ======================================================================
echo.

set PYTHON_PATH="C:\Users\SANJOY DUTTA\AppData\Local\Programs\Python\Python312\python.exe"

if exist %PYTHON_PATH% (
    %PYTHON_PATH% "%~dp0main.py"
) else (
    python "%~dp0main.py"
)

echo.
echo Launching Interactive Visual Dashboard in default browser...
start "" "%~dp0figures\airport_hub_interactive_dashboard.html"

echo.
echo ======================================================================
echo Project executed successfully! Press any key to exit.
echo ======================================================================
pause
