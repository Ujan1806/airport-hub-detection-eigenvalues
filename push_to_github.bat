@echo off
title Push Airport Hub Detection to GitHub
echo ======================================================================
echo    Pushing Airport Hub Detection Project to GitHub Repository
echo    https://github.com/Ujan1806/airport-hub-detection-eigenvalues
echo ======================================================================
echo.

set GIT_PATH="C:\Users\SANJOY DUTTA\AppData\Local\GitHubDesktop\app-3.5.4\resources\app\git\cmd\git.exe"

%GIT_PATH% push -u origin main

echo.
echo ======================================================================
echo Push complete! Check your repository at:
echo https://github.com/Ujan1806/airport-hub-detection-eigenvalues
echo ======================================================================
pause
