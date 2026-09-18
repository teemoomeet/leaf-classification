@echo off
chcp 65001 >nul 2>&1
REM Double-click this file to run fix_github_hosts.ps1 as Administrator.
REM It will pop up a UAC prompt - click "Yes".

setlocal
set "SCRIPT=%~dp0fix_github_hosts.ps1"
if not exist "%SCRIPT%" (
    echo [x] fix_github_hosts.ps1 not found next to this file.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-NoExit','-File','%SCRIPT%'"
endlocal
