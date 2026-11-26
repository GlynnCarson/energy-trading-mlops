@echo off
rem Double-click to start everything for the defense demo (see start_demo.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_demo.ps1" %*
pause
