@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Detener-MAGI-Eleven.ps1"
if errorlevel 1 pause
