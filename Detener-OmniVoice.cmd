@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Detener-OmniVoice.ps1"
if errorlevel 1 pause
