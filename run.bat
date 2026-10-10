@echo off
title Offline Face Finder
cd /d "%~dp0"
echo Starting Offline Face Finder...
start "" "%~dp0\.venv\Scripts\pythonw.exe" "%~dp0\main.py"
exit
