@echo off
cd /d "%~dp0"
uv run --extra gui python run_measure_gui.py
pause
