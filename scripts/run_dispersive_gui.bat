@echo off
cd /d "%~dp0"
uv run --extra gui python run_dispersive_gui.py
pause
