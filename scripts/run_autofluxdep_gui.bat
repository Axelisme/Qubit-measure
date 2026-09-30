@echo off
cd /d "%~dp0"
uv run --extra gui python run_autofluxdep_gui.py
pause
