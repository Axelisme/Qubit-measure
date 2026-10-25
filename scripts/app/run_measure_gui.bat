@echo off
cd /d "%~dp0..\.."
uv run --directory . --extra gui -- python scripts/app/run_measure_gui.py
pause
