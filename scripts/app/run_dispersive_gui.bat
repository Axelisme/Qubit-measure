@echo off
cd /d "%~dp0..\.."
uv run --directory . --extra gui -- python scripts/app/run_dispersive_gui.py
pause
