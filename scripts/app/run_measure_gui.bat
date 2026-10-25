@echo off
cd /d "%~dp0..\.."
uv run --directory . --extra gui -- python -m apps.measure
pause
