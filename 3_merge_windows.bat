@echo off
REM Parfois project - merge the 5 reviewed batches (run by ONE person once all reviews are in).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run 1_setup_windows.bat first. & pause & exit /b 1)
.venv\Scripts\python src\phase1_checks.py check
.venv\Scripts\python src\phase1_checks.py merge
pause
