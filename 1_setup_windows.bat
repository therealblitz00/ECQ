@echo off
REM Parfois project - one-time setup for Windows (double-click this file).
cd /d "%~dp0"
echo === Parfois setup (Windows) ===

set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY (where python >nul 2>nul && set PY=python)
if not defined PY (
  echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add python.exe to PATH" during installation, then run this file again.
  pause & exit /b 1
)

if not exist .venv (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv || (echo Could not create the virtual environment. & pause & exit /b 1)
)
echo Installing requirements ...
.venv\Scripts\python -m pip install --upgrade pip -q
.venv\Scripts\python -m pip install -r requirements.txt || (echo Installation failed. & pause & exit /b 1)

echo Running the automatic checks ...
.venv\Scripts\python src\sprint1_preprocess.py check || (echo Checks failed. & pause & exit /b 1)
.venv\Scripts\python src\phase1_clean.py || (echo Cleaning failed. & pause & exit /b 1)

echo.
echo Setup complete. Next: double-click 2_review_windows.bat
pause
