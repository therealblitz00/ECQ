@echo off
REM Parfois project - open my Sprint 1 review batch (Windows).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run 1_setup_windows.bat first. & pause & exit /b 1)

set /p ID=Your team member number (1-5): 
.venv\Scripts\python src\sprint1_preprocess.py batch --members 5 --id %ID% || (pause & exit /b 1)

start "" "outputs\sprint1\batches\batch_0%ID%_of_05.html"
start "" "outputs\sprint1\batches"
echo.
echo The review page opened in your browser. Fill in batch_0%ID%_of_05.csv (see SPRINT1_GUIDE.md).
pause
