@echo off
REM Parfois project - review my Sprint 1 batch in the browser (Windows).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run 1_setup_windows.bat first. & pause & exit /b 1)

echo Team numbers: 1=Andre  2=Pedro Correia  3=Pedro Meireles  4=Manuel  5=Ze
set /p ID=Your team member number (1-5): 
if not exist "outputs\sprint1\batches\batch_0%ID%_of_05.csv" (
  .venv\Scripts\python src\sprint1_preprocess.py batch --members 5 --id %ID% || (pause & exit /b 1)
)

echo.
echo The review app opens in your browser. Your clicks are saved automatically.
echo KEEP THIS WINDOW OPEN while you review. Close it when you are done.
.venv\Scripts\python src\review_app.py --id %ID%
pause
