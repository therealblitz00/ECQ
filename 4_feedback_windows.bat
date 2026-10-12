@echo off
REM Parfois project - teach the similarity engine in the browser (Phase 4, Windows).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run 1_setup_windows.bat first. & pause & exit /b 1)

echo Team numbers: 1=Andre  2=Pedro Correia  3=Pedro Meireles  4=Manuel  5=Ze
set /p ID=Your team member number (1-5): 

echo.
echo The feedback app opens in your browser. Every query you sort is saved automatically.
echo KEEP THIS WINDOW OPEN while you sort. Close it when you are done.
.venv\Scripts\python src\phase4_feedback.py app --id %ID%
pause
