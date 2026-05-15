@echo off
REM Runner for Windows Task Scheduler. Activates the venv and runs one bot cycle.
REM Output is appended to results\stocks\scheduled.log for postmortem review.

cd /d "%~dp0"
".\venv\Scripts\python.exe" -m stocks.bot --once >> "results\stocks\scheduled.log" 2>&1
