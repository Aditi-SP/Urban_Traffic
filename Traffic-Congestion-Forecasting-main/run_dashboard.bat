@echo off
rem Double-click launcher: starts the dashboard with the project's virtual environment (no manual activation).
rem Options are passed through to run_dashboard.ps1, e.g.  run_dashboard.bat -FullCheck   or   run_dashboard.bat -CheckOnly
rem ExecutionPolicy Bypass applies only to this one command and changes no system setting.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dashboard.ps1" %*
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
    echo.
    echo The launcher stopped with exit code %EXITCODE%. Read the messages above.
    pause
)
exit /b %EXITCODE%
