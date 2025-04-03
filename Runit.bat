@echo off
cd /d "%~dp0"

:: Check if 'py' is available
where py >nul 2>nul
if %errorlevel%==0 (
    set PYTHON_CMD=py
) else (
    :: Check if 'python' is available
    where python >nul 2>nul
    if %errorlevel%==0 (
        set PYTHON_CMD=python
    ) else (
        echo Error: Python is not installed or not added to the system PATH.
        pause
        exit /b
    )
)

:: Install required Python packages
echo Installing required Python packages...
%PYTHON_CMD% -m pip install --upgrade pip
%PYTHON_CMD% -m pip install pandas statsmodels scipy numpy numpy-financial openpyxl

:: Run the Python script
echo Running Simulation_Farm_2025.py...
%PYTHON_CMD% Simulation_Farm_2025.py

pause
