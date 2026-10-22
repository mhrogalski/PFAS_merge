@echo off
setlocal
rem PFAScreen needs Python 3.9 (pyopenms 3.1.0). Not every PC has the "py" launcher,
rem so try it first, then a Python 3.9 on PATH, then the default install folders.
set "SCRIPT=%~dp0batch_PFAScreen.py"
set "CHECK=import sys; sys.exit(sys.version_info[:2] != (3, 9))"

py -3.9 -c "%CHECK%" >nul 2>nul
if not errorlevel 1 (
    py -3.9 "%SCRIPT%"
    goto :end
)

python -c "%CHECK%" >nul 2>nul
if not errorlevel 1 (
    python "%SCRIPT%"
    goto :end
)

for %%P in (
    "%LocalAppData%\Programs\Python\Python39\python.exe"
    "%ProgramFiles%\Python39\python.exe"
    "%ProgramFiles(x86)%\Python39\python.exe"
) do (
    if exist "%%~P" (
        "%%~P" "%SCRIPT%"
        goto :end
    )
)

echo Could not find Python 3.9 on this computer.
echo Install Python 3.9 from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"
echo and "py launcher" in the installer^), then run:
echo     py -3.9 -m pip install pyopenms==3.1.0 numpy pandas xlsxwriter

:end
pause
