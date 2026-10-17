@echo off
setlocal
cd /d "%~dp0"
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

where python >nul 2>&1
if errorlevel 1 (
    echo Python bulunamadi. python.org uzerinden 3.10+ kurun ve PATH'e ekleyin.
    exit /b 1
)

python -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 exit /b 1

python "%~dp0generate_ist27_mock_logs.py"
if errorlevel 1 exit /b 1

python -m streamlit run "%~dp0app.py"
exit /b %ERRORLEVEL%
