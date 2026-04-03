@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "MODEL_DIR=%PROJECT_ROOT%\src\simulation"
set "VENV_PYTHON=%MODEL_DIR%\.venv\Scripts\python.exe"
set "UV_CACHE_DIR=%PROJECT_ROOT%\.uv-cache"
set "PYTHONPATH=%PROJECT_ROOT%"

echo [INFO] Setting up environment for %MODEL_DIR%
pushd "%MODEL_DIR%"
uv sync
popd

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment Python not found at %VENV_PYTHON%
    exit /b 1
)

echo [INFO] Running simulation
pushd "%PROJECT_ROOT%"
"%VENV_PYTHON%" -m src.simulation.simulate_data %*
popd

endlocal
