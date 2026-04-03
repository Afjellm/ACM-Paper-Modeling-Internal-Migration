@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "MODEL_DIR=%PROJECT_ROOT%\src\automl"
set "VENV_PYTHON=%MODEL_DIR%\.venv\Scripts\python.exe"
set "UV_CACHE_DIR=%PROJECT_ROOT%\.uv-cache"
set "PYTHONPATH=%PROJECT_ROOT%"
set "SETTINGS_FILE=%SCRIPT_DIR%settings.env"
set "ROWS_PER_AGE_GROUP=NONE"

if exist "%SETTINGS_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /r /v /c:"^#" /c:"^$" "%SETTINGS_FILE%"`) do (
        set "%%A=%%B"
    )
)

echo [INFO] Setting up environment for %MODEL_DIR%
pushd "%MODEL_DIR%"
uv sync
popd

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment Python not found at %VENV_PYTHON%
    exit /b 1
)

echo [INFO] Running model
pushd "%PROJECT_ROOT%"
"%VENV_PYTHON%" -m src.automl.autogluon_training %*
popd

endlocal


