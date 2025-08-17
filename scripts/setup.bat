@echo off
:: Setup script for rag-advanced project (Windows)

echo [*] Setting up rag-advanced project...

:: Check if UV is installed
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [*] UV not found. Installing UV...
    pip install uv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install UV
        exit /b 1
    )
)

:: Create virtual environment
echo [*] Creating virtual environment...
uv venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment
    exit /b 1
)

:: Activate and install dependencies
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment
    exit /b 1
)

echo [*] Installing dependencies...
uv pip install -e ".[dev]"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies
    exit /b 1
)

:: Setup environment
if not exist ".env" (
    echo [*] Creating .env file from example...
    copy /Y .env.example .env >nul
    echo [*] Please edit the .env file with your configuration
)

echo [*] Setup complete!
echo [*] To activate the virtual environment, run:
echo     .venv\Scripts\activate.bat
echo [*] Start the app with:
echo     streamlit run app/ui/streamlit_app.py
