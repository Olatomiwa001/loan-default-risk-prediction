@echo off
echo Setting up Loan Default Risk Prediction Project

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python from python.org
    pause
    exit /b
)

REM Create project structure
if not exist "src" mkdir src
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment and install dependencies
call venv\Scripts\activate
pip install numpy pandas scikit-learn xgboost matplotlib seaborn

REM Run the project
python src\loan_default_prediction.py

echo Project completed. Check the results.
pause