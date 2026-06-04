@echo off
echo ==============================================
echo Uveal Melanoma Prognostic System - Setup ^& Run
echo ==============================================

echo [1/4] Setting up Python Virtual Environment...
IF EXIST ".venv" rd /s /q ".venv"
py -m venv .venv
call .venv\Scripts\activate.bat

echo [2/4] Installing necessary packages...
pip install -r requirement.txt

echo [3/4] Initializing database and training ML Model...
python database\create_db.py
python model\train_model.py

echo ==============================================
echo SETUP COMPLETE!
echo ==============================================
echo [4/4] Starting Web Server...
echo.
echo PLEASE OPEN YOUR BROWSER AND GO TO: http://127.0.0.1:5000/
echo.
python app.py
pause
