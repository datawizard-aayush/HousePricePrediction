@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The project virtual environment was not found.
    echo Create it and install the dependencies with:
    echo   py -3.12 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "model_pipeline.joblib" goto train
if not exist "metadata.json" goto train
if not exist "diagnostics.json" goto train
goto run

:train
if not exist "data\raw\mumbai_house_prices_cleaned_cr.csv" (
    echo Training data was not found at data\raw\mumbai_house_prices_cleaned_cr.csv.
    pause
    exit /b 1
)
echo Required model artifacts are missing. Training the models now.
echo This may take several minutes; please keep this window open.
".venv\Scripts\python.exe" train.py
if errorlevel 1 (
    echo.
    echo Model training failed. Check the error above.
    pause
    exit /b 1
)

:run
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo The dashboard could not start. Check the error above.
    pause
    exit /b 1
)

endlocal
