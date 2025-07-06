@echo off
echo Starting all Python services for Cannes project...
echo.

REM Start ETL Service (Port 3030)
echo Starting ETL Service on port 3030...
start "ETL Service" cmd /k "cd etl && python app.py"

REM Wait a moment before starting next service
timeout /t 3

REM Start Preprocessing Service (Port 3031)
echo Starting Preprocessing Service on port 3031...
start "Preprocessing Service" cmd /k "cd preprocessing && python app.py"

REM Wait a moment before starting next service
timeout /t 3

REM Start EDA Service (Port 3035)
echo Starting EDA Service on port 3035...
start "EDA Service" cmd /k "cd eda && python app.py"

REM Wait a moment before starting next service
timeout /t 3

REM Start Analysis Service (Port 3040)
echo Starting Analysis Service on port 3040...
start "Analysis Service" cmd /k "cd analysis && python app.py"

echo.
echo All services are starting up...
echo Check each service window for startup messages.
echo.
echo Services:
echo - ETL Service: http://localhost:3030
echo - Preprocessing Service: http://localhost:3031
echo - EDA Service: http://localhost:3035
echo - Analysis Service: http://localhost:3040
echo.
echo Press any key to exit this window...
pause 