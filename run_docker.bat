@echo off
echo.
echo [CipherVerse] Building and starting Docker containers...
echo.

docker compose up --build -d

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start Docker containers.
    echo Please make sure Docker Desktop is installed and running.
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [CipherVerse] Backend is running at http://localhost:8000
echo [CipherVerse] API Documentation is available at http://localhost:8000/docs
echo.
echo Press any key to stop the containers...
pause

docker compose down
echo.
echo [CipherVerse] Containers stopped.
pause
