@echo off
echo Starting Philosophy Q&A System...

:: 启动后端（使用conda环境）
start "Backend (localhost:8000)" cmd /k "cd backend && D:\conda_envs\fastapienv\python.exe main.py"

timeout /t 3 /nobreak > nul

:: 启动前端  
start "Frontend (localhost:5173)" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo System started!
echo Backend: http://localhost:8000/docs
echo Frontend: http://localhost:5173
echo ========================================
pause