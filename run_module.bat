@echo off
title Smart Vision Analysis System
cd /d "%~dp0"

echo ============================================
echo   Smart Vision Analysis System
echo   Nerazurra Dev Studio
echo ============================================
echo.
echo [1] Dashboard Utama
echo [2] Image Detection (YOLO)
echo [3] Motion Detection
echo [4] Anomaly Detection
echo [5] Image Manipulation
echo [0] Exit
echo.
set /p choice="Pilih modul (0-5): "

if "%choice%"=="0" goto :eof
if "%choice%"=="1" python main.py
if "%choice%"=="2" python main.py --module 1
if "%choice%"=="3" python main.py --module 2
if "%choice%"=="4" python main.py --module 3
if "%choice%"=="5" python main.py --module 4

pause
