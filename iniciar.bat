@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title VHS Upscaler

REM ============================================================
REM  iniciar.bat - SOLO abre la aplicacion.
REM
REM  El entorno se prepara con sync.bat (una vez por PC, o al
REM  cambiar de backend). Si algo falta, este script te lo dice.
REM
REM  Uso:
REM     iniciar.bat            Abre con el backend guardado por sync.bat
REM     iniciar.bat cuda       Abre forzando CUDA (despues de
REM                            instalar drivers nuevos)
REM ============================================================

set "BACKEND=%~1"

REM --- UV instalado? ---
where uv >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] UV no esta instalado en esta PC.
    echo         Ejecuta primero: sync.bat
    echo         sync.bat instala UV y todo lo demas solo.
    pause
    exit /b 1
)

REM --- Entorno preparado? ---
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [ERROR] El entorno no esta preparado en esta PC.
    echo         Ejecuta primero: sync.bat
    pause
    exit /b 1
)

REM --- Backend: argumento, luego el guardado por sync.bat, luego auto ---
if "%BACKEND%"=="" if exist "config\backend.txt" set /p BACKEND=<config\backend.txt
if "%BACKEND%"=="" set "BACKEND=auto"
if /i not "%BACKEND%"=="cpu" if /i not "%BACKEND%"=="cuda" if /i not "%BACKEND%"=="directml" set "BACKEND=auto"

if /i "%BACKEND%"=="auto" (
    for /f "usebackq delims=" %%B in (`%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -Command "$g=(Get-CimInstance Win32_VideoController).Name -join ' '; if ($g -match 'NVIDIA') {'cuda'} elseif ($g -match 'AMD|Radeon|Intel|Arc') {'directml'} else {'cpu'}"`) do (
        set "BACKEND=%%B"
    )
)

echo.
echo ============================================
echo  Abriendo VHS Upscaler - backend: %BACKEND%
echo  La CPU siempre queda de respaldo
echo ============================================
echo.

uv run --group %BACKEND% --group qtgmc python -m vhs_upscaler
if errorlevel 1 (
    echo.
    echo [ERROR] La aplicacion termino con un error.
    echo         Si es la primera vez en esta PC, ejecuta: sync.bat
    pause
    exit /b 1
)
