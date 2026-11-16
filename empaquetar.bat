@echo off
setlocal
cd /d "%~dp0"
title VHS Upscaler - Empaquetar para otra PC

REM ============================================================
REM  empaquetar.bat - Copia SOLO lo necesario a una carpeta
REM  lista para llevar a otra PC (p. ej. la de la RTX 4070).
REM
REM  Uso:
REM     empaquetar.bat                  Crea ..\Video_upscaling_portable
REM     empaquetar.bat D:\destino       Copia a D:\destino
REM
REM  En la otra PC: ejecuta sync.bat (prepara todo) y luego
REM  iniciar.bat (abre la app).
REM ============================================================

set "DEST=%~1"
if "%DEST%"=="" set "DEST=%~dp0..\Video_upscaling_portable"

REM Normalizar DEST a ruta absoluta (resuelve ".." y relativas)
for %%I in ("%DEST%") do set "DEST=%%~fI"

REM Evitar copiar la carpeta sobre si misma o dentro de si misma
echo "%DEST%\" | findstr /i /b /c:"%CD%\" >nul
if not errorlevel 1 (
    echo [ERROR] El destino no puede ser la carpeta del proyecto ni estar dentro de ella.
    pause
    exit /b 1
)

echo.
echo Copiando a: %DEST%
echo (se excluye .venv, VIDEOS, output, config y caches: sync.bat los recrea)
echo.

robocopy "%~dp0." "%DEST%" /E /XD .venv VIDEOS output config runtime\cache __pycache__ .git .idea .vscode /XF *.pyc *.ffindex *.log /NFL /NDL /NJH /NJS /R:1 /W:1
if errorlevel 8 (
    echo.
    echo [ERROR] Fallo la copia.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  CARPETA LISTA: %DEST%
echo.
echo  En la otra PC:
echo    1. sync.bat      (prepara: instala UV, CUDA, QTGMC, modelos)
echo    2. iniciar.bat   (abre la app)
echo ============================================================
pause
