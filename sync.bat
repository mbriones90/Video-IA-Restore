@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title VHS Upscaler - Sincronizar entorno

REM ============================================================
REM  sync.bat - PREPARA el entorno (ejecutar UNA VEZ por PC, o
REM  cuando cambies de backend o actualices el proyecto)
REM
REM  Uso:
REM     sync.bat                  Detecta el hardware y sincroniza todo
REM     sync.bat cpu              Fuerza backend CPU
REM     sync.bat cuda             Fuerza backend NVIDIA CUDA
REM     sync.bat directml         Fuerza backend DirectML (AMD/Intel)
REM
REM  Que hace:
REM     0. Instala UV oficialmente si no esta en la PC.
REM     1. Elige el backend y lo guarda en config\backend.txt.
REM     2. uv sync instala las dependencias de ese backend + QTGMC.
REM     3. Descarga los modelos de IA.
REM     4. Instala los plugins de QTGMC (VapourSynth).
REM
REM  Despues de esto, abre la app con iniciar.bat
REM ============================================================

set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=auto"

REM ---------- 0. Instalar UV oficial si falta ----------
where uv >nul 2>&1
if %errorlevel%==0 goto uv_ok
echo.
echo UV no esta instalado. Instalando UV oficialmente...
%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
where uv >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo instalar UV. Instalalo manualmente y reintenta:
    echo         https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)
echo UV instalado correctamente.
:uv_ok

REM ---------- 1. Elegir backend ----------
if /i "%BACKEND%"=="auto" (
    echo Detectando hardware de video...
    for /f "usebackq delims=" %%B in (`%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -Command "$g=(Get-CimInstance Win32_VideoController).Name -join ' '; if ($g -match 'NVIDIA') {'cuda'} elseif ($g -match 'AMD|Radeon|Intel|Arc') {'directml'} else {'cpu'}"`) do (
        set "BACKEND=%%B"
    )
)
if /i not "%BACKEND%"=="cpu" if /i not "%BACKEND%"=="cuda" if /i not "%BACKEND%"=="directml" (
    echo [ERROR] Backend desconocido: %BACKEND%  ^(usa auto, cpu, cuda o directml^)
    pause
    exit /b 1
)
if not exist config mkdir config
echo %BACKEND%> config\backend.txt

echo.
echo ============================================
echo  Backend seleccionado: %BACKEND%
echo  (guardado en config\backend.txt)
echo ============================================
echo.

REM ---------- 2. Sincronizar dependencias ----------
echo [1/4] Sincronizando dependencias (backend %BACKEND% + QTGMC)...
uv sync --group %BACKEND% --group qtgmc
if errorlevel 1 (
    echo.
    echo [ERROR] Fallo uv sync para el backend %BACKEND%.
    if /i "%BACKEND%"=="cuda" echo         Si esta PC no tiene NVIDIA, prueba: sync.bat cpu
    pause
    exit /b 1
)

REM ---------- 3. Modelos de IA ----------
REM Nota: todos los "uv run" llevan los mismos grupos que el sync inicial;
REM un "uv run" sin --group resincroniza el entorno y podria quitar paquetes.
echo.
echo [2/4] Descargando modelos de IA...
uv run --group %BACKEND% --group qtgmc python -m vhs_upscaler.tools.download_models --backend %BACKEND%
if errorlevel 1 (
    echo [ERROR] Fallo la descarga de modelos. Revisa tu conexion e intenta de nuevo.
    pause
    exit /b 1
)

REM ---------- 4. Plugins de QTGMC ----------
echo.
echo [3/4] Instalando plugins de QTGMC (VapourSynth)...
uv run --group %BACKEND% --group qtgmc python tools/install_qtgmc.py --skip-sync
if errorlevel 1 (
    echo.
    echo [AVISO] QTGMC no se pudo instalar/comprobar. La app funcionara con
    echo         bwdif de respaldo. Reintenta despues con:
    echo         uv run python tools\install_qtgmc.py
)

echo.
echo [4/4] Comprobacion final...
uv run --group %BACKEND% --group qtgmc python -c "import vhs_upscaler, onnxruntime, vapoursynth; from vhs_upscaler.hardware import detect_hardware; hw=detect_hardware(); print('Entorno OK | Providers:', ', '.join(hw.available_providers) or 'ninguno')"
if errorlevel 1 (
    echo [ERROR] La comprobacion final fallo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  SINCRONIZACION COMPLETA - backend: %BACKEND%
echo  Abre la aplicacion con: iniciar.bat
echo ============================================================
pause
