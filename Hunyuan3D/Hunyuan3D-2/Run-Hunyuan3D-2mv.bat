@echo off
setlocal EnableExtensions
cd /d "%~dp0"

:: ---------- Portable Python ----------
set "PYEXE=%CD%\pyhome\Python311\python.exe"
if not exist "%PYEXE%" (
  echo [ERROR] ไม่เจอ portable Python: "%PYEXE%"
  pause & exit /b 1
)

:: ---------- Portable cache ----------
set "HF_HOME=%CD%\.hycache"
set "HF_HUB_CACHE=%HF_HOME%"
set "TRANSFORMERS_CACHE=%HF_HOME%"
set "XDG_CACHE_HOME=%HF_HOME%"
set "TMP=%HF_HOME%\tmp"
set "TEMP=%HF_HOME%\tmp"
if not exist "%TMP%" mkdir "%TMP%"

:: ---------- Config ----------
set "HOST=127.0.0.1"
set "PORT=8080"
set "MAX_WAIT_SECS=1200"
set "CHECK_INTERVAL=1"

echo === Using portable Python ===
"%PYEXE%" -c "import sys;print(sys.executable)" || (echo [ERR] รัน Python ไม่ได้ & pause & exit /b 1)

:: ---------- Run server ----------
start "Hunyuan3D-2mv" "%PYEXE%" "%~dp0gradio_app.py" ^
  --model_path "tencent/Hunyuan3D-2-mv" ^
  --subfolder "hunyuan3d-dit-v2-mv-turbo" ^
  --texgen_model_path "tencent/Hunyuan3D-2" ^
  --low_vram_mode ^
  --enable_t23d --enable_flashvdm ^
  --host %HOST% --port %PORT%

echo.
echo Waiting for port %PORT% to LISTEN (up to %MAX_WAIT_SECS%s) ...

set /a __t=0
:WAIT_LISTEN
for /f "tokens=1-5" %%a in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  echo.
  echo Server is listening on port %PORT%. Opening browser...
  start "" "http://%HOST%:%PORT%/"
  goto END
)
<nul set /p=.
timeout /t %CHECK_INTERVAL% >nul
set /a __t+=%CHECK_INTERVAL%
if %__t% GEQ %MAX_WAIT_SECS% goto TIMEOUT
goto WAIT_LISTEN

:TIMEOUT
echo.
echo [WARN] Server did not become ready within %MAX_WAIT_SECS% seconds.
echo ตรวจหน้าต่างเซิร์ฟเวอร์ "Hunyuan3D-2mv" ว่ามี error หรือไม่
pause

:END
endlocal
