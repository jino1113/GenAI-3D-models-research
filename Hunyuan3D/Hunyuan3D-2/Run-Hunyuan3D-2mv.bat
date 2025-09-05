@echo off
cd /d "%~dp0"

set "PYEXE=.venv\Scripts\python.exe"
set PORT=8080
set HOST=0.0.0.0
set HEALTH=http://127.0.0.1:%PORT%/info

REM บังคับให้ใช้แคช/โมเดลภายในโฟลเดอร์ (ออฟไลน์ได้)
set HF_HOME=%CD%\.hycache
set TRANSFORMERS_CACHE=%CD%\.hycache
set XDG_CACHE_HOME=%CD%\.hycache

REM รันเซิร์ฟเวอร์ในหน้าต่างแยก (เห็น log)
start "Hunyuan3D-2mv" cmd /k ^
"%PYEXE%" gradio_app.py ^
  --model_path tencent/Hunyuan3D-2mv ^
  --subfolder hunyuan3d-dit-v2-mv ^
  --texgen_model_path tencent/Hunyuan3D-2 ^
  --low_vram_mode ^
  --host %HOST% --port %PORT% ^
  --cache-path ".hycache"

echo Waiting for server at %HEALTH% ...

REM เช็คจนกว่าจะตอบ 200 (สูงสุด 600 วิ)
for /l %%i in (1,1,600) do (
  curl -s "%HEALTH%" >nul 2>&1
  if not errorlevel 1 goto :OPEN
  timeout /t 1 >nul
)

echo Server did not become ready in time.
goto :END

:OPEN
start http://127.0.0.1:%PORT%/
echo Browser opened.
:END
