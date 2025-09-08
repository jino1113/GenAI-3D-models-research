@echo off
setlocal
cd /d "%~dp0"
set "PYEXE=python"
set "PORT=8080"
set "HOST=127.0.0.1"
set "HF_HOME=%CD%\.hycache"
set "HF_HUB_CACHE=%HF_HOME%"
set "TRANSFORMERS_CACHE=%HF_HOME%"
set "XDG_CACHE_HOME=%HF_HOME%"
set "TMP=%HF_HOME%\tmp"
set "TEMP=%HF_HOME%\tmp"
if not exist "%TMP%" mkdir "%TMP%"

start "Hunyuan3D-2mv" cmd /k ^
%PYEXE% gradio_app.py ^
  --model_path tencent/Hunyuan3D-2mv ^
  --subfolder hunyuan3d-dit-v2-mv-turbo ^
  --texgen_model_path tencent/Hunyuan3D-2 ^
  --low_vram_mode ^
  --enable_t23d --enable_flashvdm ^
  --host %HOST% --port %PORT%

for /l %%i in (1,1,600) do (curl -s http://127.0.0.1:%PORT%/info >nul 2>&1 && goto OPEN || timeout /t 1 >nul)
echo Server did not start in time.& goto END
:OPEN
start http://127.0.0.1:%PORT%/
:END
endlocal
