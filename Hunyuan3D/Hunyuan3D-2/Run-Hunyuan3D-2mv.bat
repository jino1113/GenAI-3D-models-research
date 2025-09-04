@echo off
cd /d "%~dp0"

rem รัน Gradio App ในหน้าต่างนี้
start "Hunyuan3D-2mv" cmd /k ^
python gradio_app.py --model_path tencent/Hunyuan3D-2mv ^
  --subfolder hunyuan3d-dit-v2-mv ^
  --texgen_model_path tencent/Hunyuan3D-2 ^
  --low_vram_mode ^
  --host 0.0.0.0 --port 8080

rem รอจนกว่า server พร้อม (เช็คทุก 1 วิ)
echo Waiting for server at http://127.0.0.1:8080/info ...
for /l %%i in (1,1,300) do (
  curl -s http://127.0.0.1:8080/info >nul 2>&1
  if not errorlevel 1 goto :OPEN
  timeout /t 1 >nul
)

echo Server did not start in time.
goto :END

:OPEN
start http://127.0.0.1:8080/
echo Opened browser.

:END
pause
