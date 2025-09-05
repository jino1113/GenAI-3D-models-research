@echo off
cd /d "%~dp0"

REM 1) สร้าง venv แบบพกพาในโฟลเดอร์โปรเจกต์
python -m venv .venv || goto :err

REM 2) อัปเดต pip และติดตั้ง libs
".venv\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools || goto :err

REM 3) ติดตั้งตาม requirements
".venv\Scripts\pip.exe" install -r requirements.txt || goto :err

REM 4) ติดตั้ง Gradio เวอร์ชันที่ต้องการ
".venv\Scripts\pip.exe" install gradio==3.41.2 || goto :err

REM 5) ติดตั้ง PyTorch ที่ตรงกับ CUDA 12.4
".venv\Scripts\pip.exe" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 || goto :err

REM 6) สร้างโฟลเดอร์แคชภายในโปรเจกต์ (กันไปดึงเน็ตเครื่องใหม่)
if not exist ".hycache" mkdir ".hycache"

echo.
echo Portable venv พร้อมแล้ว. ต่อไปให้ทดสอบรันด้วย Run-Hunyuan3D-2mv.bat
goto :end

:err
echo.
echo !!! เกิดข้อผิดพลาดระหว่าง build. ดูข้อความด้านบน !!!
:end
pause
