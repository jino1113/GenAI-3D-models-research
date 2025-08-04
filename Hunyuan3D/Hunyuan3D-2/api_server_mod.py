from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
import uuid
import base64
from PIL import Image
from io import BytesIO
import os
import logging
import requests

# ========== CONFIG ==========
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# ========== UTILITIES ==========
def load_image_from_base64(b64_str):
    decoded = base64.b64decode(b64_str)
    return Image.open(BytesIO(decoded)).convert("RGB")

def save_temp_image(img: Image.Image, uid: uuid.UUID, view_name: str) -> str:
    path = os.path.join(OUTPUT_FOLDER, f"{uid}_{view_name}.png")
    img.save(path)
    return path

# ========== REAL PIPELINE ==========
class HunyuanAPI:
    def __init__(self, base_url="http://127.0.0.1:8081"):  # ต้องตรงกับเซิร์ฟเวอร์ Hunyuan ที่รัน
        self.base_url = base_url

    def generate(self, uid, params):
        try:
            logger.info(f"Sending request to Hunyuan3D API...")
            response = requests.post(f"{self.base_url}/generate", json=params)

            if response.status_code != 200:
                raise Exception(f"Hunyuan3D API failed: {response.status_code} - {response.text}")

            output_path = os.path.join(OUTPUT_FOLDER, f"{uid}_result.glb")
            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path, {}
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

worker = HunyuanAPI()

# ========== /multiview ENDPOINT ==========
@app.post("/multiview")
async def multiview(request: Request):
    logger.info("Worker multiview...")

    views = await request.json()
    required_keys = ['front', 'back', 'left', 'right']
    for key in required_keys:
        if key not in views:
            return JSONResponse({"error": f"Missing view: {key}"}, status_code=400)

    try:
        uid = uuid.uuid4()

        # แปลง base64 เป็นภาพ และบันทึกไว้ (optional)
        front_img = load_image_from_base64(views['front'])
        back_img = load_image_from_base64(views['back'])
        left_img = load_image_from_base64(views['left'])
        right_img = load_image_from_base64(views['right'])

        save_temp_image(front_img, uid, "front")
        save_temp_image(back_img, uid, "back")
        save_temp_image(left_img, uid, "left")
        save_temp_image(right_img, uid, "right")

        # ส่งเฉพาะ front เข้า model จริง
        params = {
            "image": views['front'],
            "texture": True,
            "octree_resolution": 128,
            "num_inference_steps": 5,
            "guidance_scale": 5.0
        }

        file_path, _ = worker.generate(uid, params)
        return FileResponse(file_path, media_type="model/gltf-binary")

    except Exception as e:
        logger.error(f"Multiview generation error: {e}")
        return JSONResponse({"error": f"Multiview generation failed: {str(e)}"}, status_code=500)

# ========== /generate ENDPOINT ==========
@app.post("/generate")
async def generate(request: Request):
    try:
        data = await request.json()

        if "image" not in data:
            return JSONResponse({"error": "Missing 'image' in payload"}, status_code=400)

        uid = uuid.uuid4()
        file_path, _ = worker.generate(uid, data)

        return FileResponse(file_path, media_type="model/gltf-binary")

    except Exception as e:
        return JSONResponse({"error": f"Generation failed: {str(e)}"}, status_code=500)

# ========== START SERVER ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server_mod:app", host="127.0.0.1", port=8080, reload=True)
