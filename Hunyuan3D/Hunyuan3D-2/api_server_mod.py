from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
import uuid
import base64
from PIL import Image
from io import BytesIO
import os
import logging
import requests

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

def load_image_from_base64(b64_str):
    decoded = base64.b64decode(b64_str)
    return Image.open(BytesIO(decoded)).convert("RGB")

def save_temp_image(img: Image.Image, uid: uuid.UUID, view_name: str) -> str:
    path = os.path.join(OUTPUT_FOLDER, f"{uid}_{view_name}.png")
    img.save(path)
    return path

class HunyuanAPI:
    def __init__(self, base_url="http://127.0.0.1:8081"):  # ชี้ไปที่เซิร์ฟเวอร์หลัก
        self.base_url = base_url

    def generate_multiview(self, uid, params):
        try:
            logger.info(f"Sending request to Hunyuan3D /multiview API...")
            multiview_params = {k: params[k] for k in ['front', 'back', 'left', 'right'] if k in params}
            multiview_params.update({
                "texture": params.get("texture", True),
                "octree_resolution": params.get("octree_resolution", 512),
                "num_inference_steps": params.get("num_inference_steps", 20),
                "guidance_scale": params.get("guidance_scale", 5.5)
            })

            response = requests.post(f"{self.base_url}/multiview", json=multiview_params)

            if response.status_code != 200:
                raise Exception(f"Hunyuan3D API failed: {response.status_code} - {response.text}")

            output_path = os.path.join(OUTPUT_FOLDER, f"{uid}_result.glb")
            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

worker = HunyuanAPI()

@app.post("/multiview")
async def multiview(request: Request):
    logger.info("Worker multiview...")
    views = await request.json()

    allowed_keys = ['front', 'back', 'left', 'right']
    uid = uuid.uuid4()

    # ตรวจว่ามีอย่างน้อย 1 มุม
    if not any(k in views for k in allowed_keys):
        return JSONResponse({"error": "At least one view must be provided."}, status_code=400)

    try:
        # save เฉพาะมุมที่มี
        for key in allowed_keys:
            if key in views:
                img = load_image_from_base64(views[key])
                save_temp_image(img, uid, key)

        file_path = worker.generate_multiview(uid, views)
        return FileResponse(file_path, media_type="model/gltf-binary")

    except Exception as e:
        logger.error(f"Multiview generation error: {e}")
        return JSONResponse({"error": f"Multiview generation failed: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server_mod:app", host="127.0.0.1", port=8080, reload=True)
