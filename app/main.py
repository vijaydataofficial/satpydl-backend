from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import uuid
import os

app = FastAPI(title="SatPyDL Image Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_DIM = 2048  # ✅ SAFE FOR CLOUD RUN

def safe_resize(img):
    h, w = img.shape[:2]
    scale = min(MAX_DIM / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image")

        img = safe_resize(img)

        denoise = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
        blur = cv2.GaussianBlur(denoise, (0, 0), 1.0)
        enhanced = cv2.addWeighted(denoise, 1.25, blur, -0.25, 0)

        name = f"{uuid.uuid4()}.png"
        path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(path, enhanced)

        return FileResponse(path, media_type="image/png")

    except Exception as e:
        raise HTTPException(400, str(e))
