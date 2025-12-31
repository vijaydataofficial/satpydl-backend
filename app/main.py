from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import uuid
import os

app = FastAPI()

# ===== CORS FIX =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Backend running OK"}

# ✅ MATCH FRONTEND ENDPOINT
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # ---- HD ENHANCE PIPELINE ----
    h, w = img.shape[:2]
    upscaled = cv2.resize(
        img,
        (w * 2, h * 2),
        interpolation=cv2.INTER_LANCZOS4
    )

    denoised = cv2.fastNlMeansDenoisingColored(
        upscaled, None, 5, 5, 7, 21
    )

    gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.2)
    enhanced = cv2.addWeighted(
        denoised, 1.4,
        gaussian, -0.4,
        0
    )

    filename = f"{uuid.uuid4()}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, enhanced)

    return FileResponse(path, media_type="image/png")
