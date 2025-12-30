# app/main.py
import io
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="SatPyDL Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUT_DIR)), name="outputs")

# ---------------- BASIC HEALTH ----------------
@app.get("/")
def home():
    return {"message": "SatPyDL backend running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- ANALYZE ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    import numpy as np
    import cv2
    from PIL import Image
    from skimage import exposure

    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    arr = np.array(img)

    uid = uuid.uuid4().hex[:10]
    out_name = f"{uid}_orig.png"
    out_path = OUT_DIR / out_name
    img.save(out_path)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(np.mean(gray))

    edges = cv2.Canny(gray, 50, 150)
    edges_rgb = np.stack([edges]*3, axis=-1)
    edges_name = f"{uid}_edges.png"
    Image.fromarray(edges_rgb).save(OUT_DIR / edges_name)

    return JSONResponse({
        "status": "ok",
        "mean_brightness": mean_brightness,
        "outputs": {
            "original": out_name,
            "edges": edges_name
        }
    })

# ---------------- REMOVE CLOUDS ----------------
@app.post("/remove_clouds")
async def remove_clouds(file: UploadFile = File(...)):
    import numpy as np
    import cv2
    from PIL import Image

    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    arr = np.array(img)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cleaned = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    buf = io.BytesIO()
    Image.fromarray(cleaned).save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
