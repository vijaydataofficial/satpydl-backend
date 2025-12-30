# satpydl-backend/app/main.py
import io
import os
import uuid
from pathlib import Path
from typing import Tuple

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image
import numpy as np
import cv2
from skimage import exposure

# App setup
app = FastAPI(title="SatPyDL Backend - With Water")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# mount static outputs
app.mount("/outputs", StaticFiles(directory=str(OUT_DIR)), name="outputs")


# ---------- Helpers ----------
def read_image_bytes(b: bytes) -> Tuple[np.ndarray, Image.Image]:
    pil = Image.open(io.BytesIO(b))
    pil = pil.convert("RGB")
    arr = np.array(pil)
    return arr, pil


def save_array_as_png(arr: np.ndarray, path: Path):
    if arr.dtype != np.uint8:
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        im = Image.fromarray(arr)
    else:
        if arr.shape[2] == 3:
            im = Image.fromarray(arr)
        else:
            arr3 = np.stack([arr[..., 0]]*3, axis=-1) if arr.ndim == 3 else np.stack([arr]*3, axis=-1)
            im = Image.fromarray(arr3.astype(np.uint8))
    im.save(str(path), format="PNG")


def mean_brightness(arr: np.ndarray) -> float:
    r, g, b = arr[..., 0].astype(np.float32), arr[..., 1].astype(np.float32), arr[..., 2].astype(np.float32)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return float(np.mean(lum))


def histogram_stats(arr: np.ndarray):
    stats = {}
    for i, ch in enumerate(["r", "g", "b"]):
        channel = arr[..., i].flatten()
        hist, _ = np.histogram(channel, bins=256, range=(0, 255))
        stats[ch] = {"mean": float(np.mean(channel)), "std": float(np.std(channel)), "hist_len": int(channel.size)}
    return stats


def simple_cloud_mask(arr: np.ndarray, bright_pct=65, blue_ratio_thresh=0.55, min_area=100) -> np.ndarray:
    img = arr.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    bright_thresh = np.percentile(lum, bright_pct)
    blue_ratio = b / (r + g + 1e-6)
    mask = (lum > bright_thresh) | (blue_ratio > blue_ratio_thresh)
    mask = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def ndvi_raw_and_vis(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    nir = (r * 0.5 + g * 0.5)
    denom = (nir + r + 1e-6)
    ndvi = (nir - r) / denom
    ndvi = np.clip(ndvi, -1.0, 1.0)
    ndvi_norm = ((ndvi + 1.0) / 2.0 * 255).astype(np.uint8)
    ndvi_color = cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_JET)
    ndvi_color = cv2.cvtColor(ndvi_color, cv2.COLOR_BGR2RGB)
    return ndvi, ndvi_color


def compute_ndwi_and_vis(arr: np.ndarray, threshold: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute NDWI approximate and produce:
      - ndwi_raw (float -1..1)
      - ndwi_color (visualization, RGB uint8)
      - water_mask (binary 0/255)
      - water_fraction (0..1)
    NDWI = (G - NIR) / (G + NIR) ; NIR approximated from R+G/2
    """
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    nir = (r * 0.5 + g * 0.5)  # heuristic NIR
    denom = (g + nir + 1e-6)
    ndwi = (g - nir) / denom
    ndwi = np.clip(ndwi, -1.0, 1.0)
    ndwi_norm = ((ndwi + 1.0) / 2.0 * 255).astype(np.uint8)
    ndwi_color = cv2.applyColorMap(ndwi_norm, cv2.COLORMAP_OCEAN)
    ndwi_color = cv2.cvtColor(ndwi_color, cv2.COLOR_BGR2RGB)
    water_mask = (ndwi > threshold).astype(np.uint8) * 255
    # remove small speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    water_fraction = float((ndwi > threshold).sum()) / ndwi.size
    return ndwi, ndwi_color, water_mask, water_fraction


def edge_visual(arr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray_eq = exposure.equalize_adapthist(gray, clip_limit=0.03)
    gray_u8 = (gray_eq * 255).astype(np.uint8)
    edges = cv2.Canny(gray_u8, 50, 150)
    return np.stack([edges, edges, edges], axis=-1)


def inpaint_clouds_rgb(img_rgb: np.ndarray, cloud_mask: np.ndarray, method: str = "telea", radius: int = 5, dilate_iter: int = 2) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask_u8 = (cloud_mask > 0).astype(np.uint8) * 255
    if mask_u8.sum() == 0:
        return img_rgb.copy()
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=dilate_iter)
    flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    inpainted_bgr = cv2.inpaint(img_bgr, mask_u8, radius, flags=flags)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    return inpainted_rgb


# ---------- API endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return {"message": "SatPyDL backend running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    img_arr, pil = read_image_bytes(content)
    uid = uuid.uuid4().hex[:10]

    # Save original into uploads and outputs
    orig_name = f"{uid}_orig.png"
    (UPLOAD_DIR / orig_name).write_bytes(content)
    save_array_as_png(img_arr, OUT_DIR / orig_name)

    # Analysis
    hist = histogram_stats(img_arr)
    mean_bright = mean_brightness(img_arr)
    cloud_mask = simple_cloud_mask(img_arr)  # H,W uint8 0/255
    ndvi_raw, ndvi_vis = ndvi_raw_and_vis(img_arr)
    ndwi_raw, ndwi_vis, water_mask, water_frac = compute_ndwi_and_vis(img_arr, threshold=0.05)
    edges = edge_visual(img_arr)

    # Ensure cloud mask saved as RGB for display
    cloud_rgb = np.stack([cloud_mask, cloud_mask, cloud_mask], axis=-1)

    cloud_name = f"{uid}_cloudmask.png"
    ndvi_name = f"{uid}_ndvi.png"
    edges_name = f"{uid}_edges.png"
    ndwi_name = f"{uid}_ndwi.png"
    water_mask_name = f"{uid}_watermask.png"

    save_array_as_png(cloud_rgb, OUT_DIR / cloud_name)
    save_array_as_png(ndvi_vis, OUT_DIR / ndvi_name)
    save_array_as_png(edges, OUT_DIR / edges_name)
    save_array_as_png(ndwi_vis, OUT_DIR / ndwi_name)
    save_array_as_png(water_mask, OUT_DIR / water_mask_name)

    veg_frac = float((ndvi_raw > 0.2).sum()) / ndvi_raw.size
    cloud_frac = float((cloud_mask > 0).sum()) / cloud_mask.size

    print(f"[analyze] uid={uid} cloud_frac={cloud_frac:.3f} veg_frac={veg_frac:.3f} water_frac={water_frac:.3f}")

    response = {
        "status": "ok",
        "mean_brightness": mean_bright,
        "histogram": {k: {"mean": v["mean"], "std": v["std"]} for k, v in hist.items()},
        "cloud_fraction": cloud_frac,
        "vegetation_fraction": veg_frac,
        "water_fraction": water_frac,
        "outputs": {
            "original": orig_name,
            "cloud_mask": cloud_name,
            "ndvi_vis": ndvi_name,
            "edges": edges_name,
            "ndwi_vis": ndwi_name,
            "water_mask": water_mask_name,
        },
    }
    return JSONResponse(content=response)


@app.post("/remove_clouds")
async def remove_clouds(file: UploadFile = File(...), method: str = "telea"):
    content = await file.read()
    img_arr, _pil = read_image_bytes(content)
    uid = uuid.uuid4().hex[:10]

    cloud_mask = simple_cloud_mask(img_arr)
    cleaned = inpaint_clouds_rgb(img_arr, cloud_mask, method=method)

    cleaned_name = f"{uid}_cleaned.png"
    save_array_as_png(cleaned, OUT_DIR / cleaned_name)

    print(f"[remove_clouds] saved cleaned image {cleaned_name} (method={method})")

    buf = io.BytesIO()
    Image.fromarray(cleaned).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/outputs/{fname}")
def get_output(fname: str):
    path = OUT_DIR / fname
    if not path.exists():
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    return FileResponse(str(path), media_type="image/png", filename=fname)
