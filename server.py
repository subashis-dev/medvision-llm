import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from inference import MedVisionPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

pipeline: Optional[MedVisionPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    ckpt_path = "checkpoints/medvision_best.pt"
    pipeline = MedVisionPipeline(checkpoint_path=ckpt_path)
    yield


app = FastAPI(title="MedVision-LLM Diagnostic Engine", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="web"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("web/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "model": "MedVision-LLM",
        "parameters": "6.4M",
        "architecture": "ConvNeXt Visual Stem + Causal Multimodal Transformer",
        "device": pipeline.device if pipeline else "cuda",
        "checkpoint_epoch": getattr(pipeline, "epoch", 0),
        "validation_loss": getattr(pipeline, "val_loss", 0.0),
    }


@app.get("/api/samples")
async def get_samples():
    """Returns verified real Brain MRI test samples across all 4 pathology classes."""
    test_dir = "data/brain_mri_raw/Testing"
    if not os.path.exists(test_dir):
        test_dir = "data/samples"
    curated_map = {
        "glioma_tumor": ["image(59).jpg", "image(63).jpg", "image(64).jpg"],
        "meningioma_tumor": ["image(10).jpg", "image(100).jpg", "image(107).jpg"],
        "pituitary_tumor": ["image(1).jpg", "image(10).jpg", "image(11).jpg"],
        "no_tumor": ["image(1).jpg", "image(10).jpg", "image(100).jpg"],
    }
    categories = {
        "glioma_tumor": {"label": "Glioma Tumor (Brain Parenchyma)", "severity": "High"},
        "meningioma_tumor": {"label": "Meningioma (Dural-Based)", "severity": "Moderate"},
        "pituitary_tumor": {"label": "Pituitary Macroadenoma (Sellar)", "severity": "Moderate"},
        "no_tumor": {"label": "Normal Brain MRI (No Pathology)", "severity": "Normal"},
    }
    samples = []
    for cat_key, meta in categories.items():
        folder = os.path.join(test_dir, cat_key)
        if os.path.exists(folder):
            # Prioritize verified high-confidence benchmark files
            target_files = curated_map.get(cat_key, [])
            actual_files = [f for f in target_files if os.path.exists(os.path.join(folder, f))]
            if not actual_files:
                actual_files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))][:3]
            for fn in actual_files:
                samples.append({
                    "id": f"{cat_key}/{fn}",
                    "category": cat_key,
                    "label": meta["label"],
                    "severity": meta["severity"],
                    "filename": fn,
                    "url": f"/api/sample_image/{cat_key}/{fn}",
                })
    return {"samples": samples}


@app.get("/api/sample_image/{category}/{filename}")
async def serve_sample_image(category: str, filename: str):
    safe_cat = os.path.basename(category)
    safe_fn = os.path.basename(filename)
    path = os.path.join("data", "brain_mri_raw", "Testing", safe_cat, safe_fn)
    if not os.path.exists(path):
        path = os.path.join("data", "samples", safe_cat, safe_fn)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/analyze")
async def analyze_scan(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    prompt: str = Form("Scan: Brain MRI | Task: Analyze clinical case and diagnose"),
    temperature: float = Form(0.4),
):
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is still loading")

    img_bytes = None
    if file and file.filename:
        img_bytes = await file.read()
    elif sample_id:
        safe_parts = sample_id.replace("\\", "/").split("/")
        if len(safe_parts) == 2:
            sample_path = os.path.join("data", "brain_mri_raw", "Testing", safe_parts[0], safe_parts[1])
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    img_bytes = f.read()

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Please upload a scan or select a sample scan.")

    try:
        result = pipeline.analyze_scan(img_bytes, prompt=prompt, temperature=temperature)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 66)
    print("  🧠 MEDVISION-LLM — MEDICAL VLM CLINICAL DIAGNOSTIC SERVER")
    print("  🌐 Web Dashboard : http://localhost:8086")
    print("=" * 66 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8086, reload=False)
