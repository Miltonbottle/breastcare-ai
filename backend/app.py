"""FastAPI backend for the BreastCare AI segmentation and geometry pipeline."""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


SOURCE_DIR = Path(__file__).resolve().parents[1]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from agent import BreastCareAgent  # noqa: E402
from agent.schemas import AgentWorkflowError  # noqa: E402
from inference import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    load_model,
)


OUTPUT_DIR = DEFAULT_OUTPUT_DIR
RESULTS_ROUTE = "/results"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: torch.nn.Module | None = None
MODEL_ERROR: str | None = None
MODEL_LOCK = threading.Lock()
AGENT: BreastCareAgent | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the checkpoint once so requests reuse the same model instance."""
    global MODEL, MODEL_ERROR, AGENT
    try:
        MODEL = load_model(DEFAULT_CHECKPOINT, DEVICE)
        AGENT = BreastCareAgent(MODEL, DEVICE, OUTPUT_DIR, MODEL_LOCK)
        MODEL_ERROR = None
    except Exception as exc:
        MODEL = None
        AGENT = None
        MODEL_ERROR = str(exc)
    yield
    MODEL = None
    AGENT = None


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="BreastCare AI Backend", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(RESULTS_ROUTE, StaticFiles(directory=str(OUTPUT_DIR)), name="results")


def _result_url(path: Path) -> str:
    """Convert a generated file to a non-absolute URL exposed by this API."""
    return f"{RESULTS_ROUTE}/{path.name}"


def _temporary_upload_path(upload: UploadFile) -> Path:
    """Create a unique temporary path using a permitted image suffix."""
    filename = upload.filename or "uploaded_image"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".upload"
    return Path(tempfile.gettempdir()) / f"breastcare_{uuid.uuid4().hex}{suffix}"


def _run_pipeline(temp_image_path: Path) -> dict[str, Any]:
    """Run the bounded BreastCareAgent workflow using the preloaded model."""
    if AGENT is None or not AGENT.ready:
        detail = "Model is not available."
        if MODEL_ERROR:
            detail = f"Model is not available: {MODEL_ERROR}"
        raise HTTPException(status_code=503, detail=detail)

    try:
        return AGENT.execute(temp_image_path)
    except AgentWorkflowError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"message": str(exc), "agent_trace": exc.trace_dict()},
        ) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    """Return backend readiness without exposing filesystem details."""
    response: dict[str, Any] = {
        "status": "ok" if MODEL is not None else "unavailable",
        "model_loaded": MODEL is not None,
        "device": DEVICE.type,
    }
    if MODEL is None and MODEL_ERROR:
        response["message"] = "Model checkpoint could not be loaded."
    return response


@app.get("/agent/status")
def agent_status() -> dict[str, Any]:
    """Report whether the preloaded model and agent workflow are ready."""
    return {
        "status": "ok" if AGENT is not None and AGENT.ready else "unavailable",
        "agent_ready": AGENT is not None and AGENT.ready,
        "model_loaded": MODEL is not None,
        "device": DEVICE.type,
    }


@app.post("/analyze")
def analyze_upload(image: UploadFile | None = File(default=None)) -> dict[str, Any]:
    """Analyze one uploaded ultrasound image through the existing local pipeline."""
    if image is None or not image.filename:
        raise HTTPException(status_code=400, detail="An image upload is required.")

    temp_path = _temporary_upload_path(image)
    try:
        with temp_path.open("wb") as temporary_file:
            shutil.copyfileobj(image.file, temporary_file)
        if temp_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")

        result = _run_pipeline(temp_path)
    finally:
        image.file.close()
        temp_path.unlink(missing_ok=True)

    output_paths: dict[str, Path] = result["output_paths"]
    return {
        "success": True,
        "image": {
            "filename": Path(image.filename).name,
            "width": result["image_width"],
            "height": result["image_height"],
            "original_path": _result_url(output_paths["original"]),
        },
        "segmentation": {
            "lesion_area_percentage": result["features"]["lesion_area_percentage"],
            "mask_path": _result_url(output_paths["binary_mask"]),
            "overlay_path": _result_url(output_paths["overlay"]),
        },
        "features": result["features"],
        "analysis": result["analysis"],
        "segmentation_quality": result["segmentation_quality"],
        "model": {
            "name": "HybridViTGABVSSMUNet",
            "checkpoint": "fold_1_best_model.pth",
        },
        "performance": {"inference_time_seconds": result["inference_seconds"]},
        "agent_trace": result["agent_trace"],
    }
