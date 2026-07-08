"""FastAPI app for the AIGI detection ensemble demo."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from demo.backend.config import (
    ALLOWED_EXTENSIONS,
    FRONTEND_ORIGINS,
    MAX_UPLOAD_BYTES,
)
from demo.backend.inference import ModelRegistry
from demo.backend.models import HealthResponse, PredictionResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.registry = ModelRegistry()
        logger.info(
            "Models loaded (device=%s, d3qe=%s)",
            app.state.registry.device,
            app.state.registry.d3qe_available,
        )
    except FileNotFoundError as exc:
        logger.error("Model loading failed: %s", exc)
        app.state.registry = None
    yield


app = FastAPI(
    title="AIGI Detection Ensemble",
    description="Detect AI-generated images using a multi-member ensemble",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _get_registry() -> ModelRegistry:
    registry = app.state.registry
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check server logs for missing files.",
        )
    return registry


@app.get("/health", response_model=HealthResponse)
async def health():
    registry = app.state.registry
    if registry is None:
        return HealthResponse(
            status="degraded", d3qe_available=False, device="unknown"
        )
    return HealthResponse(
        status="ok",
        d3qe_available=registry.d3qe_available,
        device=registry.device,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    mode: str = Query(default="fast", pattern="^(fast|full)$"),
):
    registry = _get_registry()

    # Validate extension
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Validate size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents)} bytes). Max: {MAX_UPLOAD_BYTES} bytes.",
        )

    # Validate image
    import io

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Run prediction
    try:
        result = registry.predict(image, mode=mode)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.")

    return result
