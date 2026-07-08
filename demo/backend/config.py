"""Demo-specific configuration. Imports shared paths from src.config."""
from __future__ import annotations

import os

from src.config import (
    MODELS_DIR,
    EXTERNAL_DIR,
    NORM_JPEG_QUALITY,
    NORM_CROP_SIZE,
    get_device,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
FRONTEND_ORIGINS = ["http://localhost:3000"]
REQUIRED_MODELS = [
    MODELS_DIR / "m1a_clip_linear.joblib",
    MODELS_DIR / "m2_spectral.joblib",
    MODELS_DIR / "calib_m1.joblib",
    MODELS_DIR / "calib_m2.joblib",
    MODELS_DIR / "combiner_logreg.joblib",
    MODELS_DIR / "combiner_mlp.joblib",
]

D3QE_MODELS = [
    EXTERNAL_DIR / "D3QE" / "pretrained" / "model_epoch_best.pth",
    EXTERNAL_DIR / "D3QE" / "pretrained" / "vq_ds16_c2i.pt",
    MODELS_DIR / "calib_m3.joblib",
]

# Auto-detect D3QE availability: load if all weight files are present,
# unless explicitly overridden via LOAD_D3QE env var.
_d3qe_env = os.environ.get("LOAD_D3QE")
if _d3qe_env is not None:
    LOAD_D3QE = _d3qe_env == "1"
else:
    LOAD_D3QE = all(p.exists() for p in D3QE_MODELS)
