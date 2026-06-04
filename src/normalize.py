"""Phase 1.5 normalization (CRITICAL — Grommelt confound control).

Every image (real + fake, all pools) goes through the SAME pipeline before any
feature extraction:
  1. decode -> RGB (drop alpha/palette/grayscale differences)
  2. center-crop to a square, then resize to NORM_CROP_SIZE (common resolution)
  3. re-encode to JPEG quality 95 (common compression)
This removes the JPEG/size shortcuts detectors otherwise key on.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from src import config


def normalize_pil(img: Image.Image, size: int = config.NORM_CROP_SIZE) -> Image.Image:
    """RGB + center-square-crop + resize to (size, size). Returns a PIL image."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return img


def save_normalized(img: Image.Image, out_path: Path,
                    quality: int = config.NORM_JPEG_QUALITY,
                    size: int = config.NORM_CROP_SIZE) -> Path:
    """Normalize and write a JPEG q95. Returns the output path."""
    out_path = Path(out_path).with_suffix(".jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm = normalize_pil(img, size=size)
    norm.save(out_path, format="JPEG", quality=quality)
    return out_path
