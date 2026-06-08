"""Phase 3 / M2 — spectral (FFT) features. Decorrelated-by-construction member.

M2 ignores image *content* (M1's job) and looks only at the frequency-domain
fingerprint generators leave behind (upsampling/deconvolution periodicities,
anomalous high-frequency falloff). Different inductive bias => decorrelated errors
=> the ensemble + combiner have something to gain (Phase 4 measures this).

Feature vector per image (compact, rotation-robust):
  - azimuthally-averaged radial power spectrum (1D energy-vs-spatial-frequency
    curve), resampled to RADIAL_BINS points and log-scaled
  - a few high-frequency energy ratios (fraction of spectral energy beyond
    increasing radii) summarizing the falloff

Deliberately NO CLIP normalization — M2 wants raw pixel spectra. Operates on the
same normalized 256px JPEGs as every other member (identical Grommelt-controlled
conditions). Cache is keyed by image_id, mirroring src/embeddings.py, so Phase 5
reuses it with zero recompute.

Caveat (report it honestly): JPEG q95 recompression attenuates exactly the
high-frequency artifacts M2 keys on, so M2's task is genuinely harder here. That
is the fair, confound-free setup.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src import config

RADIAL_BINS = 64                      # length of the resampled radial PSD curve
HF_RATIO_FRACS = (0.5, 0.6, 0.7, 0.8, 0.9)  # outer-radius cutoffs for HF energy ratios
FEATURE_DIM = RADIAL_BINS + len(HF_RATIO_FRACS)


def cache_path() -> Path:
    """Path to the spectral-feature cache."""
    return config.CACHE_DIR / "m2_spectral_feat.npz"


def load_cache(path: Path | None = None) -> dict[str, np.ndarray]:
    """Load the cache as {image_id -> feature vector}. Empty dict if none yet."""
    path = path or cache_path()
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    ids, feat = data["image_id"], data["feat"]
    return {str(i): feat[k] for k, i in enumerate(ids)}


def save_cache(cache: dict[str, np.ndarray], path: Path | None = None) -> Path:
    """Atomically write {image_id -> feature vector} to the npz cache."""
    path = path or cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array(list(cache.keys()), dtype=np.str_)
    feat = np.stack([cache[i] for i in cache]).astype(np.float32)
    tmp = path.with_name(path.name + ".tmp.npz")  # np.savez auto-appends ".npz"
    np.savez(tmp, image_id=ids, feat=feat)
    os.replace(tmp, path)
    return path


def matrix_for(cache: dict[str, np.ndarray], image_ids) -> tuple[np.ndarray, list[str]]:
    """Stack cached features in `image_ids` order. Returns (X, missing_ids)."""
    rows, missing = [], []
    for i in image_ids:
        i = str(i)
        vec = cache.get(i)
        if vec is None:
            missing.append(i)
        else:
            rows.append(vec)
    X = np.stack(rows).astype(np.float32) if rows else np.empty((0, FEATURE_DIM), np.float32)
    return X, missing


def _radial_profile(power: np.ndarray) -> np.ndarray:
    """Azimuthal average of a 2D power spectrum -> 1D curve indexed by radius."""
    h, w = power.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), weights=power.ravel())
    nbin = np.bincount(r.ravel())
    return tbin / np.maximum(nbin, 1)


def features_from_image(path: Path) -> np.ndarray:
    """Normalized 256px JPEG -> spectral feature vector (FEATURE_DIM,)."""
    from PIL import Image

    img = Image.open(path).convert("L")  # luminance; spectral artifacts are HF-luma
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr - arr.mean()               # remove DC so it doesn't dominate

    f = np.fft.fftshift(np.fft.fft2(arr))
    power = np.abs(f) ** 2
    logp = np.log1p(power)

    radial = _radial_profile(logp)
    # drop the DC bin, resample the curve to a fixed length (image-size robust)
    radial = radial[1:]
    src_x = np.linspace(0.0, 1.0, num=len(radial))
    dst_x = np.linspace(0.0, 1.0, num=RADIAL_BINS)
    radial_rs = np.interp(dst_x, src_x, radial).astype(np.float32)

    # high-frequency energy ratios off the (linear) radial power profile
    lin_radial = _radial_profile(power)[1:]
    total = lin_radial.sum() + 1e-8
    n = len(lin_radial)
    hf = np.array([lin_radial[int(frac * n):].sum() / total
                   for frac in HF_RATIO_FRACS], dtype=np.float32)

    return np.concatenate([radial_rs, hf]).astype(np.float32)
