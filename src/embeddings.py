"""Phase 3 — CLIP image-embedding extraction + on-disk cache (M1 backbone).

Single source of truth for turning normalized images into frozen `ViT-L-14`
embeddings. Used by BOTH M1a (frozen linear probe, Phase 3) and downstream phases
(combiner Phase 5). Embeddings are keyed by `image_id` so every split reuses the
same cache with zero recompute.

Cache format: one .npz at `cache/clip_<model>_emb.npz` with two aligned arrays:
  - image_id : (N,) unicode  — manifest image_id
  - emb      : (N, D) float32 — L2-normalizable raw CLIP image embedding

Extraction is RESUMABLE: re-running skips image_ids already in the cache and
checkpoints periodically, so an interrupted run (or a crash on the 8GB dev box)
loses at most one checkpoint window.

Env gotcha: python.org Python 3.13 urllib has no CA store, so open_clip's first
download of the OpenAI checkpoint fails TLS. We set SSL_CERT_FILE from certifi
before touching open_clip (no-op if the weights are already cached).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src import config


def _ensure_ssl_certs() -> None:
    """Point urllib at certifi's CA bundle (open_clip 'openai' download gotcha)."""
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass


def cache_path() -> Path:
    """Path to the combined embedding cache for the configured CLIP backbone."""
    tag = config.CLIP_MODEL.replace("/", "-")
    return config.CACHE_DIR / f"clip_{tag}_emb.npz"


def load_cache(path: Path | None = None) -> dict[str, np.ndarray]:
    """Load the cache as {image_id -> embedding}. Empty dict if none yet."""
    path = path or cache_path()
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    ids, emb = data["image_id"], data["emb"]
    return {str(i): emb[k] for k, i in enumerate(ids)}


def save_cache(cache: dict[str, np.ndarray], path: Path | None = None) -> Path:
    """Atomically write {image_id -> embedding} to the npz cache."""
    path = path or cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array(list(cache.keys()), dtype=np.str_)
    emb = np.stack([cache[i] for i in cache]).astype(np.float32)
    # np.savez auto-appends ".npz", so the tmp name must already end in ".npz".
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, image_id=ids, emb=emb)
    os.replace(tmp, path)
    return path


def matrix_for(cache: dict[str, np.ndarray], image_ids) -> tuple[np.ndarray, list[str]]:
    """Stack cached embeddings in `image_ids` order.

    Returns (X, missing) where X is (n_found, D) float32 aligned to the ids that
    WERE present, and `missing` is the list of image_ids absent from the cache.
    """
    rows, missing = [], []
    for i in image_ids:
        i = str(i)
        vec = cache.get(i)
        if vec is None:
            missing.append(i)
        else:
            rows.append(vec)
    X = np.stack(rows).astype(np.float32) if rows else np.empty((0, config.CLIP_EMBED_DIM), np.float32)
    return X, missing


def load_clip(device: str | None = None):
    """Create the frozen CLIP backbone + its preprocess transform (eval mode)."""
    _ensure_ssl_certs()
    import open_clip
    import torch

    device = device or config.get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_grad_enabled(False)
    return model, preprocess, device


def embed_paths(paths: list[Path], model, preprocess, device: str,
                batch_size: int = 32) -> np.ndarray:
    """Encode a list of image paths -> (len(paths), D) float32 embeddings."""
    import torch
    from PIL import Image

    out: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        batch = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in chunk
        ]).to(device)
        with torch.no_grad():
            emb = model.encode_image(batch)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0, config.CLIP_EMBED_DIM), np.float32)
