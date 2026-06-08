"""Phase 3 / M3 — D3QE pretrained detector (external "upside" member).

D3QE (github.com/Zhangyr2022/D3QE) is an autoregressive-image-specialized detector
that fuses VQ-VAE codebook residuals with CLIP features. We use it FROZEN, as a
pretrained specialist — no training, just inference -> per-image fake logit.

Input contract (from networks/D3QE.py forward): raw RGB in [0,1], 256x256. The
model normalizes internally — VAE branch maps [0,1]->[-1,1]; CLIP branch does its
own CenterCrop(224)+CLIP mean/std. So we feed plain ToTensor() images and must NOT
pre-apply CLIP normalization (that's the M1 path, different member).

Reproduction note: we replicate the authors' eval (validate.py) exactly — load
checkpoint, .eval(), model(x).sigmoid(). We deliberately do NOT set
`freq_log_counter`; it is a plain int (not a buffer), so it loads as 0 and the
frequency-difference bias runs in its released default state — i.e. identical to
how the public checkpoint was evaluated. Matching their pipeline > "fixing" it.

strict=False on load is expected: the checkpoint excludes vq_model.*/clip_model.*
keys by design (those come from the VQ ckpt + the vendored CLIP download). We
assert no *non-backbone* keys are missing, which would mean a real load failure.

CLIP download uses urllib -> needs SSL_CERT_FILE (certifi) on python.org 3.13.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from src import config

D3QE_DIR = config.EXTERNAL_DIR / "D3QE"
VQVAE_PATH = D3QE_DIR / "pretrained" / "vq_ds16_c2i.pt"
CKPT_PATH = D3QE_DIR / "pretrained" / "model_epoch_best.pth"
INPUT_SIZE = 256


def _ensure_ssl_certs() -> None:
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass


def cache_path() -> Path:
    return config.CACHE_DIR / "m3_d3qe_logit.npz"


def load_cache(path: Path | None = None) -> dict[str, float]:
    """Load cache as {image_id -> raw fake logit}. Empty dict if none yet."""
    path = path or cache_path()
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    ids, logit = data["image_id"], data["logit"]
    return {str(i): float(logit[k]) for k, i in enumerate(ids)}


def save_cache(cache: dict[str, float], path: Path | None = None) -> Path:
    path = path or cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array(list(cache.keys()), dtype=np.str_)
    logit = np.array([cache[i] for i in cache], dtype=np.float32)
    tmp = path.with_name(path.name + ".tmp.npz")  # np.savez auto-appends ".npz"
    np.savez(tmp, image_id=ids, logit=logit)
    os.replace(tmp, path)
    return path


def vector_for(cache: dict[str, float], image_ids) -> tuple[np.ndarray, list[str]]:
    """Logits aligned to image_ids. Returns (logits (n_found,), missing_ids)."""
    rows, missing = [], []
    for i in image_ids:
        i = str(i)
        if i in cache:
            rows.append(cache[i])
        else:
            missing.append(i)
    return np.array(rows, dtype=np.float32), missing


def load_model(device: str = "cpu"):
    """Build D3QE, load the pretrained checkpoint, return an eval-mode model.

    CPU by default: the VQ-VAE + custom frequency-attention have MPS op-support
    gaps, so CPU is the robust choice on the dev box (matches the triage).
    """
    _ensure_ssl_certs()
    import torch

    if str(config.ROOT) not in sys.path:
        sys.path.insert(0, str(config.ROOT))
    if str(D3QE_DIR) not in sys.path:
        sys.path.insert(0, str(D3QE_DIR))  # so `from networks...`/`from .clip` resolve
    from networks.D3QE import D3QE

    model = D3QE(vqvae_path=str(VQVAE_PATH))
    state = torch.load(str(CKPT_PATH), map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    info = model.load_state_dict(sd, strict=False)
    missing_nonbackbone = [k for k in info.missing_keys
                           if not (k.startswith("vq_model") or k.startswith("clip_model"))]
    if missing_nonbackbone:
        raise RuntimeError(f"D3QE load: missing non-backbone keys {missing_nonbackbone[:10]}")
    if info.unexpected_keys:
        raise RuntimeError(f"D3QE load: unexpected keys {list(info.unexpected_keys)[:10]}")
    return model.to(device).eval()


def logits_for_paths(paths, model, device: str = "cpu", batch_size: int = 8) -> np.ndarray:
    """Encode image paths -> (len(paths),) raw fake logits. ToTensor()=[0,1]."""
    import torch
    import torchvision.transforms as T
    from PIL import Image

    to_tensor = T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor()])
    out: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        batch = torch.stack([
            to_tensor(Image.open(p).convert("RGB")) for p in chunk
        ]).to(device)
        with torch.no_grad():
            logits = model(batch)  # [B, 1]
        out.append(logits.float().cpu().numpy().reshape(-1))
    return np.concatenate(out, axis=0) if out else np.empty((0,), np.float32)
