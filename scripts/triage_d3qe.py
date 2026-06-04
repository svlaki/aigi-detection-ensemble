"""Phase 3 M3 triage: can D3QE load pretrained weights and produce sane logits?

✓ Gate (M3): produces sane per-image fake probabilities on 10 images by Thu noon.
Decision: sane -> 3-member ensemble (M1+M2+M3); broken -> fall back to 2-member.

Runs on CPU (robust; avoids MPS op-support gaps in the VQ-VAE / custom attention).
D3QE expects input 256x256 in [0,1]: VQ-16 -> 16x16 = 256 tokens == token_num.
The checkpoint excludes vq_model/clip_model keys by design, so strict=False is
expected (those come from the LlamaGen VQ ckpt + the vendored CLIP download).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
D3QE_DIR = ROOT / "external" / "D3QE"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D3QE_DIR))   # so `from networks...` and `from .clip` resolve

from src import config  # noqa: E402

VQVAE_PATH = D3QE_DIR / "pretrained" / "vq_ds16_c2i.pt"
CKPT_PATH = D3QE_DIR / "pretrained" / "model_epoch_best.pth"


def make_batch(n=10, size=256, seed=config.SEED):
    """Deterministic synthetic RGB images in [0,1], shape [n,3,size,size]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    imgs = []
    for i in range(n):
        base = np.stack([xx, yy, (xx + yy) / 2], axis=0)          # [3,H,W]
        noise = rng.random((3, size, size), dtype=np.float32)
        arr = (0.6 * base + 0.4 * noise).clip(0, 1)
        arr = np.roll(arr, shift=i * 7, axis=1)
        imgs.append(arr)
    return torch.from_numpy(np.stack(imgs)).float()


def main() -> int:
    config.set_seed()
    device = "cpu"  # triage on CPU for robustness
    print(f"[m3] torch {torch.__version__} | device = {device}")
    print(f"[m3] vqvae : {VQVAE_PATH} ({VQVAE_PATH.exists()})")
    print(f"[m3] ckpt  : {CKPT_PATH} ({CKPT_PATH.exists()})")

    from networks.D3QE import D3QE

    print("[m3] building D3QE (loads VQ-VAE + downloads vendored CLIP ViT-L/14)...")
    model = D3QE(vqvae_path=str(VQVAE_PATH))

    print("[m3] loading classifier checkpoint...")
    state = torch.load(str(CKPT_PATH), map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    info = model.load_state_dict(sd, strict=False)
    # missing keys should be ONLY vq_model.* / clip_model.* (excluded at save time)
    unexpected = list(info.unexpected_keys)
    missing_nonbackbone = [k for k in info.missing_keys
                           if not (k.startswith("vq_model") or k.startswith("clip_model"))]
    print(f"[m3] loaded. unexpected keys: {len(unexpected)} | "
          f"missing non-backbone keys: {len(missing_nonbackbone)}")
    if missing_nonbackbone:
        print("     !! missing (non-backbone):", missing_nonbackbone[:10])
    if unexpected:
        print("     !! unexpected:", unexpected[:10])

    model = model.to(device).eval()

    x = make_batch(10, size=256).to(device)
    print(f"[m3] input batch: {tuple(x.shape)} range=[{x.min():.2f},{x.max():.2f}]")
    with torch.no_grad():
        logits = model(x)              # [B,1]
    logits = logits.float().cpu().flatten()
    probs = torch.sigmoid(logits)

    print(f"[m3] raw logits : {np.round(logits.numpy(), 3).tolist()}")
    print(f"[m3] fake probs : {np.round(probs.numpy(), 4).tolist()}")
    print(f"[m3] prob stats : min={probs.min():.4f} max={probs.max():.4f} "
          f"mean={probs.mean():.4f} std={probs.std():.4f}")

    has_nan = bool(torch.isnan(logits).any())
    in_range = bool((probs >= 0).all() and (probs <= 1).all())
    varied = float(probs.std()) > 1e-4 or float(logits.std()) > 1e-3
    sane = (not has_nan) and in_range and varied
    print(f"\n[m3] sanity: nan={has_nan} in_range={in_range} varied={varied}")
    print("[m3] VERDICT:", "SANE -> 3-member ✓" if sane else "BROKEN -> fall back to 2-member ✗")
    return 0 if sane else 1


if __name__ == "__main__":
    raise SystemExit(main())
