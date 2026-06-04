"""Phase 0.4 smoke test: open_clip ViT-L/14 end-to-end on 10 images.

✓ Gate: embeddings have expected dim (768 for ViT-L/14) and no errors.

Self-contained: synthesizes 10 sample images (no download dependency) so the
gate can run before any dataset arrives. Exercises the real path:
PIL.Image -> open_clip preprocess -> model.encode_image -> embedding tensor.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


def make_sample_images(n: int = 10, size: int = 256) -> list[Image.Image]:
    """Deterministic synthetic RGB images (gradients + noise) for a shape test."""
    rng = np.random.default_rng(config.SEED)
    imgs = []
    for i in range(n):
        # mix a smooth gradient with structured noise so they aren't all alike
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
        base = np.stack([xx, yy, (xx + yy) / 2], axis=-1)
        noise = rng.random((size, size, 3), dtype=np.float32)
        arr = ((0.6 * base + 0.4 * noise) * 255).clip(0, 255).astype(np.uint8)
        # vary per image so embeddings differ
        arr = np.roll(arr, shift=i * 7, axis=0)
        imgs.append(Image.fromarray(arr, mode="RGB"))
    return imgs


def main() -> int:
    config.set_seed()
    device = config.get_device()
    print(f"[smoke] torch {torch.__version__} | device = {device}")

    import open_clip
    print(f"[smoke] open_clip {open_clip.__version__}")
    print(f"[smoke] loading {config.CLIP_MODEL} / {config.CLIP_PRETRAINED} ...")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
    )
    model = model.to(device).eval()
    print(f"[smoke] model loaded in {time.time() - t0:.1f}s")

    imgs = make_sample_images(10, size=config.NORM_CROP_SIZE)
    print(f"[smoke] preprocessing {len(imgs)} images -> tensor")
    batch = torch.stack([preprocess(im) for im in imgs]).to(device)
    print(f"[smoke] preprocessed batch shape: {tuple(batch.shape)}")

    t0 = time.time()
    with torch.no_grad():
        emb = model.encode_image(batch)
    emb = emb.float().cpu()
    dt = time.time() - t0

    print(f"[smoke] embeddings shape : {tuple(emb.shape)}  (dtype={emb.dtype})")
    print(f"[smoke] per-image dim    : {emb.shape[1]}")
    print(f"[smoke] inference time   : {dt:.2f}s for {len(imgs)} imgs")
    print(f"[smoke] embedding stats  : mean={emb.mean():.4f} std={emb.std():.4f} "
          f"nan={torch.isnan(emb).any().item()}")

    ok = (
        emb.shape == (10, config.CLIP_EMBED_DIM)
        and not torch.isnan(emb).any().item()
    )
    print("\n[smoke] GATE:", "PASS ✓" if ok else "FAIL ✗",
          f"(expected shape (10, {config.CLIP_EMBED_DIM}))")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
