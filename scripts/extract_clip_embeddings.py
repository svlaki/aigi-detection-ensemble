"""Phase 3 — extract & cache frozen CLIP embeddings for the WHOLE manifest.

Runs the M1 backbone (ViT-L-14-quickgelu/openai) over every normalized image in
the manifest and caches embeddings keyed by image_id. ALL splits in one pass, so
M1a (Phase 3), the combiner (Phase 5), and the M1b/LoRA ablation (Phase 6) all
reuse the same cache with zero recompute.

Resumable: skips image_ids already cached, checkpoints every --checkpoint images.

Usage (dev box, MPS):
  SSL_CERT_FILE=$(./.venv/bin/python -c 'import certifi;print(certifi.where())') \
  ./.venv/bin/python scripts/extract_clip_embeddings.py
  # optional: --limit N (smoke), --batch-size 32, --checkpoint 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, embeddings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--checkpoint", type=int, default=1000,
                    help="save the cache every N newly-embedded images")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N pending images (0 = all)")
    args = ap.parse_args()

    config.set_seed()
    df = pd.read_csv(config.MANIFEST_CSV)
    cache = embeddings.load_cache()
    print(f"[extract] manifest rows: {len(df)} | already cached: {len(cache)}")

    pending = df[~df["image_id"].isin(cache.keys())].reset_index(drop=True)
    if args.limit:
        pending = pending.head(args.limit)
    if pending.empty:
        print("[extract] nothing to do — cache is complete.")
        print(f"[extract] cache at: {embeddings.cache_path()}")
        return 0

    model, preprocess, device = embeddings.load_clip()
    print(f"[extract] device={device} | pending={len(pending)} | "
          f"batch={args.batch_size} | checkpoint={args.checkpoint}")

    done_since_ckpt = 0
    t0 = time.time()
    for start in range(0, len(pending), args.batch_size):
        rows = pending.iloc[start:start + args.batch_size]
        paths = [config.ROOT / p for p in rows["path"]]
        emb = embeddings.embed_paths(paths, model, preprocess, device,
                                     batch_size=len(paths))
        for k, image_id in enumerate(rows["image_id"]):
            cache[str(image_id)] = emb[k]
        done_since_ckpt += len(rows)

        n_done = start + len(rows)
        rate = n_done / max(time.time() - t0, 1e-6)
        eta = (len(pending) - n_done) / max(rate, 1e-6)
        print(f"[extract] {n_done}/{len(pending)}  "
              f"({rate:.1f} img/s, eta {eta/60:.1f}m)", flush=True)

        if done_since_ckpt >= args.checkpoint:
            embeddings.save_cache(cache)
            print(f"[extract]   ✓ checkpoint saved ({len(cache)} total)", flush=True)
            done_since_ckpt = 0

    embeddings.save_cache(cache)
    dt = time.time() - t0
    print(f"[extract] DONE — {len(cache)} embeddings cached in {dt/60:.1f}m")
    print(f"[extract] cache at: {embeddings.cache_path()}")

    # sanity gate
    import numpy as np
    sample = np.stack([cache[i] for i in list(cache)[:64]])
    ok = sample.shape[1] == config.CLIP_EMBED_DIM and not np.isnan(sample).any()
    print("[extract] GATE:", "PASS ✓" if ok else "FAIL ✗",
          f"(dim={sample.shape[1]}, expected {config.CLIP_EMBED_DIM})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
