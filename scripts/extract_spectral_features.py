"""Phase 3 / M2 — extract & cache spectral (FFT) features for the WHOLE manifest.

CPU-only, fast. Resumable: skips image_ids already cached, checkpoints every
--checkpoint images. All splits in one pass so Phase 5 reuses with no recompute.

Usage:
  ./.venv/bin/python scripts/extract_spectral_features.py
  # optional: --limit N (smoke), --checkpoint 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, spectral


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    config.set_seed()
    df = pd.read_csv(config.MANIFEST_CSV)
    cache = spectral.load_cache()
    print(f"[m2-extract] manifest rows: {len(df)} | already cached: {len(cache)}")

    pending = df[~df["image_id"].isin(cache.keys())].reset_index(drop=True)
    if args.limit:
        pending = pending.head(args.limit)
    if pending.empty:
        print("[m2-extract] nothing to do — cache complete.")
        print(f"[m2-extract] cache at: {spectral.cache_path()}")
        return 0

    print(f"[m2-extract] pending={len(pending)} | dim={spectral.FEATURE_DIM} "
          f"| checkpoint={args.checkpoint}")
    done_since_ckpt = 0
    t0 = time.time()
    for k in range(len(pending)):
        row = pending.iloc[k]
        feat = spectral.features_from_image(config.ROOT / row["path"])
        cache[str(row["image_id"])] = feat
        done_since_ckpt += 1

        if (k + 1) % 500 == 0 or k + 1 == len(pending):
            rate = (k + 1) / max(time.time() - t0, 1e-6)
            eta = (len(pending) - (k + 1)) / max(rate, 1e-6)
            print(f"[m2-extract] {k+1}/{len(pending)}  "
                  f"({rate:.0f} img/s, eta {eta/60:.1f}m)", flush=True)
        if done_since_ckpt >= args.checkpoint:
            spectral.save_cache(cache)
            print(f"[m2-extract]   ✓ checkpoint ({len(cache)} total)", flush=True)
            done_since_ckpt = 0

    spectral.save_cache(cache)
    print(f"[m2-extract] DONE — {len(cache)} features cached in "
          f"{(time.time()-t0)/60:.1f}m -> {spectral.cache_path()}")

    sample = np.stack([cache[i] for i in list(cache)[:64]])
    ok = sample.shape[1] == spectral.FEATURE_DIM and not np.isnan(sample).any()
    print("[m2-extract] GATE:", "PASS ✓" if ok else "FAIL ✗",
          f"(dim={sample.shape[1]}, expected {spectral.FEATURE_DIM})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
