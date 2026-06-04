"""Bounded diversity-aware re-pull of Community Forensics (Pool C terminal).

CompEval is 51,836 rows / 206 GB, generators grouped by shard, so streaming the
head only yields the first 1-2 generators. We instead read a SPREAD of shards
(one HF stream per shard, capped) to capture many generators, with a per-generator
cap and a hard wall-clock budget so it always finishes. Replaces the old cf_eval
rows in the manifest. Falls back gracefully: whatever it collects in-budget is used.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src import manifest as M
from src.normalize import save_normalized

REPO = "OwensLab/CommunityForensics-Eval"
N_SHARDS_TOTAL = 413
N_SHARDS_PICK = 30          # spread across the set for generator diversity
PER_SHARD = 140             # max rows to read per shard
GEN_CAP = 90               # max fakes per generator (forces diversity)
FAKE_TARGET = 1000
REAL_TARGET = 1000
BUDGET_S = 720             # hard 12-min abort
KEY = "cf_eval"


def shard_url(idx: int) -> str:
    return (f"hf://datasets/{REPO}/data/"
            f"CompEval-{idx:05d}-of-{N_SHARDS_TOTAL:05d}.parquet")


def to_pil(val):
    if isinstance(val, (bytes, bytearray)):
        return Image.open(io.BytesIO(val))
    if isinstance(val, dict) and val.get("bytes"):
        return Image.open(io.BytesIO(val["bytes"]))
    if isinstance(val, Image.Image):
        return val
    return None


def main():
    config.ensure_dirs(); config.set_seed()
    # evenly spread shard indices
    step = N_SHARDS_TOTAL // N_SHARDS_PICK
    shards = list(range(0, N_SHARDS_TOTAL, step))[:N_SHARDS_PICK]
    print(f"[cf] picking {len(shards)} shards (step {step}): {shards[:6]}...")

    out_dir = config.NORM_DIR / KEY
    # wipe old cf_eval files + manifest rows
    if out_dir.exists():
        for p in out_dir.glob("*.jpg"):
            p.unlink()
    df = pd.read_csv(config.MANIFEST_CSV) if config.MANIFEST_CSV.exists() \
        else pd.DataFrame(columns=config.MANIFEST_COLUMNS)
    df = df[df["source_dataset"] != "community_forensics"].copy()
    print(f"[cf] manifest after dropping old CF: {len(df)} rows")

    rows, gen_count = [], {}
    n_fake = n_real = 0
    t0 = time.time()
    for si, idx in enumerate(shards):
        if time.time() - t0 > BUDGET_S:
            print(f"[cf] budget hit before shard {idx}; stopping")
            break
        if n_fake >= FAKE_TARGET and n_real >= REAL_TARGET:
            break
        try:
            ds = load_dataset("parquet", data_files=[shard_url(idx)],
                              split="train", streaming=True)
        except Exception as e:
            print(f"[cf] shard {idx} load err {repr(e)[:80]}")
            continue
        kept_here = 0
        for ex in ds:
            if kept_here >= PER_SHARD:
                break
            if time.time() - t0 > BUDGET_S:
                break
            label = ex.get("label")
            if label == 1:
                if n_fake >= FAKE_TARGET:
                    continue
                gen = str(ex.get("model_name", "")).strip().lower().replace(" ", "_") or "unknown"
                if gen_count.get(gen, 0) >= GEN_CAP:
                    continue
            elif label == 0:
                if n_real >= REAL_TARGET:
                    continue
                gen = "real"
            else:
                continue
            img = to_pil(ex.get("image_data"))
            if img is None:
                continue
            image_id = f"{KEY}_{label}_{(n_fake if label==1 else n_real):06d}"
            try:
                out_path = save_normalized(img, out_dir / image_id)
            except Exception:
                continue
            rows.append({"image_id": image_id,
                         "path": str(out_path.relative_to(config.ROOT)),
                         "label": int(label), "source_dataset": "community_forensics",
                         "generator_name": gen, "pool": "C"})
            kept_here += 1
            if label == 1:
                n_fake += 1; gen_count[gen] = gen_count.get(gen, 0) + 1
            else:
                n_real += 1
        print(f"[cf] shard {idx} (#{si+1}/{len(shards)}): kept {kept_here} | "
              f"fake={n_fake} real={n_real} gens={len(gen_count)} | "
              f"{time.time()-t0:.0f}s")

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(config.MANIFEST_CSV, index=False)
    print(f"\n[cf] === re-pull done in {time.time()-t0:.0f}s ===")
    print(f"[cf] new CF: fake={n_fake} real={n_real} | distinct fake generators={len(gen_count)}")
    print("[cf] generator counts:", dict(sorted(gen_count.items(), key=lambda x:-x[1])))
    print(f"[cf] manifest now {len(df)} rows")
    print("\n[cf] gate problems:", M.validate(df) or "NONE ✓")


if __name__ == "__main__":
    main()
