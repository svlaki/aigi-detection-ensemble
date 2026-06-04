"""Phase 1 download + normalize + manifest, streaming & resumable.

Pulls a tiny balanced subset per source (counts in config), normalizes every
image (JPEG q95 + 256 center-crop), and appends rows to manifests/manifest.csv.
Resumable: skips image_ids already present. Run per-source via --only.

SOURCE column mappings are finalized from scripts/probe_sources.py output.
"""
from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src import manifest as M
from src.normalize import save_normalized


@dataclass
class Source:
    key: str                       # short id used in paths + image_id prefix
    hf_name: str
    split: str
    image_col: str                 # PIL-image column OR raw-bytes column
    source_dataset: str            # manifest source_dataset tag
    pool: str                      # A/B/C
    n: int                         # images to keep PER label bucket
    label: int | None = None       # fixed label, or None -> use label_col
    label_col: str | None = None   # per-row label column
    generator_name: str | None = None  # fixed token, or None -> use generator_col
    generator_col: str | None = None   # per-row generator column (fakes)
    config_name: str | None = None
    hf_kwargs: dict = field(default_factory=dict)
    balanced_labels: tuple = (0, 1)  # which labels to balance when label_col set


def _to_pil(val) -> Image.Image | None:
    if val is None:
        return None
    if isinstance(val, Image.Image):
        return val
    if isinstance(val, (bytes, bytearray)):
        return Image.open(io.BytesIO(val))
    if isinstance(val, dict) and "bytes" in val and val["bytes"]:
        return Image.open(io.BytesIO(val["bytes"]))
    return None


def _row_label(src: Source, ex) -> int | None:
    return src.label if src.label is not None else ex.get(src.label_col)


def _row_generator(src: Source, ex, label: int) -> str:
    if label == 0:
        return "real"
    if src.generator_name is not None:
        return src.generator_name
    g = ex.get(src.generator_col, "")
    return str(g).strip().lower().replace(" ", "_") or "unknown"


def run_source(src: Source, existing_ids: set[str]) -> list[dict]:
    out_dir = config.NORM_DIR / src.key
    rows, seen = [], 0
    # per-label keep counts (so CF stays balanced); single-class sources use one bucket
    targets = {l: src.n for l in (src.balanced_labels if src.label_col else
                                  (src.label if src.label is not None else 1,))}
    kept = {l: 0 for l in targets}
    print(f"[dl] {src.key}: {src.hf_name} split={src.split} targets={targets}")
    ds = load_dataset(src.hf_name, name=src.config_name, split=src.split,
                      streaming=True, **src.hf_kwargs)
    it = iter(ds)
    while any(kept[l] < targets[l] for l in targets):
        try:
            ex = next(it)
        except StopIteration:
            print(f"[dl] {src.key}: exhausted at {kept}")
            break
        except Exception as e:
            print(f"[dl] {src.key}: stream error ({repr(e)[:80]}); stop at {kept}")
            break
        seen += 1
        label = _row_label(src, ex)
        if label not in targets or kept[label] >= targets[label]:
            continue
        img = _to_pil(ex.get(src.image_col))
        if img is None:
            continue
        idx = kept[label]
        image_id = f"{src.key}_{label}_{idx:06d}"
        if image_id in existing_ids:
            kept[label] += 1
            continue
        try:
            out_path = save_normalized(img, out_dir / image_id)
        except Exception as e:
            print(f"[dl] {src.key}: skip bad image ({repr(e)[:60]})")
            continue
        rows.append({
            "image_id": image_id,
            "path": str(out_path.relative_to(config.ROOT)),
            "label": int(label),
            "source_dataset": src.source_dataset,
            "generator_name": _row_generator(src, ex, label),
            "pool": src.pool,
        })
        kept[label] += 1
        if sum(kept.values()) % 200 == 0:
            print(f"[dl] {src.key}: {kept} (scanned {seen})")
    print(f"[dl] {src.key}: done {kept} (scanned {seen})")
    return rows


def load_manifest() -> pd.DataFrame:
    if config.MANIFEST_CSV.exists():
        return pd.read_csv(config.MANIFEST_CSV)
    return pd.DataFrame(columns=config.MANIFEST_COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of source keys to run")
    ap.add_argument("--limit", type=int, default=None, help="cap n per source (smoke test)")
    args = ap.parse_args()
    config.ensure_dirs()
    config.set_seed()

    sources = build_sources()
    if args.only:
        sources = [s for s in sources if s.key in args.only]
        print("[dl] running only:", [s.key for s in sources])
    if args.limit:
        for s in sources:
            s.n = min(s.n, args.limit)
        print(f"[dl] limit per source -> {args.limit}")

    df = load_manifest()
    existing = set(df["image_id"]) if len(df) else set()
    for src in sources:
        rows = run_source(src, existing)
        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            df.to_csv(config.MANIFEST_CSV, index=False)  # checkpoint after each source
            existing |= {r["image_id"] for r in rows}
            print(f"[dl] manifest now {len(df)} rows -> {config.MANIFEST_CSV}")

    print("\n[dl] ===== manifest summary =====")
    print(M.summary(df))
    problems = M.validate(df)
    print("\n[dl] gate problems:", problems if problems else "NONE ✓")


def build_sources() -> list[Source]:
    """Finalized from probe output — see scripts/probe_sources.py / notes/data_sources.md."""
    FAKE_PER_GEN = 500          # GenImage fakes per generator (3 per pool -> 1500/pool)
    REALS_PER_POOL = 1500       # ImageNet reals per pool (balances the 1500 fakes)
    CF_PER_CLASS = config.CF_PER_CLASS          # 1250 each label
    MOD_PER_GEN = config.MODERN_PER_GENERATOR   # 750 each modern generator
    MOD_REALS_EACH = 1125       # FFHQ + COCO -> 2250 reals ~ balances 2250 modern fakes

    s: list[Source] = []

    # --- Pool A: GenImage fakes {adm,biggan,glide} + ImageNet reals (train) ---
    for gen, hf in [("adm", "bitmind/GenImage_ADM"),
                    ("biggan", "bitmind/GenImage_BigGAN"),
                    ("glide", "bitmind/GenImage_glide")]:
        s.append(Source(key=f"genimage_{gen}", hf_name=hf, split="train",
                        image_col="image", label=1, generator_name=gen,
                        source_dataset="genimage", pool="A", n=FAKE_PER_GEN))
    s.append(Source(key="imagenet_real_A", hf_name="evanarlian/imagenet_1k_resized_256",
                    split="train", image_col="image", label=0, generator_name="real",
                    source_dataset="imagenet", pool="A", n=REALS_PER_POOL))

    # --- Pool B: GenImage fakes {midjourney,wukong,vqdm} + ImageNet reals (val) ---
    for gen, hf in [("midjourney", "bitmind/GenImage_MidJourney"),
                    ("wukong", "bitmind/GenImage_wukong"),
                    ("vqdm", "bitmind/GenImage_VQDM")]:
        s.append(Source(key=f"genimage_{gen}", hf_name=hf, split="train",
                        image_col="image", label=1, generator_name=gen,
                        source_dataset="genimage", pool="B", n=FAKE_PER_GEN))
    s.append(Source(key="imagenet_real_B", hf_name="evanarlian/imagenet_1k_resized_256",
                    split="val", image_col="image", label=0, generator_name="real",
                    source_dataset="imagenet", pool="B", n=REALS_PER_POOL))

    # --- Pool C: Community Forensics (terminal in-the-wild), real+fake balanced ---
    s.append(Source(key="cf_eval", hf_name="OwensLab/CommunityForensics-Eval",
                    split="CompEval", image_col="image_data",
                    label=None, label_col="label",
                    generator_name=None, generator_col="model_name",
                    source_dataset="community_forensics", pool="C", n=CF_PER_CLASS))

    # --- Pool C: self-collected modern generators (fakes) ---
    for gen, hf in [("flux", "LukasT9/Flux-1-Dev-Images-1k"),
                    ("sd35", "momodawoud/sd3.5_generated_examples_hard_medium"),
                    # davidmunechika/midjourney-images had only 8 imgs -> swapped
                    ("midjourney_modern", "ehristoforu/midjourney-images")]:
        s.append(Source(key=f"modern_{gen}", hf_name=hf, split="train",
                        image_col="image", label=1, generator_name=gen,
                        source_dataset="modern_self", pool="C", n=MOD_PER_GEN))

    # --- Pool C: modern-slice reals (FFHQ faces + COCO scenes) ---
    s.append(Source(key="modern_real_ffhq", hf_name="merkol/ffhq-256", split="train",
                    image_col="image", label=0, generator_name="real",
                    source_dataset="modern_self", pool="C", n=MOD_REALS_EACH))
    s.append(Source(key="modern_real_coco", hf_name="detection-datasets/coco", split="val",
                    image_col="image", label=0, generator_name="real",
                    source_dataset="modern_self", pool="C", n=MOD_REALS_EACH))
    return s


if __name__ == "__main__":
    main()
