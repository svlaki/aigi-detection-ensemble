"""Robustness evaluation — M1 + M2 re-extraction on perturbed eval images.

Applies 14 perturbation conditions to the eval split, re-extracts CLIP and
spectral features (bypassing all caches), and scores with frozen probes.
Outputs per-tag parquets for later aggregation with D3QE logits.

Usage:
  python scripts/robustness_eval.py                    # all perturbations
  python scripts/robustness_eval.py --tags jpeg_q75,blur_s10  # subset
  python scripts/robustness_eval.py --keep-tmp         # don't delete temp images
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, embeddings, spectral

# ---------------------------------------------------------------------------
# Perturbation definitions
# ---------------------------------------------------------------------------

ALL_TAGS = [
    "jpeg_q90", "jpeg_q75", "jpeg_q50", "jpeg_q30",
    "blur_s05", "blur_s10", "blur_s20", "blur_s30",
    "noise_s2", "noise_s5", "noise_s10", "noise_s20",
    "resize_128",
    "social_media",
]

TAG_LABELS = {
    "clean": "Clean (baseline)",
    "jpeg_q90": "JPEG q=90",
    "jpeg_q75": "JPEG q=75",
    "jpeg_q50": "JPEG q=50",
    "jpeg_q30": "JPEG q=30",
    "blur_s05": "Blur σ=0.5",
    "blur_s10": "Blur σ=1.0",
    "blur_s20": "Blur σ=2.0",
    "blur_s30": "Blur σ=3.0",
    "noise_s2": "Noise σ=2",
    "noise_s5": "Noise σ=5",
    "noise_s10": "Noise σ=10",
    "noise_s20": "Noise σ=20",
    "resize_128": "Resize 128→256",
    "social_media": "Social Media Sim",
}


def apply_perturbation(img: Image.Image, tag: str, rng: np.random.Generator) -> Image.Image:
    """Apply the named perturbation to a PIL RGB image (256x256)."""
    if tag.startswith("jpeg_q"):
        quality = int(tag.split("q")[1])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    if tag.startswith("blur_s"):
        # Tag encodes sigma * 10 (e.g., blur_s05 = sigma 0.5)
        sigma = int(tag.removeprefix("blur_s")) / 10.0
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    if tag.startswith("noise_s"):
        sigma = int(tag.removeprefix("noise_s"))
        arr = np.asarray(img, dtype=np.float32)
        noisy = arr + rng.normal(0, sigma, arr.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, "RGB")

    if tag == "resize_128":
        small = img.resize((128, 128), Image.BICUBIC)
        return small.resize((256, 256), Image.BICUBIC)

    if tag == "social_media":
        # JPEG q75 + resize 128→256
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        jpg = Image.open(buf).convert("RGB")
        small = jpg.resize((128, 128), Image.BICUBIC)
        return small.resize((256, 256), Image.BICUBIC)

    raise ValueError(f"Unknown perturbation tag: {tag}")


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_m1(paths: list[Path], clip_model, clip_preprocess, device: str) -> np.ndarray:
    """Re-extract CLIP embeddings from (perturbed) image paths."""
    return embeddings.embed_paths(paths, clip_model, clip_preprocess, device, batch_size=32)


def extract_m2(paths: list[Path]) -> np.ndarray:
    """Re-extract spectral features from (perturbed) image paths."""
    feats = []
    for p in paths:
        feats.append(spectral.features_from_image(p))
    return np.stack(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Robustness eval (M1+M2)")
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma-separated perturbation tags (default: all)")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Don't delete temp perturbed images")
    args = parser.parse_args()

    config.set_seed()
    rng = np.random.default_rng(config.SEED)

    tags = args.tags.split(",") if args.tags else ALL_TAGS
    for t in tags:
        if t not in ALL_TAGS:
            print(f"[robustness] ERROR: unknown tag '{t}'. Valid: {ALL_TAGS}")
            return 1

    # Load manifest, filter to eval split
    df = pd.read_csv(config.MANIFEST_CSV)
    eval_df = df[df["split"] == config.SPLIT_EVAL].reset_index(drop=True)
    print(f"[robustness] eval split: {len(eval_df)} images")

    # Load frozen models
    import joblib
    m1_probe = joblib.load(config.MODELS_DIR / "m1a_clip_linear.joblib")
    m2_probe = joblib.load(config.MODELS_DIR / "m2_spectral.joblib")

    # Load CLIP model
    print("[robustness] loading CLIP model...")
    clip_model, clip_preprocess, device = embeddings.load_clip()
    print(f"[robustness] CLIP loaded on {device}")

    tmp_root = config.CACHE_DIR / "robustness_tmp"
    out_dir = config.CACHE_DIR / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag in tags:
        out_path = out_dir / f"m12_{tag}.parquet"
        if out_path.exists():
            print(f"[robustness] {tag}: already done, skipping")
            continue

        t0 = time.time()
        print(f"\n[robustness] === {tag} ({TAG_LABELS.get(tag, tag)}) ===")

        # Apply perturbations and save to temp dir
        tag_dir = tmp_root / tag
        tag_dir.mkdir(parents=True, exist_ok=True)

        perturbed_paths = []
        for _, row in eval_df.iterrows():
            src_path = config.ROOT / row["path"]
            dst_path = tag_dir / f"{row['image_id']}.jpg"
            if not dst_path.exists():
                img = Image.open(src_path).convert("RGB")
                perturbed = apply_perturbation(img, tag, rng)
                perturbed.save(dst_path, format="JPEG", quality=95)
            perturbed_paths.append(dst_path)

        print(f"  perturbed {len(perturbed_paths)} images ({time.time() - t0:.1f}s)")

        # M1: CLIP embeddings -> probe
        t1 = time.time()
        X_clip = extract_m1(perturbed_paths, clip_model, clip_preprocess, device)
        logit1 = m1_probe.decision_function(X_clip)
        p1 = m1_probe.predict_proba(X_clip)[:, 1]
        print(f"  M1 extracted ({time.time() - t1:.1f}s)")

        # M2: Spectral features -> probe
        t2 = time.time()
        X_spec = extract_m2(perturbed_paths)
        logit2 = m2_probe.decision_function(X_spec)
        p2 = m2_probe.predict_proba(X_spec)[:, 1]
        print(f"  M2 extracted ({time.time() - t2:.1f}s)")

        # Save per-tag parquet
        result = pd.DataFrame({
            "image_id": eval_df["image_id"].astype(str).values,
            "label": eval_df["label"].values,
            "generator_name": eval_df["generator_name"].values,
            "source_dataset": eval_df["source_dataset"].values,
            "p1": p1, "logit1": logit1,
            "p2": p2, "logit2": logit2,
        })
        result.to_parquet(out_path, index=False)
        print(f"  saved -> {out_path} ({time.time() - t0:.1f}s total)")

        # Cleanup temp images for this tag
        if not args.keep_tmp:
            shutil.rmtree(tag_dir, ignore_errors=True)

    # Cleanup temp root if empty
    if not args.keep_tmp and tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\n[robustness] M1+M2 extraction complete.")
    print(f"  Outputs in {out_dir}/")
    print("  Next: run D3QE on Modal, then run scripts/robustness_score.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
