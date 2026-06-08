"""Central config: paths, seed, device, and the generator-disjoint pool lists.

Single source of truth for the whole project. Import as `from src import config`
or `import config` (when run from inside src/). Keep this file dependency-light
(only stdlib + torch for device detection) so it imports fast everywhere.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all absolute, derived from repo root = parent of this file's dir)
# --------------------------------------------------------------------------- #
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # downloaded-as-is (pre-normalization)
NORM_DIR = DATA_DIR / "normalized"   # JPEG q95 + center-crop (Grommelt control)
CACHE_DIR = ROOT / "cache"           # per-image member features / logits
MODELS_DIR = ROOT / "models"         # trained heads, LoRA adapters, checkpoints
RESULTS_DIR = ROOT / "results"       # metrics tables (csv/json)
FIGURES_DIR = ROOT / "figures"
MANIFEST_DIR = ROOT / "manifests"
EXTERNAL_DIR = ROOT / "external"     # cloned third-party repos (D3QE, etc.)
NOTES_DIR = ROOT / "notes"           # provenance / version pins

MANIFEST_CSV = MANIFEST_DIR / "manifest.csv"

# columns every manifest row must have (Phase 1.6)
MANIFEST_COLUMNS = [
    "image_id",        # stable unique id
    "path",            # path to the NORMALIZED image
    "label",           # 0 = real, 1 = fake
    "source_dataset",  # genimage | community_forensics | modern_self | coco | ffhq | ...
    "generator_name",  # e.g. sd14, adm, glide, biggan, flux, sd35, midjourney; "real" for reals
    "pool",            # A | B | C  (origin pool)
    "split",           # member_train | combiner_fit | eval | lora_train  (training/eval ROLE)
]

# --- Training/eval roles (the `split` column is authoritative for who-uses-what) ---
# pool encodes data ORIGIN; split encodes ROLE. The eval set is split==EVAL, NOT raw
# pool=="C", because the modern slice (origin pool C) is partitioned into a LoRA-train
# part (never evaluated) and a held-out modern_test part (which IS in the eval set).
SPLIT_MEMBER_TRAIN = "member_train"   # Pool A — train frozen M1 probe + M2
SPLIT_COMBINER_FIT = "combiner_fit"   # Pool B — fit combiner + calibration
SPLIT_EVAL = "eval"                   # held-out eval = CF + modern_test (the "Pool C" of figures)
SPLIT_LORA_TRAIN = "lora_train"       # modern slice, LoRA fine-tune ONLY (never evaluated)

# Modern-slice partition (image-disjoint, stratified by generator/source). Of each
# modern fake generator's 750: this many go to lora_train, the rest to modern_test.
LORA_TRAIN_FAKE_PER_GEN = 1000        # -> 500 per generator held out as modern_test
LORA_TRAIN_REAL_PER_SOURCE = 1500     # of 2250 ffhq/coco each -> 750 held out as modern_test

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 1337

def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy, and torch (incl. CUDA/MPS) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# --------------------------------------------------------------------------- #
# Device (portable: cuda on the rented GPU, mps on this Mac, cpu fallback)
# --------------------------------------------------------------------------- #
def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

# --------------------------------------------------------------------------- #
# Model / preprocessing config
# --------------------------------------------------------------------------- #
# M1 backbone (UnivFD-style CLIP-linear). ViT-L/14 -> 768-dim image embedding.
# Use the *-quickgelu* variant: the OpenAI checkpoint was trained with QuickGELU,
# so this matches the original activation exactly (plain "ViT-L-14" warns about a
# QuickGELU mismatch and would use the wrong nonlinearity).
CLIP_MODEL = "ViT-L-14-quickgelu"
CLIP_PRETRAINED = "openai"
CLIP_EMBED_DIM = 768

# Normalization (Phase 1.5 — CRITICAL Grommelt confound control).
# Recompress EVERY image (real + fake, all sets) to JPEG q95 and center-crop
# to a common resolution before any feature extraction.
NORM_JPEG_QUALITY = 95
NORM_CROP_SIZE = 256  # center-crop / resize target (square)

# --------------------------------------------------------------------------- #
# Generator-disjoint pools (Phase 2). PLACEHOLDERS — finalize once the manifest
# reveals which generators actually downloaded. Invariant: A ∩ B == ∅ on
# generator_name; Pool C is held out and never touched in training.
# Names are lowercased canonical tokens; map raw dataset labels -> these.
# --------------------------------------------------------------------------- #
# Locked 3v3 (no SD1.4 mirror under bitmind -> use available GenImage generators).
POOL_A_GENERATORS = ["adm", "biggan", "glide"]                  # train members
POOL_B_GENERATORS = ["midjourney", "wukong", "vqdm"]            # fit combiner + calibrate
POOL_C_GENERATORS = [                                           # final eval ONLY
    # Community Forensics test generators are many/unknown -> tagged by source.
    "flux", "sd35", "midjourney_modern",
]

# Datasets that, regardless of generator, belong entirely to Pool C (held-out).
POOL_C_SOURCES = ["community_forensics", "modern_self"]


def assert_no_pool_leakage() -> None:
    """Phase 2 ✓ Gate: Pool A and Pool B generators must be disjoint."""
    overlap = set(POOL_A_GENERATORS) & set(POOL_B_GENERATORS)
    assert not overlap, f"Pool A/B generator leakage: {overlap}"


# --------------------------------------------------------------------------- #
# Subsampling targets (keep TINY — 8GB RAM / 44GB disk on the dev box)
# --------------------------------------------------------------------------- #
GENIMAGE_PER_CLASS = 3000          # ~6k balanced total
CF_PER_CLASS = 2500                # ~5k balanced total
MODERN_PER_GENERATOR = 1500        # ~1500 per modern generator


def ensure_dirs() -> None:
    for d in (RAW_DIR, NORM_DIR, CACHE_DIR, MODELS_DIR, RESULTS_DIR,
              FIGURES_DIR, MANIFEST_DIR, EXTERNAL_DIR, NOTES_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    set_seed()
    print(f"repo root      : {ROOT}")
    print(f"device         : {get_device()}")
    print(f"seed           : {SEED}")
    print(f"CLIP backbone  : {CLIP_MODEL}/{CLIP_PRETRAINED} (dim={CLIP_EMBED_DIM})")
    print(f"norm           : JPEG q{NORM_JPEG_QUALITY}, crop {NORM_CROP_SIZE}px")
    assert_no_pool_leakage()
    print(f"Pool A (train) : {POOL_A_GENERATORS}")
    print(f"Pool B (fit)   : {POOL_B_GENERATORS}")
    print(f"Pool C (eval)  : sources={POOL_C_SOURCES} + {POOL_C_GENERATORS}")
    print("pool leakage check: OK (A ∩ B = ∅)")
