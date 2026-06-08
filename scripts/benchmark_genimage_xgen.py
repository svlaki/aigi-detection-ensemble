"""Tier-2 external validation — GenImage cross-generator reproduction (positive control).

Our M1 is the UniversalFakeDetect method (CLIP ViT-L/14 + linear probe; Ojha et al.
CVPR 2023). The established finding (UnivFD; GenImage, Zhu et al. NeurIPS 2023) is
that CLIP-linear probes generalize STRONGLY across GAN/early-diffusion generators —
cross-generator AUROC typically in the high-0.8s–0.9s, far better than CNN detectors.

This reproduces that protocol on the standard GenImage generators (using our cached
CLIP embeddings — no download, no GPU): train the probe on ONE generator + ImageNet
reals, test on EACH generator, building a 6x6 AUROC matrix. If our matrix sits in the
published regime, our M1 implementation is validated against the literature; if the
modern-collapse pattern also appears, that too is consistent with the field.

Protocol (balanced, disjoint):
  - each generator's 500 fakes -> 250 train / 250 test (seeded)
  - 3000 ImageNet reals -> fixed 250 train-reals + 250 test-reals (disjoint)
  - cell (G_tr, G_te): probe = Normalizer(l2)+LogReg on G_tr train-fakes + train-reals;
    AUROC on G_te test-fakes + test-reals. Diagonal = in-generator (held-out images).

Outputs: results/benchmark_genimage_xgen.csv (matrix), .json (summary)

Usage:
  ./.venv/bin/python scripts/benchmark_genimage_xgen.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, embeddings

GENS = ["adm", "biggan", "glide", "midjourney", "wukong", "vqdm"]
PER_CELL = 250  # balanced fakes/reals per train and per test cell


def main() -> int:
    config.set_seed()
    rng = np.random.default_rng(config.SEED)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    df = pd.read_csv(config.MANIFEST_CSV)
    cache = embeddings.load_cache()

    # fakes per generator -> train/test halves
    fake_tr, fake_te = {}, {}
    for g in GENS:
        ids = df[(df.generator_name == g) & (df.label == 1)]["image_id"].tolist()
        rng.shuffle(ids)
        half = len(ids) // 2
        fake_tr[g], fake_te[g] = ids[:half], ids[half:]

    # ImageNet reals -> disjoint train/test pools -> fixed balanced subsets
    reals = df[(df.label == 0) & (df.split.isin([config.SPLIT_MEMBER_TRAIN,
                                                 config.SPLIT_COMBINER_FIT]))]["image_id"].tolist()
    rng.shuffle(reals)
    mid = len(reals) // 2
    real_tr = reals[:mid][:PER_CELL]
    real_te = reals[mid:][:PER_CELL]

    def mat(ids):
        X, miss = embeddings.matrix_for(cache, ids)
        assert not miss, f"{len(miss)} embeddings missing"
        return X

    Xreal_tr, Xreal_te = mat(real_tr), mat(real_te)

    M = np.zeros((len(GENS), len(GENS)))
    for i, gtr in enumerate(GENS):
        Xf_tr = mat(fake_tr[gtr][:PER_CELL])
        Xtr = np.vstack([Xf_tr, Xreal_tr])
        ytr = np.r_[np.ones(len(Xf_tr)), np.zeros(len(Xreal_tr))]
        clf = make_pipeline(Normalizer("l2"),
                            LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        clf.fit(Xtr, ytr)
        for j, gte in enumerate(GENS):
            Xf_te = mat(fake_te[gte][:PER_CELL])
            Xte = np.vstack([Xf_te, Xreal_te])
            yte = np.r_[np.ones(len(Xf_te)), np.zeros(len(Xreal_te))]
            M[i, j] = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

    mat_df = pd.DataFrame(M, index=[f"tr:{g}" for g in GENS],
                          columns=[f"te:{g}" for g in GENS])
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mat_df.to_csv(config.RESULTS_DIR / "benchmark_genimage_xgen.csv")

    diag = np.diag(M).mean()
    off = M[~np.eye(len(GENS), dtype=bool)].mean()
    summary = {"diagonal_in_generator_auroc_mean": round(float(diag), 3),
               "offdiagonal_cross_generator_auroc_mean": round(float(off), 3),
               "overall_auroc_mean": round(float(M.mean()), 3),
               "per_cell_n": 2 * PER_CELL, "generators": GENS}
    (config.RESULTS_DIR / "benchmark_genimage_xgen.json").write_text(json.dumps(summary, indent=2))

    print("=== GenImage cross-generator AUROC (rows=train, cols=test) ===\n")
    print(mat_df.round(3).to_string())
    print(f"\nin-generator (diagonal) mean AUROC  : {diag:.3f}")
    print(f"cross-generator (off-diag) mean AUROC: {off:.3f}")
    print("\nPublished regime (UnivFD Ojha+ CVPR'23; GenImage Zhu+ NeurIPS'23):")
    print("  CLIP-linear probes generalize strongly across GAN/early-diffusion")
    print("  generators — cross-generator AUROC typically high-0.8s to 0.9s.")
    in_regime = off >= 0.80
    print(f"\nVERDICT: cross-generator {off:.3f} "
          + ("IN published regime ✓ — M1 reproduces the CLIP-detector finding"
             if in_regime else
             "BELOW typical regime — note subsampling/JPEG-normalization effects honestly"))
    print("saved -> results/benchmark_genimage_xgen.{csv,json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
