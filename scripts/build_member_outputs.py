"""Phase 3 (final) — assemble the per-image member-output table.

Produces ONE keyed table of every member's output for every image:
  image_id, label, split, pool, generator_name, source_dataset,
  p1, logit1,   # M1a  (frozen CLIP + linear probe)
  p2, logit2,   # M2   (spectral LogReg)
  p3, logit3    # M3   (D3QE pretrained; logit cached, p = sigmoid)

This is the direct input to Phase 4 (decorrelation matrix) and Phase 5 (combiner).

Production members: M1a and M2 are REFIT here on the FULL member_train split (the
80/20 split in train_m1a/train_m2 only measured the in-distribution gate; the
member used downstream should use all its training data). The refit probes
overwrite models/m1a_clip_linear.joblib and models/m2_spectral.joblib. Members
never see combiner_fit/eval/lora_train, so those outputs are out-of-sample (clean).

  ⚠ Provenance: member_train's OWN p1/p2 are IN-SAMPLE for M1/M2 (the member was
    trained on them). M3 is pretrained and never saw our data, so p3 is always
    out-of-sample. Headline metrics use combiner_fit (Pool B) and eval — both
    disjoint from member_train — so this does not contaminate results.

  ✓ Gate: row count == manifest, every member column present, no NaNs, ids aligned.

Usage:
  ./.venv/bin/python scripts/build_member_outputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, embeddings, spectral, d3qe


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def main() -> int:
    config.set_seed()
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler
    import joblib

    df = pd.read_csv(config.MANIFEST_CSV)
    ids = df["image_id"].astype(str).tolist()
    y = df["label"].to_numpy(int)

    clip_cache = embeddings.load_cache()
    spec_cache = spectral.load_cache()
    m3_cache = d3qe.load_cache()

    X_clip, miss1 = embeddings.matrix_for(clip_cache, ids)
    X_spec, miss2 = spectral.matrix_for(spec_cache, ids)
    lg3, miss3 = d3qe.vector_for(m3_cache, ids)
    for name, miss in [("CLIP", miss1), ("spectral", miss2), ("D3QE", miss3)]:
        if miss:
            print(f"[members] ERROR: {len(miss)} {name} features missing (run its extractor). "
                  f"e.g. {miss[:3]}")
            return 1

    # --- refit production members on the FULL member_train split ---
    train_mask = (df["split"] == config.SPLIT_MEMBER_TRAIN).to_numpy()
    print(f"[members] refit M1a/M2 on full member_train ({int(train_mask.sum())} imgs)")

    m1 = make_pipeline(Normalizer(norm="l2"),
                       LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    m1.fit(X_clip[train_mask], y[train_mask])

    m2 = make_pipeline(StandardScaler(),
                       LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    m2.fit(X_spec[train_mask], y[train_mask])

    joblib.dump(m1, config.MODELS_DIR / "m1a_clip_linear.joblib")
    joblib.dump(m2, config.MODELS_DIR / "m2_spectral.joblib")

    # --- member outputs for ALL images ---
    logit1 = m1.decision_function(X_clip)      # w·x + b == logit for LogReg
    p1 = m1.predict_proba(X_clip)[:, 1]
    logit2 = m2.decision_function(X_spec)
    p2 = m2.predict_proba(X_spec)[:, 1]
    logit3 = lg3
    p3 = _sigmoid(lg3)

    out = pd.DataFrame({
        "image_id": df["image_id"].astype(str),
        "label": y,
        "split": df["split"],
        "pool": df["pool"],
        "generator_name": df["generator_name"],
        "source_dataset": df["source_dataset"],
        "p1": p1, "logit1": logit1,
        "p2": p2, "logit2": logit2,
        "p3": p3, "logit3": logit3,
    })

    # --- gate ---
    member_cols = ["p1", "logit1", "p2", "logit2", "p3", "logit3"]
    n_ok = len(out) == len(df)
    no_nan = not out[member_cols].isna().any().any()
    aligned = (out["image_id"].tolist() == ids)
    print(f"[members] rows={len(out)} (manifest {len(df)})  no_nan={no_nan}  aligned={aligned}")

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.CACHE_DIR / "member_outputs.parquet"
    out.to_parquet(out_path, index=False)
    out.to_csv(config.CACHE_DIR / "member_outputs.csv", index=False)
    print(f"[members] saved -> {out_path} (+ .csv)")

    # quick per-member AUROC by split (orientation sanity)
    from sklearn.metrics import roc_auc_score
    print("\n[members] per-member AUROC by split:")
    for split in ["member_train", "combiner_fit", "eval"]:
        s = out[out["split"] == split]
        aucs = [roc_auc_score(s["label"], s[c]) for c in ("p1", "p2", "p3")]
        print(f"  {split:13s} M1={aucs[0]:.3f}  M2={aucs[1]:.3f}  M3={aucs[2]:.3f}")

    ok = n_ok and no_nan and aligned
    print("\n[members] GATE:", "PASS ✓" if ok else "FAIL ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
