"""Phase 3 / M1a — frozen CLIP + linear probe (UnivFD-style).

Fits a logistic-regression probe on top of FROZEN ViT-L-14 embeddings using the
`member_train` split (Pool A). Evaluates on an in-distribution held-out slice of
member_train (stratified val split) — this is the M1a ✓ gate.

  ✓ Gate: M1a ≥ ~90% accuracy on the in-distribution held-out sample.
          ~60% => preprocessing/feature path is broken; fix before proceeding.

Pipeline = L2-Normalizer -> LogisticRegression (CLIP-feature convention). The
fitted pipeline is saved to models/ for reuse as a member (Phase 5 combiner) and
as the M1a baseline in the M1a-vs-M1b ablation (Phase 6).

Usage:
  ./.venv/bin/python scripts/train_m1a.py            # uses cached embeddings
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, embeddings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--C", type=float, default=1.0, help="LogReg inverse reg strength")
    args = ap.parse_args()

    config.set_seed()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    import joblib

    df = pd.read_csv(config.MANIFEST_CSV)
    train_rows = df[df["split"] == config.SPLIT_MEMBER_TRAIN].reset_index(drop=True)
    cache = embeddings.load_cache()
    X, missing = embeddings.matrix_for(cache, train_rows["image_id"])
    if missing:
        print(f"[m1a] ERROR: {len(missing)} member_train embeddings missing from cache "
              f"(run scripts/extract_clip_embeddings.py first). e.g. {missing[:3]}")
        return 1
    y = train_rows["label"].to_numpy(dtype=int)
    print(f"[m1a] member_train: {len(y)} imgs | real={int((y==0).sum())} "
          f"fake={int((y==1).sum())} | dim={X.shape[1]}")

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=args.val_frac, stratify=y, random_state=config.SEED)
    print(f"[m1a] train={len(ytr)}  val(in-distribution held-out)={len(yva)}")

    clf = make_pipeline(
        Normalizer(norm="l2"),
        LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced"),
    )
    clf.fit(Xtr, ytr)

    p_va = clf.predict_proba(Xva)[:, 1]
    pred = (p_va >= 0.5).astype(int)
    acc = accuracy_score(yva, pred)
    auc = roc_auc_score(yva, p_va)
    acc_real = accuracy_score(yva[yva == 0], pred[yva == 0])
    acc_fake = accuracy_score(yva[yva == 1], pred[yva == 1])

    print(f"\n[m1a] === in-distribution held-out (val) ===")
    print(f"[m1a] accuracy : {acc:.4f}")
    print(f"[m1a] AUROC    : {auc:.4f}")
    print(f"[m1a] acc real : {acc_real:.4f}   acc fake : {acc_fake:.4f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "m1a_clip_linear.joblib"
    joblib.dump(clf, model_path)
    metrics = {
        "member_train_n": int(len(y)), "val_n": int(len(yva)),
        "val_frac": args.val_frac, "C": args.C,
        "accuracy": float(acc), "auroc": float(auc),
        "acc_real": float(acc_real), "acc_fake": float(acc_fake),
    }
    (config.RESULTS_DIR / "m1a_indist.json").write_text(json.dumps(metrics, indent=2))
    print(f"[m1a] saved probe  -> {model_path}")
    print(f"[m1a] saved metrics-> {config.RESULTS_DIR / 'm1a_indist.json'}")

    ok = acc >= 0.90
    print("\n[m1a] GATE:", "PASS ✓" if ok else "FAIL ✗",
          f"(accuracy {acc:.3f} vs >=0.90 target)")
    if not ok and acc < 0.70:
        print("[m1a] WARNING: accuracy near chance — check preprocessing/feature path.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
