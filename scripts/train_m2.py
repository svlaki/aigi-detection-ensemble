"""Phase 3 / M2 — spectral logistic-regression detector.

Fits a LogReg classifier on cached FFT features (src/spectral.py) using the
`member_train` split (Pool A), evaluated on the SAME stratified in-distribution
held-out slice convention as M1a (seed-matched), so the two members are compared
on equal footing.

  ✓ Gate: M2 clearly beats chance in-distribution (>=~75% accuracy). Lower bar
          than M1a on purpose — spectral detection is weaker (esp. post-JPEG).
          Its value is decorrelated errors (Phase 4), not raw accuracy.

StandardScaler -> LogisticRegression. Saved to models/ for the combiner (Phase 5).

Usage:
  ./.venv/bin/python scripts/train_m2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, spectral


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    config.set_seed()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    df = pd.read_csv(config.MANIFEST_CSV)
    train_rows = df[df["split"] == config.SPLIT_MEMBER_TRAIN].reset_index(drop=True)
    cache = spectral.load_cache()
    X, missing = spectral.matrix_for(cache, train_rows["image_id"])
    if missing:
        print(f"[m2] ERROR: {len(missing)} member_train features missing from cache "
              f"(run scripts/extract_spectral_features.py first). e.g. {missing[:3]}")
        return 1
    y = train_rows["label"].to_numpy(dtype=int)
    print(f"[m2] member_train: {len(y)} imgs | real={int((y==0).sum())} "
          f"fake={int((y==1).sum())} | dim={X.shape[1]}")

    # seed-matched split => same val images as M1a (fair head-to-head)
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=args.val_frac, stratify=y, random_state=config.SEED)
    print(f"[m2] train={len(ytr)}  val(in-distribution held-out)={len(yva)}")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced"),
    )
    clf.fit(Xtr, ytr)

    p_va = clf.predict_proba(Xva)[:, 1]
    pred = (p_va >= 0.5).astype(int)
    acc = accuracy_score(yva, pred)
    auc = roc_auc_score(yva, p_va)
    acc_real = accuracy_score(yva[yva == 0], pred[yva == 0])
    acc_fake = accuracy_score(yva[yva == 1], pred[yva == 1])

    print(f"\n[m2] === in-distribution held-out (val) ===")
    print(f"[m2] accuracy : {acc:.4f}")
    print(f"[m2] AUROC    : {auc:.4f}")
    print(f"[m2] acc real : {acc_real:.4f}   acc fake : {acc_fake:.4f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "m2_spectral.joblib"
    joblib.dump(clf, model_path)
    metrics = {
        "member_train_n": int(len(y)), "val_n": int(len(yva)),
        "val_frac": args.val_frac, "C": args.C,
        "accuracy": float(acc), "auroc": float(auc),
        "acc_real": float(acc_real), "acc_fake": float(acc_fake),
    }
    (config.RESULTS_DIR / "m2_indist.json").write_text(json.dumps(metrics, indent=2))
    print(f"[m2] saved model  -> {model_path}")
    print(f"[m2] saved metrics-> {config.RESULTS_DIR / 'm2_indist.json'}")

    ok = acc >= 0.75
    print("\n[m2] GATE:", "PASS ✓" if ok else "FAIL ✗",
          f"(accuracy {acc:.3f} vs >=0.75 target)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
