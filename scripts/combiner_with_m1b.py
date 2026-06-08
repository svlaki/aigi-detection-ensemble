"""Phase 6 step 4 (optional) — fold the LoRA-adapted M1b into the combiner.

Re-runs the Phase 5 combiner pipeline TWICE on combiner_fit ∪ eval, identical except
for which "M1" member is used:
  - config "M1a": logit1 from member_outputs (frozen CLIP probe, Pool-A trained) — this
    reproduces the Phase 5 combiner.
  - config "M1b": M1's logit replaced by the LoRA-adapted M1b logits (cache/m1b_logits.npz,
    trained on lora_train modern slice).
Each config: Platt-calibrate all 3 members on Pool B, build the same combiner features,
fit a LogReg combiner on Pool B, evaluate on eval (all) and on the modern-generator
subset (source_dataset=='modern_self'), where M1b should help most.

Answers: does swapping the adapted member into the ensemble improve eval detection?

Outputs: results/phase6_combiner_fold.csv, results/phase6_combiner_fold.json

Usage:
  ./.venv/bin/python scripts/combiner_with_m1b.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


def metrics(y, score, thr=0.5):
    from sklearn.metrics import roc_auc_score, accuracy_score
    pred = (score >= thr).astype(int)
    return {
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "acc": float(accuracy_score(y, pred)),
        "real_acc": float(accuracy_score(y[y == 0], pred[y == 0])) if (y == 0).any() else float("nan"),
        "fake_acc": float(accuracy_score(y[y == 1], pred[y == 1])) if (y == 1).any() else float("nan"),
    }


def run_config(B, E, m1_logit_col, tag):
    """Platt-calibrate 3 members on Pool B, fit LogReg combiner, eval on E + modern subset."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    yB = B["label"].to_numpy(int)
    member_cols = [m1_logit_col, "logit2", "logit3"]

    def calibrate(frame):
        ps, zs = [], []
        for col in member_cols:
            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(B[[col]].to_numpy(), yB)
            ps.append(lr.predict_proba(frame[[col]].to_numpy())[:, 1])
            zs.append(lr.decision_function(frame[[col]].to_numpy()))
        return ps, zs

    def feats(frame):
        ps, zs = calibrate(frame)
        p1, p2, p3 = ps
        return np.column_stack([p1, p2, p3, zs[0], zs[1], zs[2],
                                np.abs(p1 - p2), np.abs(p1 - p3), np.abs(p2 - p3)])

    comb = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    comb.fit(feats(B), yB)

    rows = []
    modern = E[E["source_dataset"] == "modern_self"]
    for subset_name, frame in [("eval", E), ("eval_modern", modern)]:
        sc = comb.predict_proba(feats(frame))[:, 1]
        rows.append({"config": tag, "subset": subset_name,
                     **metrics(frame["label"].to_numpy(int), sc)})
    return rows


def main() -> int:
    config.set_seed()
    df = pd.read_parquet(config.CACHE_DIR / "member_outputs.parquet")

    # attach M1b logits
    z = np.load(config.CACHE_DIR / "m1b_logits.npz", allow_pickle=False)
    m1b = dict(zip([str(i) for i in z["image_id"]], z["logit"].astype(float)))
    df["logit1b"] = df["image_id"].astype(str).map(m1b)

    B = df[df["split"] == config.SPLIT_COMBINER_FIT].reset_index(drop=True)
    E = df[df["split"] == config.SPLIT_EVAL].reset_index(drop=True)
    miss = int(B["logit1b"].isna().sum() + E["logit1b"].isna().sum())
    if miss:
        print(f"[fold] ERROR: {miss} combiner_fit/eval rows missing M1b logits")
        return 1
    print(f"[fold] Pool B n={len(B)} | eval n={len(E)} | "
          f"eval_modern n={int((E['source_dataset']=='modern_self').sum())}")

    rows = run_config(B, E, "logit1", "combiner_M1a") \
        + run_config(B, E, "logit1b", "combiner_M1b")
    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS_DIR / "phase6_combiner_fold.csv", index=False)

    print("\n[fold] === combiner with M1a vs with M1b ===")
    piv = res.set_index(["config", "subset"])[["auroc", "acc", "real_acc", "fake_acc"]]
    print(piv.round(3).to_string())

    def get(cfg, sub, m): return float(res[(res.config == cfg) & (res.subset == sub)][m].iloc[0])
    summary = {}
    for sub in ("eval", "eval_modern"):
        summary[sub] = {
            "auroc_delta": round(get("combiner_M1b", sub, "auroc") - get("combiner_M1a", sub, "auroc"), 4),
            "acc_delta": round(get("combiner_M1b", sub, "acc") - get("combiner_M1a", sub, "acc"), 4),
        }
    (config.RESULTS_DIR / "phase6_combiner_fold.json").write_text(json.dumps(summary, indent=2))
    print("\n[fold] M1b - M1a combiner deltas:")
    for sub, d in summary.items():
        print(f"  {sub:12s} Δauroc={d['auroc_delta']:+.3f}  Δacc={d['acc_delta']:+.3f}")
    print("\n[fold] saved -> results/phase6_combiner_fold.{csv,json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
