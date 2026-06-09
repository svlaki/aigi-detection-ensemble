"""Ensemble size ablation — accuracy curves for 1, 2, and 3-member combiners.

Refits the LogReg combiner on Pool B using subsets of members:
  - M1 only
  - M1 + M2
  - M1 + M3
  - M2 + M3
  - M1 + M2 + M3 (full)
Shows marginal value of each member and whether the combiner helps at all
stages. Uses the same combiner feature template as Phase 5.

Outputs:
  results/ensemble_ablation.csv
  figures/fig7_ensemble_ablation.png

Usage:
  python scripts/ensemble_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cache_path = config.CACHE_DIR / "member_outputs.parquet"
    if not cache_path.exists():
        print(f"[ens-abl] ERROR: {cache_path} not found.")
        print("[ens-abl] Run: python scripts/build_member_outputs.py")
        return 1

    config.set_seed()
    df = pd.read_parquet(cache_path)
    B = df[df["split"] == config.SPLIT_COMBINER_FIT].reset_index(drop=True)
    E = df[df["split"] == config.SPLIT_EVAL].reset_index(drop=True)
    yB, yE = B["label"].to_numpy(int), E["label"].to_numpy(int)

    member_cols = {
        "M1": ("p1", "logit1"),
        "M2": ("p2", "logit2"),
        "M3": ("p3", "logit3"),
    }

    # define subsets to test
    subsets = [
        ("M1 only", ["M1"]),
        ("M2 only", ["M2"]),
        ("M3 only", ["M3"]),
        ("M1+M2", ["M1", "M2"]),
        ("M1+M3", ["M1", "M3"]),
        ("M2+M3", ["M2", "M3"]),
        ("M1+M2+M3", ["M1", "M2", "M3"]),
    ]

    def build_features(frame, members_list):
        """Build combiner features for a subset of members."""
        # calibrate each member on Pool B
        cals = {}
        for m in members_list:
            _, lcol = member_cols[m]
            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(B[[lcol]].to_numpy(), yB)
            cals[m] = (
                lr.predict_proba(frame[[lcol]].to_numpy())[:, 1],
                lr.decision_function(frame[[lcol]].to_numpy()),
            )
        # features: calibrated probs + logits + pairwise diffs
        parts = []
        names = list(members_list)
        for m in names:
            parts.append(cals[m][0])  # calibrated prob
            parts.append(cals[m][1])  # calibrated logit
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                parts.append(np.abs(cals[names[i]][0] - cals[names[j]][0]))
        return np.column_stack(parts)

    rows = []
    for label, members_list in subsets:
        FB = build_features(B, members_list)
        FE = build_features(E, members_list)

        comb = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
        )
        comb.fit(FB, yB)

        for split_name, F, y in [("combiner_fit", FB, yB), ("eval", FE, yE)]:
            sc = comb.predict_proba(F)[:, 1]
            pred = (sc >= 0.5).astype(int)
            rows.append({
                "config": label,
                "n_members": len(members_list),
                "split": split_name,
                "auroc": float(roc_auc_score(y, sc)),
                "acc": float(accuracy_score(y, pred)),
                "real_acc": float(accuracy_score(y[y == 0], pred[y == 0])),
                "fake_acc": float(accuracy_score(y[y == 1], pred[y == 1])),
            })

    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS_DIR / "ensemble_ablation.csv", index=False)

    eval_res = res[res["split"] == "eval"].reset_index(drop=True)
    print("\n[ens-abl] === Ensemble ablation (eval set) ===")
    print(eval_res[["config", "n_members", "auroc", "acc", "real_acc", "fake_acc"]]
          .to_string(index=False, float_format="%.3f"))

    # figure
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(eval_res))
    width = 0.3
    ax.bar(x - width, eval_res["acc"], width, label="Accuracy", color="tab:blue")
    ax.bar(x, eval_res["auroc"], width, label="AUROC", color="tab:orange")
    ax.bar(x + width, eval_res["real_acc"], width / 2, label="Real acc", color="tab:green", alpha=0.7)
    ax.bar(x + width + width / 2, eval_res["fake_acc"], width / 2, label="Fake acc", color="tab:red", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(eval_res["config"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    ax.legend(fontsize=9)
    ax.set_title("Figure 7 — Ensemble size ablation: combiner accuracy by member subset")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / "fig7_ensemble_ablation.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n[ens-abl] saved -> {out_path}, results/ensemble_ablation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
