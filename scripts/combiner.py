"""Phase 5 — calibration + learned combiner + baselines (the gradeable milestone).

Pipeline (all disjoint from eval):
  1. Per-member calibration on Pool B (combiner_fit): Platt scaling — a 1-D LogReg
     on each member's logit. Learns BOTH scale and threshold, so it captures the
     "high AUROC / low accuracy@0.5" free win (just rescaling can't move the
     boundary; Platt can).
  2. Learned combiner on Pool B. Features = [p1,p2,p3 (calibrated),
     logit margins (calibrated logits z1,z2,z3), pairwise agreement
     |pi-pj|]. Train LogReg AND a 2-layer MLP; report both.
  3. Baselines on calibrated outputs: mean-probability + majority vote, to
     isolate the *learned* combiner's added value.
  4. Evaluate members (raw + calibrated), baselines, combiners on eval AND Pool B,
     with PER-CLASS metrics always (real_acc / fake_acc), AUROC, acc@0.5,
     acc@calibrated-threshold, plus the in-distribution->OOD gap.

Outputs: results/phase5_results.csv, results/phase5_summary.json,
         models/{calib_m{1,2,3},combiner_logreg,combiner_mlp}.joblib,
         figures/fig3_results.png

Usage:
  ./.venv/bin/python scripts/combiner.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


def metrics(y: np.ndarray, score: np.ndarray, thr: float = 0.5) -> dict:
    from sklearn.metrics import roc_auc_score, accuracy_score
    pred = (score >= thr).astype(int)
    return {
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "acc": float(accuracy_score(y, pred)),
        "real_acc": float(accuracy_score(y[y == 0], pred[y == 0])) if (y == 0).any() else float("nan"),
        "fake_acc": float(accuracy_score(y[y == 1], pred[y == 1])) if (y == 1).any() else float("nan"),
    }


def main() -> int:
    config.set_seed()
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    df = pd.read_parquet(config.CACHE_DIR / "member_outputs.parquet")
    B = df[df["split"] == config.SPLIT_COMBINER_FIT].reset_index(drop=True)   # Pool B
    E = df[df["split"] == config.SPLIT_EVAL].reset_index(drop=True)           # held-out
    yB, yE = B["label"].to_numpy(int), E["label"].to_numpy(int)
    print(f"[combiner] Pool B (fit) n={len(B)} | eval n={len(E)}")

    SPLITS = {"combiner_fit": B, "eval": E}

    # ---- 1. per-member Platt calibration (fit on Pool B) ----
    calibrators, calib = {}, {s: {} for s in SPLITS}
    for m, lcol in [("M1", "logit1"), ("M2", "logit2"), ("M3", "logit3")]:
        lr = LogisticRegression(C=1e6, max_iter=1000)  # near-unregularized 1-D Platt
        lr.fit(B[[lcol]].to_numpy(), yB)
        calibrators[m] = lr
        for s, frame in SPLITS.items():
            xb = frame[[lcol]].to_numpy()
            calib[s][m] = (lr.predict_proba(xb)[:, 1], lr.decision_function(xb))
        joblib.dump(lr, config.MODELS_DIR / f"calib_{m.lower()}.joblib")

    def features(split: str, frame: pd.DataFrame) -> np.ndarray:
        p1, z1 = calib[split]["M1"]; p2, z2 = calib[split]["M2"]; p3, z3 = calib[split]["M3"]
        return np.column_stack([
            p1, p2, p3, z1, z2, z3,
            np.abs(p1 - p2), np.abs(p1 - p3), np.abs(p2 - p3),
        ])

    FB, FE = features("combiner_fit", B), features("eval", E)

    # ---- 2. learned combiners (fit on Pool B) ----
    comb_lr = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    comb_lr.fit(FB, yB)
    comb_mlp = make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000,
                                           early_stopping=True, random_state=config.SEED))
    comb_mlp.fit(FB, yB)
    joblib.dump(comb_lr, config.MODELS_DIR / "combiner_logreg.joblib")
    joblib.dump(comb_mlp, config.MODELS_DIR / "combiner_mlp.joblib")

    # ---- 3 & 4. score everything on both splits ----
    def scores(split, frame):
        p1, _ = calib[split]["M1"]; p2, _ = calib[split]["M2"]; p3, _ = calib[split]["M3"]
        F = features(split, frame)
        vote = ((p1 >= .5).astype(int) + (p2 >= .5).astype(int) + (p3 >= .5).astype(int)) / 3.0
        return {
            "M1_raw": frame["p1"].to_numpy(), "M2_raw": frame["p2"].to_numpy(),
            "M3_raw": frame["p3"].to_numpy(),
            "M1_cal": p1, "M2_cal": p2, "M3_cal": p3,
            "mean_prob": (p1 + p2 + p3) / 3.0, "majority_vote": vote,
            "combiner_logreg": comb_lr.predict_proba(F)[:, 1],
            "combiner_mlp": comb_mlp.predict_proba(F)[:, 1],
        }

    rows = []
    for split, frame, y in [("combiner_fit", B, yB), ("eval", E, yE)]:
        for name, sc in scores(split, frame).items():
            rows.append({"method": name, "split": split, **metrics(y, sc)})
    res = pd.DataFrame(rows)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.RESULTS_DIR / "phase5_results.csv", index=False)

    ev = res[res["split"] == "eval"].set_index("method")
    print("\n[combiner] === EVAL (held-out modern/CF) — per-class ===")
    print(ev[["auroc", "acc", "real_acc", "fake_acc"]].round(3).to_string())

    best_member = ev.loc[["M1_cal", "M2_cal", "M3_cal"], "acc"].idxmax()
    gap_lr = (res[(res.method == "combiner_logreg") & (res.split == "combiner_fit")]["acc"].iloc[0]
              - ev.loc["combiner_logreg", "acc"])
    summary = {
        "best_member_eval": best_member,
        "best_member_eval_acc": float(ev.loc[best_member, "acc"]),
        "best_member_eval_auroc": float(ev.loc[best_member, "auroc"]),
        "combiner_logreg_eval": ev.loc["combiner_logreg", ["auroc", "acc", "real_acc", "fake_acc"]].to_dict(),
        "combiner_mlp_eval": ev.loc["combiner_mlp", ["auroc", "acc", "real_acc", "fake_acc"]].to_dict(),
        "mean_prob_eval_acc": float(ev.loc["mean_prob", "acc"]),
        "majority_vote_eval_acc": float(ev.loc["majority_vote", "acc"]),
        "combiner_logreg_indist_minus_ood_acc_gap": float(gap_lr),
    }
    (config.RESULTS_DIR / "phase5_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[combiner] best single member (eval): {best_member} acc={ev.loc[best_member,'acc']:.3f}")
    print(f"[combiner] combiner LogReg eval acc={ev.loc['combiner_logreg','acc']:.3f} "
          f"auroc={ev.loc['combiner_logreg','auroc']:.3f}")
    print(f"[combiner] combiner MLP    eval acc={ev.loc['combiner_mlp','acc']:.3f} "
          f"auroc={ev.loc['combiner_mlp','auroc']:.3f}")
    print(f"[combiner] baselines eval acc: mean={ev.loc['mean_prob','acc']:.3f} "
          f"majority={ev.loc['majority_vote','acc']:.3f}")

    # ---- figure: eval results (overall acc/auroc + per-class) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["M1_cal", "M2_cal", "M3_cal", "mean_prob", "majority_vote",
             "combiner_logreg", "combiner_mlp"]
    sub = ev.loc[order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(x - 0.2, sub["acc"], 0.4, label="accuracy")
    axes[0].bar(x + 0.2, sub["auroc"], 0.4, label="AUROC")
    axes[0].set_title("Eval: overall accuracy & AUROC"); axes[0].legend()
    axes[0].axhline(0.5, ls="--", c="gray", lw=0.8)
    axes[1].bar(x - 0.2, sub["real_acc"], 0.4, label="real_acc")
    axes[1].bar(x + 0.2, sub["fake_acc"], 0.4, label="fake_acc")
    axes[1].set_title("Eval: per-class accuracy"); axes[1].legend()
    axes[1].axhline(0.5, ls="--", c="gray", lw=0.8)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
    fig.suptitle("Figure 3 — Combiner vs members & baselines on held-out modern data")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "fig3_results.png", dpi=150)
    print(f"\n[combiner] saved -> results/phase5_results.csv, phase5_summary.json, "
          f"figures/fig3_results.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
