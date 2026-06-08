"""Phase 4 — decorrelation analysis (Figure 2, the intellectual core).

An ensemble only helps if members make UNCORRELATED errors. This computes the
pairwise error-correlation between M1/M2/M3 on the held-out eval set (and on the
OOD eval FAKES specifically — the spec's focus), plus logit correlation and an
oracle-coverage stat that says how much head-room a combiner even has.

Outputs:
  results/decorrelation_err_evalfakes.csv   pairwise error-corr on eval fakes
  results/decorrelation_err_eval.csv        pairwise error-corr on all eval
  results/decorrelation_logit_eval.csv      pairwise logit-corr on all eval
  results/decorrelation_summary.json        oracle coverage + best-member
  figures/fig2_decorrelation.png            Figure 2

Interpretation: low error-correlation -> members miss different images -> a
combiner can beat the best single member. High -> combiner ~= best member (still
a valid finding).

Usage:
  ./.venv/bin/python scripts/decorrelation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

MEMBERS = [("M1", "p1"), ("M2", "p2"), ("M3", "p3")]
LOGITS = [("M1", "logit1"), ("M2", "logit2"), ("M3", "logit3")]


def error_frame(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-image error indicator (1 = wrong at 0.5) for each member."""
    return pd.DataFrame({
        name: ((sub[col] > 0.5).astype(int) != sub["label"]).astype(int)
        for name, col in MEMBERS
    })


def corr_or_nan(frame: pd.DataFrame) -> pd.DataFrame:
    """Pearson corr; constant columns (no error variance) -> NaN, diagonal 1."""
    c = frame.corr(method="pearson")
    for m in c.columns:
        if frame[m].nunique() <= 1:
            c.loc[m, :] = np.nan
            c.loc[:, m] = np.nan
            c.loc[m, m] = 1.0
    return c


def main() -> int:
    config.set_seed()
    path = config.CACHE_DIR / "member_outputs.parquet"
    df = pd.read_parquet(path)

    eval_all = df[df["split"] == config.SPLIT_EVAL].reset_index(drop=True)
    eval_fakes = eval_all[eval_all["label"] == 1].reset_index(drop=True)
    print(f"[decorr] eval n={len(eval_all)} | eval fakes n={len(eval_fakes)}")

    err_evalfakes = corr_or_nan(error_frame(eval_fakes))
    err_eval = corr_or_nan(error_frame(eval_all))
    logit_eval = eval_all[[c for _, c in LOGITS]].corr(method="pearson")
    logit_eval.columns = logit_eval.index = [n for n, _ in LOGITS]

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    err_evalfakes.to_csv(config.RESULTS_DIR / "decorrelation_err_evalfakes.csv")
    err_eval.to_csv(config.RESULTS_DIR / "decorrelation_err_eval.csv")
    logit_eval.to_csv(config.RESULTS_DIR / "decorrelation_logit_eval.csv")

    print("\n[decorr] error-correlation on EVAL FAKES (Pool C fakes):")
    print(err_evalfakes.round(3).to_string())

    # --- oracle coverage: how much head-room does a combiner have? ---
    def acc(sub, col):
        return float(((sub[col] > 0.5).astype(int) == sub["label"]).mean())

    member_acc = {n: acc(eval_all, c) for n, c in MEMBERS}
    best_member = max(member_acc, key=member_acc.get)
    preds = {n: (eval_all[c] > 0.5).astype(int) for n, c in MEMBERS}
    correct = pd.DataFrame({n: (preds[n] == eval_all["label"]).astype(int) for n in preds})
    oracle = float((correct.sum(axis=1) >= 1).mean())             # any member right
    majority = float(((sum(preds.values()) >= 2).astype(int) == eval_all["label"]).mean())
    mean_prob = float(((eval_all[["p1", "p2", "p3"]].mean(axis=1) > 0.5).astype(int)
                       == eval_all["label"]).mean())

    summary = {
        "eval_n": int(len(eval_all)), "eval_fakes_n": int(len(eval_fakes)),
        "member_acc": member_acc, "best_member": best_member,
        "best_member_acc": member_acc[best_member],
        "oracle_any_correct": oracle,
        "baseline_majority_vote": majority,
        "baseline_mean_prob": mean_prob,
        "mean_err_corr_evalfakes": float(np.nanmean(
            err_evalfakes.where(~np.eye(3, dtype=bool)).to_numpy())),
    }
    (config.RESULTS_DIR / "decorrelation_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[decorr] eval accuracies:",
          {k: round(v, 3) for k, v in member_acc.items()})
    print(f"[decorr] best single member: {best_member} @ {member_acc[best_member]:.3f}")
    print(f"[decorr] oracle (any member correct): {oracle:.3f}  "
          f"<- combiner head-room ceiling")
    print(f"[decorr] baselines: majority={majority:.3f}  mean-prob={mean_prob:.3f}")
    print(f"[decorr] mean off-diagonal err-corr (eval fakes): "
          f"{summary['mean_err_corr_evalfakes']:.3f}")

    # --- Figure 2 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, mat, title in [
        (axes[0], err_evalfakes, "Error-correlation\n(eval fakes / Pool C)"),
        (axes[1], logit_eval, "Logit-correlation\n(all eval)"),
    ]:
        M = mat.to_numpy(dtype=float)
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(mat.columns); ax.set_yticklabels(mat.index)
        for i in range(3):
            for j in range(3):
                v = M[i, j]
                txt = "n/a" if np.isnan(v) else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if abs(0 if np.isnan(v) else v) > 0.5 else "black",
                        fontsize=11)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Figure 2 — Member decorrelation on held-out modern data", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = config.FIGURES_DIR / "fig2_decorrelation.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\n[decorr] saved Figure 2 -> {fig_path}")
    print("[decorr] saved matrices + summary -> results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
