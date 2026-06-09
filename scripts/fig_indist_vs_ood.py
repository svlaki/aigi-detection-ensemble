"""In-distribution vs OOD gap chart.

Side-by-side bars showing each method's accuracy on Pool B (combiner_fit,
in-distribution) vs eval (OOD). Quantifies the generalization collapse.

Outputs: figures/fig8_indist_vs_ood.png

Usage:
  python scripts/fig_indist_vs_ood.py
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

    csv_path = config.RESULTS_DIR / "phase5_results.csv"
    if not csv_path.exists():
        print(f"[gap-fig] ERROR: {csv_path} not found.")
        return 1

    df = pd.read_csv(csv_path)

    methods = ["M1_cal", "M2_cal", "M3_cal", "mean_prob", "majority_vote",
               "combiner_logreg", "combiner_mlp"]
    labels = ["M1", "M2", "M3", "Mean\nprob", "Majority\nvote", "Combiner\n(LogReg)", "Combiner\n(MLP)"]

    indist = df[df["split"] == "combiner_fit"].set_index("method")
    ood = df[df["split"] == "eval"].set_index("method")

    x = np.arange(len(methods))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: overall accuracy
    ax = axes[0]
    bars_in = ax.bar(x - width / 2, [indist.loc[m, "acc"] for m in methods],
                     width, label="In-distribution (Pool B)", color="tab:blue", alpha=0.85)
    bars_ood = ax.bar(x + width / 2, [ood.loc[m, "acc"] for m in methods],
                      width, label="OOD (eval)", color="tab:red", alpha=0.85)

    # annotate gaps
    for i, m in enumerate(methods):
        gap = indist.loc[m, "acc"] - ood.loc[m, "acc"]
        y_pos = max(indist.loc[m, "acc"], ood.loc[m, "acc"]) + 0.01
        ax.text(i, y_pos, f"-{gap:.0%}", ha="center", va="bottom", fontsize=7,
                color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    ax.legend(fontsize=9)
    ax.set_title("Overall accuracy: in-distribution vs OOD")

    # Panel 2: AUROC
    ax = axes[1]
    ax.bar(x - width / 2, [indist.loc[m, "auroc"] for m in methods],
           width, label="In-distribution (Pool B)", color="tab:blue", alpha=0.85)
    ax.bar(x + width / 2, [ood.loc[m, "auroc"] for m in methods],
           width, label="OOD (eval)", color="tab:red", alpha=0.85)

    for i, m in enumerate(methods):
        gap = indist.loc[m, "auroc"] - ood.loc[m, "auroc"]
        y_pos = max(indist.loc[m, "auroc"], ood.loc[m, "auroc"]) + 0.01
        ax.text(i, y_pos, f"-{gap:.0%}", ha="center", va="bottom", fontsize=7,
                color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    ax.legend(fontsize=9)
    ax.set_title("AUROC: in-distribution vs OOD")

    fig.suptitle("In-distribution (Pool B) vs OOD (eval) — the generalization gap",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / "fig8_indist_vs_ood.png"
    fig.savefig(out_path, dpi=150)
    print(f"[gap-fig] saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
