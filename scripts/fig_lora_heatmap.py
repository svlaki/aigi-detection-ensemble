"""LoRA sweep heatmap — rank × epochs colored by accuracy/AUROC.

Cleaner than the line plot for showing the overfitting pattern at high
rank + high epochs. Produces one heatmap per eval split (modern_test, cf, eval).

Outputs: figures/fig9_lora_heatmap.png

Usage:
  python scripts/fig_lora_heatmap.py
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

    csv_path = config.RESULTS_DIR / "lora_sweep.csv"
    if not csv_path.exists():
        print(f"[heatmap] ERROR: {csv_path} not found.")
        return 1

    df = pd.read_csv(csv_path)
    m1b = df[df["model"] == "M1b"].copy()
    m1a = df[df["model"] == "M1a"].copy()

    splits = ["modern_test", "cf", "eval"]
    split_labels = {
        "modern_test": "modern_test\n(in-generator)",
        "cf": "Community Forensics\n(cross-generator)",
        "eval": "Full eval",
    }
    metric_pairs = [("acc", "Accuracy"), ("auroc", "AUROC")]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    for col, split in enumerate(splits):
        sub = m1b[m1b["split"] == split]
        baseline_sub = m1a[m1a["split"] == split]

        for row, (metric, metric_label) in enumerate(metric_pairs):
            ax = axes[row, col]

            # pivot to rank × epochs grid
            piv = sub.pivot_table(index="epochs", columns="r", values=metric)
            piv = piv.sort_index(ascending=False)  # epochs 10 on top, 5 on bottom

            # M1a baseline for this split/metric
            baseline = baseline_sub[metric].mean()

            # compute delta from baseline for annotation
            im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto",
                           vmin=min(piv.values.min(), baseline) - 0.01,
                           vmax=max(piv.values.max(), baseline) + 0.01)

            # annotate cells with value and delta
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    val = piv.values[i, j]
                    delta = val - baseline
                    ax.text(j, i, f"{val:.3f}\n({delta:+.3f})",
                            ha="center", va="center", fontsize=8,
                            color="black" if 0.4 < val < 0.9 else "white")

            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([f"r={r}" for r in piv.columns], fontsize=9)
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels([f"ep={e}" for e in piv.index], fontsize=9)

            if col == 0:
                ax.set_ylabel(metric_label, fontsize=11)
            if row == 0:
                ax.set_title(split_labels[split], fontsize=10)

            plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("LoRA sweep: rank x epochs heatmap (M1b values, delta from M1a baseline)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / "fig9_lora_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[heatmap] saved -> {out_path}")

    # print the key finding
    cf_sub = m1b[m1b["split"] == "cf"]
    best = cf_sub.loc[cf_sub["acc"].idxmax()]
    worst = cf_sub.loc[cf_sub["acc"].idxmin()]
    print(f"\n[heatmap] CF best:  r={int(best['r'])} ep={int(best['epochs'])} "
          f"acc={best['acc']:.3f}")
    print(f"[heatmap] CF worst: r={int(worst['r'])} ep={int(worst['epochs'])} "
          f"acc={worst['acc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
