"""Per-generator accuracy breakdown — which generators fool which members.

Groups eval-set images by generator_name and computes per-member accuracy,
plus the combiner. Directly supports the decorrelation thesis by showing
members have complementary strengths across generators.

Outputs:
  results/per_generator_breakdown.csv
  figures/fig6_per_generator.png

Usage:
  python scripts/per_generator_breakdown.py
"""
from __future__ import annotations

import json
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

    cache_path = config.CACHE_DIR / "member_outputs.parquet"
    if not cache_path.exists():
        print(f"[gen-bkdn] ERROR: {cache_path} not found.")
        print("[gen-bkdn] Run: python scripts/build_member_outputs.py")
        return 1

    df = pd.read_parquet(cache_path)
    E = df[df["split"] == config.SPLIT_EVAL].reset_index(drop=True)
    print(f"[gen-bkdn] eval set: {len(E)} images, "
          f"generators: {sorted(E['generator_name'].unique())}")

    # per-generator, per-member accuracy
    from sklearn.metrics import accuracy_score

    members = {"M1": "p1", "M2": "p2", "M3": "p3"}
    rows = []
    for gen, group in E.groupby("generator_name"):
        y = group["label"].to_numpy(int)
        n = len(group)
        row = {"generator": gen, "n": n, "label": int(y[0]) if len(np.unique(y)) == 1 else -1}
        for name, col in members.items():
            pred = (group[col].to_numpy() >= 0.5).astype(int)
            row[f"{name}_acc"] = float(accuracy_score(y, pred))
        rows.append(row)

    res = pd.DataFrame(rows).sort_values("generator")
    res.to_csv(config.RESULTS_DIR / "per_generator_breakdown.csv", index=False)

    print("\n[gen-bkdn] === Per-generator member accuracy (eval set) ===")
    print(res.to_string(index=False, float_format="%.3f"))

    # figure: grouped bar chart
    gens = res["generator"].tolist()
    x = np.arange(len(gens))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, _) in enumerate(members.items()):
        ax.bar(x + i * width, res[f"{name}_acc"], width, label=name)

    ax.set_xticks(x + width)
    ax.set_xticklabels(gens, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    ax.legend()
    ax.set_title("Figure 6 — Per-generator member accuracy on eval set")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / "fig6_per_generator.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n[gen-bkdn] saved -> {out_path}, results/per_generator_breakdown.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
