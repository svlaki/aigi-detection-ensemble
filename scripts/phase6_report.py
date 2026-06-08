"""Phase 6 report — M1a-vs-M1b ablation table + Figure 4 (HEADLINE).

Reads results/phase6_metrics.json (written by `modal run modal_app.py::lora_main`)
and produces:
  - a printed per-class table (M1a vs M1b) for modern_test / cf / eval
  - results/phase6_summary.json  (deltas + framing)
  - figures/fig4_lora.png        (grouped per-class bars, M1a vs M1b)

modern_test = in-generator held-out images (the headline). cf = cross-generator
(unseen modern generators) generalization stress test. Report both deltas honestly
— LoRA may help in-generator yet hurt cross-generator (overfit), which is itself a
finding ("training-data diversity necessary but not sufficient").

Usage:
  ./.venv/bin/python scripts/phase6_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

SETS = ["modern_test", "cf", "eval"]
METRICS = ["auroc", "acc", "real_acc", "fake_acc"]


def main() -> int:
    meta = json.loads((config.RESULTS_DIR / "phase6_metrics.json").read_text())
    R = meta["results"]

    print(f"[p6] LoRA config: r={meta['r']} alpha={meta['alpha']} "
          f"dropout={meta['dropout']} targets={meta['target_modules']} "
          f"epochs={meta['epochs']}")
    print("\n[p6] === M1a (frozen) vs M1b (LoRA) — per-class ===")
    header = f"{'set':12s} {'model':4s}  {'auroc':>6s} {'acc':>6s} {'real':>6s} {'fake':>6s}"
    print(header); print("-" * len(header))
    summary = {"lora": {k: meta[k] for k in ("r", "alpha", "dropout", "epochs", "target_modules")},
               "sets": {}}
    for s in SETS:
        a, b = R[f"M1a/{s}"], R[f"M1b/{s}"]
        print(f"{s:12s} {'M1a':4s}  " + " ".join(f"{a[m]:6.3f}" for m in METRICS))
        print(f"{'':12s} {'M1b':4s}  " + " ".join(f"{b[m]:6.3f}" for m in METRICS))
        delta = {m: round(b[m] - a[m], 4) for m in METRICS}
        print(f"{'':12s} {'Δ':4s}  " + " ".join(f"{delta[m]:+6.3f}" for m in METRICS))
        summary["sets"][s] = {"M1a": a, "M1b": b, "delta": delta}

    # headline framing
    mt = summary["sets"]["modern_test"]["delta"]
    cf = summary["sets"]["cf"]["delta"]
    summary["headline"] = {
        "modern_test_acc_delta": mt["acc"], "modern_test_auroc_delta": mt["auroc"],
        "cf_acc_delta": cf["acc"], "cf_auroc_delta": cf["auroc"],
        "reads": ("LoRA helps in-generator (modern_test) "
                  + ("and" if cf["auroc"] > 0 else "but") +
                  " cross-generator (cf)"),
    }
    (config.RESULTS_DIR / "phase6_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[p6] modern_test Δacc={mt['acc']:+.3f} Δauroc={mt['auroc']:+.3f} | "
          f"cf Δacc={cf['acc']:+.3f} Δauroc={cf['auroc']:+.3f}")

    # ---- Figure 4 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    panels = ["modern_test", "cf"]
    titles = {"modern_test": "modern_test (in-generator, held-out)",
              "cf": "Community Forensics (cross-generator)"}
    bars = ["acc", "real_acc", "fake_acc"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, s in zip(axes, panels):
        a, b = R[f"M1a/{s}"], R[f"M1b/{s}"]
        x = np.arange(len(bars))
        ax.bar(x - 0.2, [a[m] for m in bars], 0.4, label="M1a (frozen)")
        ax.bar(x + 0.2, [b[m] for m in bars], 0.4, label="M1b (LoRA)")
        ax.axhline(0.5, ls="--", c="gray", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(["acc", "real_acc", "fake_acc"])
        ax.set_ylim(0, 1)
        ax.set_title(f"{titles[s]}\nAUROC: M1a {a['auroc']:.3f} → M1b {b['auroc']:.3f}",
                     fontsize=10)
        ax.legend(fontsize=9)
    fig.suptitle("Figure 4 — LoRA ablation: frozen probe (M1a) vs LoRA fine-tune (M1b)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "fig4_lora.png", dpi=150)
    print(f"[p6] saved -> figures/fig4_lora.png, results/phase6_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
