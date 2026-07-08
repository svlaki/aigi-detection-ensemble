"""Robustness scoring — merge M1/M2 + D3QE outputs, apply frozen combiners.

Reads per-tag M1/M2 parquets from cache/robustness/ and optional D3QE logits
from cache/robustness/m3_<tag>.npz. Applies frozen calibrators and combiners
to produce results/robustness_results.csv and figures.

Usage:
  python scripts/robustness_score.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

ALL_TAGS = [
    "clean",
    "jpeg_q90", "jpeg_q75", "jpeg_q50", "jpeg_q30",
    "blur_s05", "blur_s10", "blur_s20", "blur_s30",
    "noise_s2", "noise_s5", "noise_s10", "noise_s20",
    "resize_128",
    "social_media",
]

TAG_LABELS = {
    "clean": "Clean",
    "jpeg_q90": "JPEG q90", "jpeg_q75": "JPEG q75",
    "jpeg_q50": "JPEG q50", "jpeg_q30": "JPEG q30",
    "blur_s05": "Blur σ=0.5", "blur_s10": "Blur σ=1.0",
    "blur_s20": "Blur σ=2.0", "blur_s30": "Blur σ=3.0",
    "noise_s2": "Noise σ=2", "noise_s5": "Noise σ=5",
    "noise_s10": "Noise σ=10", "noise_s20": "Noise σ=20",
    "resize_128": "Resize 128→256",
    "social_media": "Social Media",
}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def metrics(y: np.ndarray, score: np.ndarray, thr: float = 0.5) -> dict:
    from sklearn.metrics import roc_auc_score, accuracy_score
    pred = (score >= thr).astype(int)
    return {
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "acc": float(accuracy_score(y, pred)),
        "real_acc": float(accuracy_score(y[y == 0], pred[y == 0])) if (y == 0).any() else float("nan"),
        "fake_acc": float(accuracy_score(y[y == 1], pred[y == 1])) if (y == 1).any() else float("nan"),
    }


def load_d3qe_logits(tag: str) -> dict[str, float] | None:
    """Load cached D3QE logits for a perturbation tag."""
    path = config.CACHE_DIR / "robustness" / f"m3_{tag}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {str(i): float(data["logit"][k]) for k, i in enumerate(data["image_id"])}


def main() -> int:
    import joblib

    config.set_seed()
    rob_dir = config.CACHE_DIR / "robustness"

    # Load frozen models
    calib_m1 = joblib.load(config.MODELS_DIR / "calib_m1.joblib")
    calib_m2 = joblib.load(config.MODELS_DIR / "calib_m2.joblib")
    calib_m3 = joblib.load(config.MODELS_DIR / "calib_m3.joblib")
    combiner_lr = joblib.load(config.MODELS_DIR / "combiner_logreg.joblib")
    combiner_mlp = joblib.load(config.MODELS_DIR / "combiner_mlp.joblib")

    # Clean baseline from existing member_outputs
    clean_df = pd.read_parquet(config.CACHE_DIR / "member_outputs.parquet")
    clean_df = clean_df[clean_df["split"] == config.SPLIT_EVAL].reset_index(drop=True)

    rows = []

    for tag in ALL_TAGS:
        if tag == "clean":
            df = clean_df
        else:
            parquet_path = rob_dir / f"m12_{tag}.parquet"
            if not parquet_path.exists():
                print(f"[score] {tag}: M1/M2 parquet not found, skipping")
                continue
            df = pd.read_parquet(parquet_path)

        y = df["label"].to_numpy(int)

        # M1/M2 calibration
        logit1 = df["logit1"].to_numpy()
        logit2 = df["logit2"].to_numpy()
        p1_cal = calib_m1.predict_proba(logit1.reshape(-1, 1))[:, 1]
        z1 = calib_m1.decision_function(logit1.reshape(-1, 1))
        p2_cal = calib_m2.predict_proba(logit2.reshape(-1, 1))[:, 1]
        z2 = calib_m2.decision_function(logit2.reshape(-1, 1))

        # Per-member metrics (calibrated)
        rows.append({"perturbation": tag, "method": "M1_cal", **metrics(y, p1_cal)})
        rows.append({"perturbation": tag, "method": "M2_cal", **metrics(y, p2_cal)})

        # M1+M2 mean (always available)
        m12_mean = (p1_cal + p2_cal) / 2.0
        rows.append({"perturbation": tag, "method": "m12_mean", **metrics(y, m12_mean)})

        # Try to load D3QE logits
        if tag == "clean":
            logit3 = df["logit3"].to_numpy()
            has_m3 = True
        else:
            d3qe_cache = load_d3qe_logits(tag)
            if d3qe_cache is not None:
                ids = df["image_id"].astype(str).tolist()
                logit3 = np.array([d3qe_cache.get(i, np.nan) for i in ids])
                has_m3 = not np.isnan(logit3).any()
            else:
                has_m3 = False

        if has_m3:
            p3_cal = calib_m3.predict_proba(logit3.reshape(-1, 1))[:, 1]
            z3 = calib_m3.decision_function(logit3.reshape(-1, 1))
            rows.append({"perturbation": tag, "method": "M3_cal", **metrics(y, p3_cal)})

            # Mean prob (3 members)
            mean_prob = (p1_cal + p2_cal + p3_cal) / 3.0
            rows.append({"perturbation": tag, "method": "mean_prob", **metrics(y, mean_prob)})

            # Majority vote
            vote = ((p1_cal >= 0.5).astype(int) + (p2_cal >= 0.5).astype(int) +
                    (p3_cal >= 0.5).astype(int)) / 3.0
            rows.append({"perturbation": tag, "method": "majority_vote", **metrics(y, vote)})

            # Learned combiners (9-dim feature)
            feat = np.column_stack([
                p1_cal, p2_cal, p3_cal,
                z1, z2, z3,
                np.abs(p1_cal - p2_cal),
                np.abs(p1_cal - p3_cal),
                np.abs(p2_cal - p3_cal),
            ])
            lr_prob = combiner_lr.predict_proba(feat)[:, 1]
            mlp_prob = combiner_mlp.predict_proba(feat)[:, 1]
            rows.append({"perturbation": tag, "method": "combiner_logreg", **metrics(y, lr_prob)})
            rows.append({"perturbation": tag, "method": "combiner_mlp", **metrics(y, mlp_prob)})

        print(f"[score] {tag}: scored ({'full' if has_m3 else 'M1+M2 only'})")

    results = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.RESULTS_DIR / "robustness_results.csv", index=False)
    print(f"\n[score] saved -> results/robustness_results.csv ({len(results)} rows)")

    # Summary JSON
    clean_rows = results[results["perturbation"] == "clean"]
    summary = {"perturbations_tested": list(results["perturbation"].unique())}

    for method in results["method"].unique():
        clean_met = clean_rows[clean_rows["method"] == method]
        if clean_met.empty:
            continue
        clean_auroc = clean_met.iloc[0]["auroc"]
        per_pert = {}
        for tag in results["perturbation"].unique():
            if tag == "clean":
                continue
            row = results[(results["perturbation"] == tag) & (results["method"] == method)]
            if row.empty:
                continue
            per_pert[tag] = {
                "auroc": round(row.iloc[0]["auroc"], 4),
                "auroc_delta": round(row.iloc[0]["auroc"] - clean_auroc, 4),
            }
        summary[method] = {"clean_auroc": round(clean_auroc, 4), "per_perturbation": per_pert}

    (config.RESULTS_DIR / "robustness_summary.json").write_text(json.dumps(summary, indent=2))
    print("[score] saved -> results/robustness_summary.json")

    # Generate figures
    _make_figures(results)
    return 0


def _make_figures(results: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: AUROC degradation lines by method
    methods_to_plot = ["M1_cal", "M2_cal", "combiner_logreg", "m12_mean"]
    methods_available = [m for m in methods_to_plot if m in results["method"].unique()]
    if "M3_cal" in results["method"].unique():
        methods_available.insert(2, "M3_cal")

    colors = {
        "M1_cal": "#3b82f6", "M2_cal": "#a855f7", "M3_cal": "#f59e0b",
        "combiner_logreg": "#ef4444", "combiner_mlp": "#f97316", "m12_mean": "#06b6d4",
        "mean_prob": "#22c55e", "majority_vote": "#8b5cf6",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    for method in methods_available:
        sub = results[results["method"] == method].copy()
        # Order by ALL_TAGS
        tag_order = {t: i for i, t in enumerate(ALL_TAGS)}
        sub = sub[sub["perturbation"].isin(ALL_TAGS)]
        sub["order"] = sub["perturbation"].map(tag_order)
        sub = sub.sort_values("order")
        labels = [TAG_LABELS.get(t, t) for t in sub["perturbation"]]
        ax.plot(labels, sub["auroc"].values, marker="o", markersize=4,
                label=method, color=colors.get(method, "#888"))

    ax.set_ylabel("AUROC")
    ax.set_title("Robustness: AUROC under perturbation")
    ax.legend(fontsize=8)
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig_robustness.png", dpi=150)
    print(f"[score] saved -> figures/fig_robustness.png")

    # Figure 2: Heatmap — method × perturbation, AUROC delta from clean
    perturb_tags = [t for t in ALL_TAGS if t != "clean"]
    available_methods = sorted(results["method"].unique())
    clean_lookup = {}
    for _, r in results[results["perturbation"] == "clean"].iterrows():
        clean_lookup[r["method"]] = r["auroc"]

    matrix = []
    method_labels = []
    for method in available_methods:
        row = []
        for tag in perturb_tags:
            r = results[(results["perturbation"] == tag) & (results["method"] == method)]
            if r.empty:
                row.append(np.nan)
            else:
                row.append(r.iloc[0]["auroc"] - clean_lookup.get(method, 0))
        if not all(np.isnan(v) for v in row):
            matrix.append(row)
            method_labels.append(method)

    if matrix:
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        mat = np.array(matrix)
        im = ax2.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.05)
        ax2.set_xticks(range(len(perturb_tags)))
        ax2.set_xticklabels([TAG_LABELS.get(t, t) for t in perturb_tags],
                            rotation=45, ha="right", fontsize=8)
        ax2.set_yticks(range(len(method_labels)))
        ax2.set_yticklabels(method_labels, fontsize=8)
        for i in range(len(method_labels)):
            for j in range(len(perturb_tags)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax2.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=6,
                             color="white" if v < -0.1 else "black")
        fig2.colorbar(im, ax=ax2, label="AUROC delta from clean")
        ax2.set_title("Robustness: AUROC degradation heatmap")
        fig2.tight_layout()
        fig2.savefig(config.FIGURES_DIR / "fig_robustness_heatmap.png", dpi=150)
        print(f"[score] saved -> figures/fig_robustness_heatmap.png")

    plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())
