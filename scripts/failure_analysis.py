"""Failure case analysis — find images each member and the combiner get wrong.

For each method (M1, M2, M3, combiner), identifies failure cases on the eval
set and categorizes them:
  - Combiner failures: what did each member predict?
  - Per-member failures: did the other members get it right?
  - Disagreement cases: members disagree, combiner has to break the tie

Saves a detailed CSV of all eval predictions + error flags, a summary of
failure patterns, and a visual grid of sample failure images (if normalized
images are available locally).

Outputs:
  results/failure_analysis.csv       — every eval image with all predictions
  results/failure_summary.json       — aggregated failure patterns
  figures/fig_failure_examples.png   — grid of sample failures (if images exist)

Usage:
  python scripts/failure_analysis.py
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

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cache_path = config.CACHE_DIR / "member_outputs.parquet"
    if not cache_path.exists():
        print(f"[failure] ERROR: {cache_path} not found.")
        print("[failure] Run: python scripts/build_member_outputs.py")
        return 1

    config.set_seed()
    df = pd.read_parquet(cache_path)
    B = df[df["split"] == config.SPLIT_COMBINER_FIT].reset_index(drop=True)
    E = df[df["split"] == config.SPLIT_EVAL].reset_index(drop=True)
    yB = B["label"].to_numpy(int)

    # --- refit calibrators + combiner (same as combiner.py) ---
    member_logit_cols = [("M1", "logit1"), ("M2", "logit2"), ("M3", "logit3")]
    calibrators = {}
    for m, lcol in member_logit_cols:
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(B[[lcol]].to_numpy(), yB)
        calibrators[m] = lr

    def calibrated_probs(frame):
        out = {}
        for m, lcol in member_logit_cols:
            out[m] = calibrators[m].predict_proba(frame[[lcol]].to_numpy())[:, 1]
        return out

    def build_features(frame):
        cp = calibrated_probs(frame)
        p1, p2, p3 = cp["M1"], cp["M2"], cp["M3"]
        z1 = calibrators["M1"].decision_function(frame[["logit1"]].to_numpy())
        z2 = calibrators["M2"].decision_function(frame[["logit2"]].to_numpy())
        z3 = calibrators["M3"].decision_function(frame[["logit3"]].to_numpy())
        return np.column_stack([p1, p2, p3, z1, z2, z3,
                                np.abs(p1 - p2), np.abs(p1 - p3), np.abs(p2 - p3)])

    comb = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    comb.fit(build_features(B), yB)

    # --- score eval set ---
    cal_probs = calibrated_probs(E)
    FE = build_features(E)
    comb_prob = comb.predict_proba(FE)[:, 1]

    out = E[["image_id", "label", "generator_name", "source_dataset"]].copy()
    out["m1_prob"] = cal_probs["M1"]
    out["m2_prob"] = cal_probs["M2"]
    out["m3_prob"] = cal_probs["M3"]
    out["comb_prob"] = comb_prob
    out["m1_pred"] = (cal_probs["M1"] >= 0.5).astype(int)
    out["m2_pred"] = (cal_probs["M2"] >= 0.5).astype(int)
    out["m3_pred"] = (cal_probs["M3"] >= 0.5).astype(int)
    out["comb_pred"] = (comb_prob >= 0.5).astype(int)
    out["m1_correct"] = (out["m1_pred"] == out["label"]).astype(int)
    out["m2_correct"] = (out["m2_pred"] == out["label"]).astype(int)
    out["m3_correct"] = (out["m3_pred"] == out["label"]).astype(int)
    out["comb_correct"] = (out["comb_pred"] == out["label"]).astype(int)
    out["n_members_correct"] = out["m1_correct"] + out["m2_correct"] + out["m3_correct"]

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.RESULTS_DIR / "failure_analysis.csv", index=False)

    # --- summarize failure patterns ---
    y = out["label"].to_numpy(int)
    n_eval = len(out)

    # combiner failures
    comb_fails = out[out["comb_correct"] == 0]
    comb_fail_real = comb_fails[comb_fails["label"] == 0]  # real called fake
    comb_fail_fake = comb_fails[comb_fails["label"] == 1]  # fake called real

    # failure pattern: how many members got it right when combiner failed
    comb_fail_member_correct = comb_fails["n_members_correct"].value_counts().sort_index().to_dict()

    # per-member unique failures: images only this member gets wrong
    m1_only_fail = out[(out["m1_correct"] == 0) & (out["m2_correct"] == 1) & (out["m3_correct"] == 1)]
    m2_only_fail = out[(out["m1_correct"] == 1) & (out["m2_correct"] == 0) & (out["m3_correct"] == 1)]
    m3_only_fail = out[(out["m1_correct"] == 1) & (out["m2_correct"] == 1) & (out["m3_correct"] == 0)]
    all_fail = out[(out["m1_correct"] == 0) & (out["m2_correct"] == 0) & (out["m3_correct"] == 0)]
    all_correct = out[(out["m1_correct"] == 1) & (out["m2_correct"] == 1) & (out["m3_correct"] == 1)]

    # per-generator combiner failure rate
    gen_fail = (out.groupby("generator_name")
                .agg(n=("comb_correct", "count"),
                     n_fail=("comb_correct", lambda x: (x == 0).sum()))
                .reset_index())
    gen_fail["fail_rate"] = gen_fail["n_fail"] / gen_fail["n"]
    gen_fail = gen_fail.sort_values("fail_rate", ascending=False)

    summary = {
        "eval_n": n_eval,
        "combiner_failures": {
            "total": len(comb_fails),
            "real_as_fake": len(comb_fail_real),
            "fake_as_real": len(comb_fail_fake),
            "members_correct_when_combiner_wrong": comb_fail_member_correct,
        },
        "unique_failures": {
            "m1_only_wrong": len(m1_only_fail),
            "m2_only_wrong": len(m2_only_fail),
            "m3_only_wrong": len(m3_only_fail),
            "all_three_wrong": len(all_fail),
            "all_three_correct": len(all_correct),
        },
        "per_generator_combiner_fail_rate": gen_fail[["generator_name", "n", "n_fail", "fail_rate"]]
            .to_dict(orient="records"),
    }
    (config.RESULTS_DIR / "failure_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[failure] eval set: {n_eval} images")
    print(f"[failure] combiner failures: {len(comb_fails)} "
          f"({len(comb_fail_real)} real-as-fake, {len(comb_fail_fake)} fake-as-real)")
    print(f"[failure] when combiner wrong, members correct: {comb_fail_member_correct}")
    print(f"\n[failure] unique failures:")
    print(f"  only M1 wrong: {len(m1_only_fail)}")
    print(f"  only M2 wrong: {len(m2_only_fail)}")
    print(f"  only M3 wrong: {len(m3_only_fail)}")
    print(f"  all 3 wrong:   {len(all_fail)}")
    print(f"  all 3 correct: {len(all_correct)}")
    print(f"\n[failure] per-generator combiner fail rate:")
    print(gen_fail[["generator_name", "n", "n_fail", "fail_rate"]]
          .to_string(index=False, float_format="%.3f"))

    # --- visual grid of failure examples ---
    # try to load actual images; fall back to a text-only summary figure
    manifest = pd.read_csv(config.MANIFEST_CSV)
    path_map = dict(zip(manifest["image_id"].astype(str), manifest["path"]))

    categories = [
        ("Combiner wrong, all members right",
         comb_fails[comb_fails["n_members_correct"] == 3]),
        ("Combiner wrong, 2 members right",
         comb_fails[comb_fails["n_members_correct"] == 2]),
        ("All models wrong (hardest cases)",
         all_fail),
        ("Only M1 wrong",
         m1_only_fail),
        ("Only M2 wrong",
         m2_only_fail),
        ("Only M3 wrong",
         m3_only_fail),
    ]

    n_per_cat = 4
    cats_with_data = [(name, df_cat) for name, df_cat in categories if len(df_cat) > 0]
    n_rows = len(cats_with_data)

    if n_rows == 0:
        print("[failure] no failure cases to visualize")
        return 0

    fig, axes = plt.subplots(n_rows, n_per_cat, figsize=(3.5 * n_per_cat, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    has_images = False
    for row, (cat_name, df_cat) in enumerate(cats_with_data):
        samples = df_cat.sample(n=min(n_per_cat, len(df_cat)), random_state=config.SEED)
        for col in range(n_per_cat):
            ax = axes[row, col]
            if col >= len(samples):
                ax.axis("off")
                continue

            s = samples.iloc[col]
            img_path = config.ROOT / path_map.get(str(s["image_id"]), "")

            if img_path.exists():
                from PIL import Image
                img = Image.open(img_path)
                ax.imshow(img)
                has_images = True
            else:
                ax.text(0.5, 0.5, s["image_id"], ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)
                ax.set_facecolor("#f0f0f0")

            true_label = "REAL" if s["label"] == 0 else "FAKE"
            gen = s["generator_name"]
            m1 = "T" if s["m1_correct"] else "F"
            m2 = "T" if s["m2_correct"] else "F"
            m3 = "T" if s["m3_correct"] else "F"
            cb = "T" if s["comb_correct"] else "F"

            ax.set_title(f"True: {true_label} ({gen})\n"
                         f"M1:{m1} M2:{m2} M3:{m3} Comb:{cb}",
                         fontsize=7, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[row, 0].set_ylabel(cat_name, fontsize=9, rotation=0, ha="right",
                                 va="center", labelpad=10)

    fig.suptitle("Failure case examples by category\n"
                 "(T=correct, F=incorrect for each model)",
                 fontsize=12)
    fig.tight_layout(rect=(0.12, 0, 1, 0.94))
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / "fig_failure_examples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    img_note = "with images" if has_images else "text-only (images not found locally)"
    print(f"\n[failure] saved figure -> {out_path} ({img_note})")
    print(f"[failure] saved -> results/failure_analysis.csv, results/failure_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
