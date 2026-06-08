"""Validation — negative controls. If our numbers are real (no leakage, no
pipeline cheating), each of these MUST collapse to ~0.5 AUROC. The real numbers
are printed alongside for contrast.

Controls:
  A. Combiner label-permutation: shuffle Pool B (combiner_fit) labels, refit
     calibration + combiner on the shuffled labels, evaluate on TRUE eval labels.
     -> ~0.5 proves the combiner's gain comes from genuine label-feature structure,
        not from overfitting the fit set. (n_perm seeds, report mean±std.)
  B. Random-feature combiner: replace member features with Gaussian noise, fit on
     TRUE Pool B labels, eval on TRUE eval. -> ~0.5 proves it can't manufacture
        signal from noise.
  C. Member probe label-shuffle: refit M1a (CLIP) and M2 (spectral) on shuffled
     member_train labels, eval on TRUE eval. -> ~0.5 proves each member's signal is
        real, not memorized structure.

Usage:
  ./.venv/bin/python scripts/negative_controls.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, embeddings, spectral

N_PERM = 5


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def combiner_pipeline(B, E, yB):
    """Calibrate 3 members on (B, yB), fit LogReg combiner, return eval scores."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cols = ["logit1", "logit2", "logit3"]

    def feats(frame):
        ps, zs = [], []
        for c in cols:
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(B[[c]].to_numpy(), yB)
            ps.append(lr.predict_proba(frame[[c]].to_numpy())[:, 1])
            zs.append(lr.decision_function(frame[[c]].to_numpy()))
        p1, p2, p3 = ps
        return np.column_stack([p1, p2, p3, zs[0], zs[1], zs[2],
                                np.abs(p1 - p2), np.abs(p1 - p3), np.abs(p2 - p3)])

    comb = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    comb.fit(feats(B), yB)
    return comb.predict_proba(feats(E))[:, 1]


def member_probe(Xtr, ytr, Xev, kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler
    pre = Normalizer("l2") if kind == "clip" else StandardScaler()
    clf = make_pipeline(pre, LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xev)[:, 1]


def main() -> int:
    config.set_seed()
    rng = np.random.default_rng(config.SEED)
    df = pd.read_parquet(config.CACHE_DIR / "member_outputs.parquet")
    B = df[df.split == config.SPLIT_COMBINER_FIT].reset_index(drop=True)
    E = df[df.split == config.SPLIT_EVAL].reset_index(drop=True)
    yB, yE = B.label.to_numpy(int), E.label.to_numpy(int)

    out = {}

    # --- real combiner (reference) ---
    real_comb = auroc(yE, combiner_pipeline(B, E, yB))

    # --- A. combiner label-permutation ---
    perm = [auroc(yE, combiner_pipeline(B, E, rng.permutation(yB))) for _ in range(N_PERM)]
    out["A_combiner_label_perm"] = {"real_auroc": round(real_comb, 3),
                                    "perm_auroc_mean": round(float(np.mean(perm)), 3),
                                    "perm_auroc_std": round(float(np.std(perm)), 3)}

    # --- B. random-feature combiner ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    noise_clf = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
    noise_clf.fit(rng.standard_normal((len(B), 9)), yB)
    rand_auroc = auroc(yE, noise_clf.predict_proba(rng.standard_normal((len(E), 9)))[:, 1])
    out["B_random_feature_combiner"] = {"auroc": round(rand_auroc, 3)}

    # --- C. member probe label-shuffle (M1 CLIP, M2 spectral) ---
    mt = df[df.split == config.SPLIT_MEMBER_TRAIN]
    clip_c, spec_c = embeddings.load_cache(), spectral.load_cache()
    Xc_tr, _ = embeddings.matrix_for(clip_c, mt.image_id); Xc_ev, _ = embeddings.matrix_for(clip_c, E.image_id)
    Xs_tr, _ = spectral.matrix_for(spec_c, mt.image_id); Xs_ev, _ = spectral.matrix_for(spec_c, E.image_id)
    ymt = mt.label.to_numpy(int)

    real_m1 = auroc(yE, member_probe(Xc_tr, ymt, Xc_ev, "clip"))
    real_m2 = auroc(yE, member_probe(Xs_tr, ymt, Xs_ev, "spec"))
    sh_m1 = [auroc(yE, member_probe(Xc_tr, rng.permutation(ymt), Xc_ev, "clip")) for _ in range(N_PERM)]
    sh_m2 = [auroc(yE, member_probe(Xs_tr, rng.permutation(ymt), Xs_ev, "spec")) for _ in range(N_PERM)]
    out["C_member_label_shuffle"] = {
        "M1_real_auroc": round(real_m1, 3), "M1_shuffled_auroc_mean": round(float(np.mean(sh_m1)), 3),
        "M2_real_auroc": round(real_m2, 3), "M2_shuffled_auroc_mean": round(float(np.mean(sh_m2)), 3),
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "negative_controls.json").write_text(json.dumps(out, indent=2))

    print("=== NEGATIVE CONTROLS (must collapse to ~0.5) ===\n")
    a = out["A_combiner_label_perm"]
    print(f"A. Combiner label-permutation (Pool B labels shuffled, eval on true labels):")
    print(f"   real combiner AUROC = {a['real_auroc']}   "
          f"shuffled = {a['perm_auroc_mean']} ± {a['perm_auroc_std']}   (want ~0.5)")
    print(f"\nB. Random-feature combiner (noise features, true labels):")
    print(f"   AUROC = {out['B_random_feature_combiner']['auroc']}   (want ~0.5)")
    c = out["C_member_label_shuffle"]
    print(f"\nC. Member probe label-shuffle (member_train labels shuffled, eval on true):")
    print(f"   M1 real = {c['M1_real_auroc']}  shuffled = {c['M1_shuffled_auroc_mean']}   (want ~0.5)")
    print(f"   M2 real = {c['M2_real_auroc']}  shuffled = {c['M2_shuffled_auroc_mean']}   (want ~0.5)")

    ok = (a["perm_auroc_mean"] < 0.6 and out["B_random_feature_combiner"]["auroc"] < 0.6
          and c["M1_shuffled_auroc_mean"] < 0.6 and c["M2_shuffled_auroc_mean"] < 0.6)
    print("\nGATE:", "PASS ✓ (controls collapse — real numbers are not leakage/artifact)"
          if ok else "FAIL ✗ (a control did NOT collapse — investigate leakage!)")
    print("saved -> results/negative_controls.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
