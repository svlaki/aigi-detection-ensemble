# AIGI Detection Ensemble (CS231n final)

Learned combiner over decorrelated AI-generated-image detectors + a LoRA
finetune on modern generators, evaluated honestly on OOD data.

See `AIGI_ensemble_build_spec.md` for the full phase-by-phase plan and
`AI_USAGE.md` for the AI-assistance log.

## Layout
```
src/         project code (config.py = single source of truth; embeddings/spectral/d3qe = member I/O)
scripts/     runnable scripts (smoke, downloads, per-member training, analysis, validation)
modal_app.py remote A10G jobs: run_m3 (D3QE), train_lora (M1a/M1b), validate_univfd
data/raw     downloaded images (pre-normalization)         [gitignored]
data/normalized  JPEG q95 + center-crop (Grommelt control) [gitignored]
cache/       per-image member features / logits            [gitignored]
models/      trained heads, LoRA adapters                  [gitignored]
results/     metrics tables (json/csv)
figures/     plots (fig2 decorrelation, fig3 combiner, fig4 LoRA)
manifests/   manifest.csv (image_id,path,label,source_dataset,generator_name,pool,split)
external/    vendored third-party repos: D3QE (M3), UniversalFakeDetect (validation)
notes/       provenance / method / version pins
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or requirements.lock.txt for exact pins
python src/config.py                    # prints config, checks pool disjointness
python scripts/smoke_clip.py            # Phase 0.4 gate
```

Dev box: Apple M3 (MPS) for smoke tests / data prep. Real training on a rented
single L4/A10G GPU. Code is device-portable (`cuda → mps → cpu`).

## Reproduce the dataset from scratch
The ~12k normalized images and the D3QE/LlamaGen checkpoints are **not** in the
repo (gitignored, large/third-party). Rebuild them with the scripts below.

> **Prereq (required for every step that downloads):** python.org Python's
> `urllib` has no CA store, so set certifi as the bundle first:
> ```bash
> export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')
> export HF_HUB_DISABLE_XET=1            # avoids occasional xet stream errors
> ```

```bash
# 0. setup (see above): venv + pip install + smoke_clip.py gate

# 1. Download + JPEG-normalize all sources -> data/normalized/ + manifests/manifest.csv
#    (~12k imgs: GenImage A/B fakes, ImageNet reals, modern Flux/SD3.5/MJ + FFHQ/COCO,
#     and a low-diversity CF head that step 2 replaces). Resumable; safe to re-run.
python scripts/download_data.py

# 2. Replace the CF rows with a diversity-aware sample (CompEval is 206 GB / grouped
#    by generator, so we read a spread of shards; ~12-min budget, ~15 OOD generators).
python scripts/repull_cf.py

# 3. Annotate the manifest with the train/eval `split` column (member_train /
#    combiner_fit / eval / lora_train) + assert no LoRA-train image leaks into eval.
python scripts/make_lora_split.py

# (optional) M3 detector triage — clones D3QE + downloads its checkpoints (~380 MB)
git clone --depth 1 https://github.com/Zhangyr2022/D3QE external/D3QE
python scripts/triage_d3qe.py
```

Exact dataset/checkpoint sources + version notes: `notes/data_sources.md`.
Tip: to skip the redundant CF pull in step 1, run step 1 with
`--only` listing the non-`cf_eval` source keys, then run step 2.

## Training/eval roles (`split` column — authoritative)
`pool` is data ORIGIN; `split` is the ROLE. **The eval set is `split=='eval'`, not
raw `pool=='C'.**

| `split` | role | source |
|---------|------|--------|
| `member_train` | train frozen M1 probe (M1a) + M2 | Pool A (GenImage) |
| `combiner_fit` | fit combiner + calibration | Pool B (GenImage) |
| `lora_train` | LoRA fine-tune of M1b **only** (never evaluated) | modern slice |
| `eval` | held-out evaluation (15 OOD generators) | CF + `modern_test` |

**Headline experiment (LoRA is CORE):** M1 has two versions — **M1a** frozen CLIP +
linear probe, and **M1b** LoRA-fine-tuned CLIP — compared on the held-out
`modern_test`. Built/asserted by `scripts/make_lora_split.py`. See `notes/method.md`.

## Remote GPU (Modal) — required for M3 and LoRA
D3QE inference (M3) and the LoRA fine-tune are too slow on the dev box, so they run
on a rented A10G via [Modal](https://modal.com). One-time setup:
```bash
pip install modal                                  # already in requirements.txt
./.venv/bin/modal setup                            # browser auth → shared "cs-231n" workspace
./.venv/bin/modal profile activate cs-231n
# upload the normalized images + manifest to the shared volume (idempotent):
./.venv/bin/modal volume create cs231n-data        # skip if it already exists
./.venv/bin/modal volume put cs231n-data ./data/normalized/ /normalized/
./.venv/bin/modal volume put cs231n-data ./manifests/manifest.csv /manifests/manifest.csv
```
The volume mounts at `/data` inside Modal functions (`/data/normalized/...`,
`/data/manifests/manifest.csv`). Jobs are serverless/per-second-billed; whole project
≈ a few $ of A10G time. `modal_app.py` is self-documenting (docstrings per function).

## Run the modeling pipeline (Phases 3–6)
Each step caches to `cache/` keyed by `image_id`, so later steps reuse with no
recompute. All local steps are CPU/MPS and fast; Modal steps need the volume above.
```bash
# --- Phase 3: members ---
# M1 (CLIP) embeddings for ALL splits  -> cache/clip_*_emb.npz   (~20 min MPS)
SSL_CERT_FILE=$(./.venv/bin/python -c 'import certifi;print(certifi.where())') \
  ./.venv/bin/python scripts/extract_clip_embeddings.py
./.venv/bin/python scripts/train_m1a.py            # M1a probe + in-dist gate (≥0.90)
./.venv/bin/python scripts/extract_spectral_features.py   # M2 FFT feats (~1 min CPU)
./.venv/bin/python scripts/train_m2.py             # M2 probe + in-dist gate (≥0.75)
./.venv/bin/modal run modal_app.py                 # M3 = D3QE on A10G -> cache/m3_d3qe_logit.npz
./.venv/bin/python scripts/build_member_outputs.py # refit M1a/M2 on full member_train -> cache/member_outputs.parquet

# --- Phase 4: decorrelation (Figure 2) ---
./.venv/bin/python scripts/decorrelation.py        # results/decorrelation_*, figures/fig2_decorrelation.png

# --- Phase 5: calibration + combiner + baselines (gradeable milestone) ---
./.venv/bin/python scripts/combiner.py             # results/phase5_*, models/combiner_*, figures/fig3_results.png

# --- Phase 6: LoRA ablation (HEADLINE, Figure 4) ---
./.venv/bin/modal run modal_app.py::lora_main      # train M1a/M1b on A10G -> results/phase6_metrics.json, cache/m1b_logits.npz
./.venv/bin/python scripts/phase6_report.py        # figures/fig4_lora.png + per-class deltas
./.venv/bin/python scripts/combiner_with_m1b.py    # optional: fold M1b into combiner

# --- Validation (evidence the numbers are real) ---
./.venv/bin/python scripts/negative_controls.py        # label-shuffle/noise must collapse to ~0.5
./.venv/bin/python scripts/benchmark_genimage_xgen.py  # GenImage cross-generator (regime reproduction)
./.venv/bin/modal run modal_app.py::univfd_main        # EXACT UnivFD reproduction (official ckpt + data)
```

## Current status & next steps (for the next contributor)
**Done:** Phases 0–6 complete + a full validation suite. All members trained, combiner
+ calibration fit, LoRA ablation run, results/figures generated. Headline numbers live
in `results/` (see `phase5_results.csv`, `phase6_metrics.json`, `phase6_combiner_fold.json`).

Key results (held-out `eval`, modern/CF OOD):
- Combiner beats best single member and naive baselines; main win is **class balance**.
- LoRA (M1b) ≥ M1a on AUROC across all sets; **folding M1b into the combiner** lifts
  eval acc +0.124 (eval_modern +0.195) — the central "training-data diversity" result.
- Validation: negative controls collapse to ~0.5 (no leakage); our CLIP probe reproduces
  the GenImage cross-generator regime (0.845); and we **exactly reproduce** UnivFD's
  published per-domain AP (<1.1 pt; mean 95.66 vs 95.00) with their official checkpoint.

**Remaining = Phase 7 (writing):** freeze experiments, assemble the 4 figures + per-class
tables into the report. Method notes for the writeup: `notes/method.md`. Honesty caveats
to carry over are listed there. The combiner's AUROC ≈ best member (its gain is calibrated
balance, not ranking) and the LoRA fold conflates "modern data + LoRA" — state both plainly.
