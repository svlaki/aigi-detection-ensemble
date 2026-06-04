# AIGI Detection Ensemble (CS231n final)

Learned combiner over decorrelated AI-generated-image detectors + a LoRA
finetune on modern generators, evaluated honestly on OOD data.

See `AIGI_ensemble_build_spec.md` for the full phase-by-phase plan and
`AI_USAGE.md` for the AI-assistance log.

## Layout
```
src/        project code (config.py = single source of truth)
scripts/    runnable scripts (smoke tests, downloads, training)
data/raw    downloaded images (pre-normalization)        [gitignored]
data/normalized  JPEG q95 + center-crop (Grommelt control) [gitignored]
cache/      per-image member features / logits           [gitignored]
models/     trained heads, LoRA adapters                 [gitignored]
results/    metrics tables
figures/    plots
manifests/  manifest.csv (image_id,path,label,source_dataset,generator_name,pool,split)
external/   cloned third-party repos (D3QE, ...)          [gitignored]
notes/      provenance / version pins
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
