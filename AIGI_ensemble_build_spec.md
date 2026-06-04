# AIGI Detection Ensemble — Step-by-Step Build Spec (48h)

**Goal:** A learned combiner over decorrelated AI-generated-image detectors, plus a LoRA finetune on modern generators, evaluated honestly on out-of-distribution data.
**Deadline:** Friday night. **Start:** Wed evening. **Compute:** ~$20 of $400 (time is the constraint). **Executor:** Claude Code + 2 people.

> **How to use this doc:** Hand it to Claude Code phase by phase. Each step has an action, a checkpoint (✓), and a fallback. Do not skip the ✓ gates — they prevent reporting garbage. Understand each piece as it's built; you have to defend this in the report.

---

## THESIS (locked — one falsifiable question)

> Does a **learned combiner over decorrelated detectors** reduce the *fake-as-real collapse* on out-of-distribution fakes, relative to the best single member — and does a **LoRA finetune on a modern-generator slice** recover detection on modern commercial generators?

A negative result ("members were correlated → combiner ≈ best member") is still complete and gradeable. Do **not** claim to beat Community Forensics or "solve" generalization.

> **UPDATE (LoRA promoted to CORE):** The LoRA fine-tune is no longer an optional upside step — it is a core deliverable. The **headline experiment** is an M1 ablation: **(a) frozen CLIP + linear probe** vs **(b) LoRA-fine-tuned CLIP detector**, both evaluated on the held-out modern-generator set (`modern_test`). The modern slice is split image-disjoint into `lora_train` (LoRA only) and `modern_test` (held out; part of the eval set). No LoRA-train image may appear in any eval set.

---

## PHASE 0 — Setup & scope lock (Wed evening, ≤1.5h)

1. **Create env.** Python 3.10+. Install: `torch torchvision open_clip_torch transformers scikit-learn numpy scipy pandas matplotlib seaborn datasets pillow peft tqdm`.
2. **Provision one GPU.** Single L4 / A10G on Modal (per-second billing) or Colab. Do **not** spin up an A100 or multi-GPU.
3. **Repo skeleton.** `data/`, `cache/` (member logits), `models/`, `results/`, `figures/`, `report/`. One config file with paths, seeds, split definitions.
4. **Smoke test.** Extract a CLIP embedding for 10 images end-to-end and print shapes.
   - ✓ **Gate:** embeddings have expected dim (768 for ViT-L/14) and no errors.
5. **Kick off downloads now** (Phase 1) so they run overnight.

---

## PHASE 1 — Data acquisition & normalization (Wed night → Thu AM)

> Partner A owns this track (highest external-dependency risk).

1. **In-distribution set (Pool A/B source).** Pull a **GenImage** subset from a HuggingFace mirror (search "GenImage"). Subsample ~3k balanced per class. **Record the generator name** for every fake (needed for the split).
2. **Terminal in-the-wild bucket (Pool C).** Pull **Community Forensics test split** (`OwensLab/CommunityForensics` on HF). Subsample ~2–3k balanced. **Pin the checkpoint/version** in your notes (the repo changed through 2025; a 278K "Small" variant exists).
3. **Self-collected modern-generator slice (Pool C + LoRA data).** Pull **Flux / SD3.5 / Midjourney** fakes from open HF datasets + **reals from COCO/FFHQ**. Aim ~500–1000 per generator. This is the highest-value data: you control it, it can't get blocked, and it targets the modern-generator gap.
4. **(Optional) Synthbuster** (Zenodo) if time — known commercial generators, supports per-generator analysis.
5. **JPEG/size normalization (CRITICAL — Grommelt confound).** Recompress **every** image (real + fake, all sets) to **JPEG quality 95**, and center-crop/resize to a **common resolution** before any feature extraction. This stops detectors keying on JPEG/size shortcuts.
6. **Build a manifest.** One CSV/parquet: `image_id, path, label (0=real,1=fake), source_dataset, generator_name, pool (A/B/C), split (member_train/combiner_fit/eval/lora_train)`. `pool` = data ORIGIN; `split` = training/eval ROLE (authoritative). The eval set is `split==eval`, NOT raw `pool==C`, because the modern slice (origin C) is partitioned into `lora_train` and `modern_test`.
   - ✓ **Gate:** classes balanced per set; manifest has no missing generator names for fakes; all images are post-normalization.
   - **Fallback:** if any single dataset fights you past **Thu noon**, drop it (cut rule). Pool C can run on the self-collected slice alone.

> **DROPPED:** Chameleon — access is email-gated (request to the AIDE authors), not viable on this timeline. (Also note: a *different* "Chameleon" exists for AI-generated video — don't cross-cite.)

---

## PHASE 2 — Generator-disjoint split (Thu AM, ≤30min)

Assign generators to three **disjoint** pools — no generator's images cross pools:

- **Pool A (train members):** subset of GenImage generators, e.g. {SD1.4, ADM, GLIDE, BigGAN} + matched reals.
- **Pool B (fit combiner + calibrate):** *different* GenImage generators, e.g. {Midjourney, Wukong, VQDM} + reals.
- **Pool C (final eval only — never touched in training):** Community Forensics test + self-collected modern generators (Flux/SD3.5/MJ).

   - ✓ **Gate (no leakage):** assert `set(generators in A)` ∩ `set(generators in B)` = ∅; C is inherently disjoint.

**Modern-slice split (for the LoRA ablation).** Partition the self-collected modern slice (`source_dataset==modern_self`) image-disjoint, stratified by generator (fakes) / source (reals):
- **`lora_train`** — LoRA fine-tune ONLY; never evaluated. (500/gen fakes + 1500 reals)
- **`modern_test`** — held out; folded into the eval set for before/after. (250/gen fakes + 750 reals)
   - ✓ **Gate (no LoRA leakage):** assert `set(lora_train image_ids)` ∩ `set(eval image_ids)` = ∅. Implemented in `scripts/make_lora_split.py` + `src/manifest.validate`.

---

## PHASE 3 — Build the members (Thu AM–PM)

> Partner B owns M1/M2 (self-contained, guaranteed). Partner A owns M3 (external).

### M1 — CLIP detector, **TWO versions** (the ablation is the headline)
**M1a — frozen CLIP + linear probe (UnivFD-style) — cheap baseline, build first.**
1. Extract frozen **open_clip ViT-L/14** image embeddings for all images; cache to disk.
2. Train a **LogReg (or 1-layer MLP) head** on `split==member_train` (Pool A) embeddings.
   - ✓ **Gate:** M1a ≥ ~90% accuracy on an in-distribution (Pool A-style) held-out sample. If ~60%, preprocessing is broken — fix before proceeding.

**M1b — LoRA-fine-tuned CLIP detector (Phase 6) — the intervention.**
3. LoRA-fine-tune the CLIP backbone on `split==lora_train` (modern slice) with `peft` (rank 8–16, few epochs), + a classification head.
4. **Headline ablation:** evaluate M1a vs M1b on `modern_test` (and on the full eval set). Report before/after per-class. This is Figure 4 and a core result.
   - ✓ **Gate:** ablation uses the held-out `modern_test` (no LoRA-train leakage); report the delta honestly (may be modest).

### M2 — FFT/spectral — decorrelated, guaranteed
1. Per image: 2D FFT → magnitude spectrum → **azimuthally-averaged radial power spectrum** (1D) + high-freq energy ratios → feature vector.
2. Train a small classifier (LogReg / shallow MLP) on Pool A.
   - ✓ **Gate:** M2 clearly beats chance in-distribution (≥~75%). Different inductive bias from M1 by construction.

### M3 — real pretrained detector — upside member
1. Clone **D³QE** (`github.com/Zhangyr2022/D3QE`), load pretrained weights, run inference → per-image fake probability. (Autoregressive-specialized; uncovered by the zero-shot audit = your novelty hook.)
   - ✓ **Gate:** M3 roughly reproduces its paper's number on matched data; produces sane logits on 10 images **by Thu noon**.
   - **Fallback (in order):** a UnivFD or DRCT public checkpoint → else **drop to a 2-member ensemble (M1+M2)**. Project stays complete either way.

### Cache everything
- Write **per-image member outputs** (probability + raw logit) for all members to one keyed file (`image_id → {p1,logit1,p2,logit2,p3,logit3}`).
   - ✓ **Gate (cache integrity):** shapes correct, no NaNs, ids aligned to the manifest.

---

## PHASE 4 — Decorrelation analysis (Thu PM, ≤1h)

1. On **Pool C fakes** (and overall), compute the **pairwise error-correlation matrix** between members (correlate per-image error indicators and/or logits).
2. Interpret: low correlation → combiner can help; high correlation → expect combiner ≈ best member (still a finding).
   - This matrix is **Figure 2** and the intellectual core of the report.

---

## PHASE 5 — Combiner + calibration + baselines (Thu PM → Thu night)

1. **Per-member calibration** on Pool B: temperature scaling or a learned per-member threshold (captures the "high AUC / low accuracy at 0.5" free win).
2. **Learned combiner.** Features = `[p1,p2,p3, logit margins, pairwise agreement]`. Train **LogReg** on Pool B (then optionally a 2-layer MLP). Report both.
3. **Baselines.** Mean-probability ensemble + majority vote (on calibrated outputs), so the *learned* combiner's added value is isolated.
4. **Evaluate** every member + both baselines + the combiner on **Pool C** (report Pool B too).
   - ✓ **Milestone:** combiner project is complete/gradeable here — but the LoRA ablation (Phase 6) is now CORE and must also ship, not just the combiner.

---

## PHASE 6 — LoRA finetune (CORE — the M1a-vs-M1b ablation is the headline)

> **CORE component, not optional.** This is the centerpiece training experiment. Run it even if Phase 5 is only partially done.

1. Use the manifest split (already built): **`lora_train`** for training, **`modern_test`** held out for eval. Disjointness is asserted (`scripts/make_lora_split.py`).
2. **LoRA-finetune** M1's CLIP backbone (M1b) on `lora_train` using `peft` (rank 8–16, few epochs). Small data → minutes–1h on one GPU.
3. Report **before/after** = **M1a (frozen probe) vs M1b (LoRA)** on the held-out `modern_test`. This is the training-data-diversity intervention and the headline result (Figure 4).
4. **(If time)** fold the adapted M1b back into the combiner; show whether eval-set modern-generator detection improves.
   - ✓ **Gate:** before/after uses the held-out `modern_test`; assert no `lora_train` image leaks into eval; report the delta honestly (it may be modest).

---

## PHASE 7 — Figures, freeze, write (Fri)

**Freeze experiments Friday morning.** Reserve real hours for writing.

### Metrics (report per-class, ALWAYS)
- **Fake-accuracy AND real-accuracy separately** — never a single "overall" number (that's how the field hides the collapse).
- AUC + accuracy@0.5 + accuracy@calibrated threshold.
- In-distribution → OOD gap, explicitly quantified.

### Figures (4 is plenty)
1. Per-class accuracy heatmap: {M1, M2, M3, mean, vote, combiner} × {Pool B, CF-test, modern-gen}, fake/real split.
2. Error-correlation matrix on OOD fakes (Phase 4).
3. In-distribution vs OOD bar chart — the gap + how much the combiner closes it.
4. **(HEADLINE)** LoRA ablation — M1a (frozen probe) vs M1b (LoRA) before/after on the held-out `modern_test`, per-class.

### Report skeleton
- **Abstract** — combiner + LoRA, honest framing.
- **Intro/motivation** — the field's in-the-wild collapse; "no universal winner" (cite the zero-shot audit, with its citation-quality caveat).
- **Related work** — Community Forensics + audit + Grommelt + B-Free → the argument that **training-data composition dominates architecture**.
- **Method** — members, generator-disjoint split, combiner, calibration, LoRA.
- **Experiments** — datasets (with JPEG normalization), metrics.
- **Results** — the 4 figures + per-class tables.
- **Discussion** — frame the combiner as **lightweight composition of training-data diversity that already exists across separately-trained detectors**, an alternative to monolithic data-scaling (CF's 2.7M-image / 4,803-generator approach). Argue training-data diversity is **necessary but not sufficient** (CF still ~35–42% on the newest generators).
- **Limitations/caveats** — small subsampled sets (wide CIs); self-built members aren't literal SOTA checkpoints (method + analysis is the contribution); JPEG-normalized to control Grommelt but real-vs-generator source mismatch remains; numbers are your own; in-distribution inflated vs. in-the-wild.
- **Conclusion.**

---

## CUT LIST (do NOT attempt)
Agentic/LLM router · 4th/5th members · full GenImage download · reproducing the audit's whole harness · Chameleon · any repo not loading by **Thu noon** (drop, fall back).

## RISK FALLBACKS (decision points)
- M3 dead by Thu noon → 2-member ensemble.
- A dataset fights you past Thu noon → drop it; Pool C can be the self-collected slice alone.
- Phase 5 shaky → still run the LoRA ablation (M1a vs M1b is now CORE and is independent of the combiner). If forced to choose under time pressure, the M1a-vs-M1b ablation outranks the combiner polish.

## DIVISION OF LABOR
- **Partner A (external track):** datasets + JPEG normalization (Phase 1), M3 repo (Phase 3).
- **Partner B (self-contained track):** M1/M2, combiner, calibration, eval, figures, report scaffold.
- Both drive Claude Code. Cache logits early so the combiner iterates without re-running inference.

## ONE-LINE PRIORITY ORDER
Normalize data → M1+M2 working & sanity-passed → cache logits → correlation matrix → combiner+calibration+baselines (← gradeable here) → LoRA → figures → write.
