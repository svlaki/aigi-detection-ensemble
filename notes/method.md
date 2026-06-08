# Method notes (for the report's methods section)

## The two orthogonal "threes" (don't conflate them)

**Three members = three different KINDS of detector** (decorrelated by architecture,
NOT by what data they saw):
- **M1** — CLIP ViT-L features → semantic signal. (Two versions: M1a frozen probe, M1b LoRA.)
- **M2** — FFT/spectral features → frequency-artifact signal.
- **M3** — pretrained D3QE (frozen) → autoregressive quantization-error signal.

M1a and M2 both train on the SAME slice (member_train / Pool A). M3 isn't trained.
Decorrelation comes from the different inductive biases — measured by the Phase-4
error-correlation matrix (Figure 2). Combining helps only if members err differently.

**Three+ pools = different generator slices, split by STAGE** (so nothing leaks):
- `member_train` (Pool A) → fit members
- `combiner_fit` (Pool B) → fit combiner + calibration
- `eval` (Pool C: CF + modern_test) → held-out OOD evaluation
- `lora_train` (modern slice) → LoRA fine-tune only, never evaluated

`pool` = ORIGIN; `split` = ROLE. Eval set = `split=='eval'`.

## Supervision
One signal everywhere: `label` 0=real / 1=fake. `generator_name` is NOT a training
target — it only defines disjoint pools and enables per-generator analysis.

## The LoRA ablation (CORE / headline)
M1 has two versions, compared on the held-out `modern_test`:
- **M1a** — frozen CLIP ViT-L + linear probe, trained on `member_train` (old GenImage
  generators). The cheap baseline.
- **M1b** — CLIP ViT-L LoRA-fine-tuned on `lora_train` (modern Flux/SD3.5/MJ slice) + a head.

**LoRA config actually used** (`modal_app.py::train_lora`): peft `r=16, lora_alpha=32,
lora_dropout=0.05, bias="none"`, `target_modules=["attn.out_proj","mlp.c_fc","mlp.c_proj"]`
on the CLIP visual encoder (QKV is a fused `nn.MultiheadAttention` in_proj — not
peft-targetable, so attention is adapted only via out_proj; documented limitation).
1.53% of params trainable. Matched ablation: M1a and M1b use the SAME linear head,
optimizer (AdamW, head lr 1e-3 / LoRA lr 1e-4), epochs (10), and seed — the only
difference is whether the backbone adapts. Both trained in one Modal job for identical
conditions.

The question: does adapting to modern generators via LoRA recover detection on modern
commercial generators that the frozen probe misses? Report before/after per-class on
`modern_test`. Image-disjoint split (same modern generators in train and test, different
images) — asserted leakage-free by `scripts/make_lora_split.py` and `manifest.validate`.

**Result framing (honest):** on `modern_test` (in-generator, held-out images) the frozen
probe is already near-saturated, so the LoRA delta is small; the meaningful gains are
on `cf` (cross-generator) and `eval` (AUROC +0.009/+0.017), driven mainly by improved
REAL detection (a small fake-sensitivity trade-off). The bigger effect is the **combiner
fold**: swapping M1b for M1a in the ensemble lifts eval acc +0.124 (eval_modern +0.195).
NOTE this fold conflates two changes (modern training data + LoRA) — frame it as
"ensemble with vs without a modern-adapted member," not "LoRA architecture alone."

## Caching philosophy
Each image → features/logits ONCE (the GPU pass). Every fit (M1a head, M2, combiner,
calibration) then runs on cached numbers in seconds on CPU. Combiner iterates without
re-running CLIP/D3QE.

## Honesty caveats to carry into the writeup
- Small subsampled sets → wide CIs.
- Self-built members aren't literal SOTA checkpoints; the method + analysis is the contribution.
- JPEG-normalized (q95/256) to control the Grommelt confound, but real-vs-generator
  source mismatch remains.
- Pool B has GenImage `midjourney`; eval has modern `midjourney_modern` — different
  vintages/datasets, same family. A↔B disjointness (enforced) is clean; note this footnote.
- D3QE eval self-zeroes its codebook freq-diff signal (freq_log_counter starts at 0) —
  its own eval behavior, not our change.
- Combiner AUROC ≈ best single member (M1) — its value is calibrated, balanced accuracy
  at the operating point, NOT improved ranking. Say this plainly.

## Headline numbers (held-out `eval` = CF + modern_test, OOD) — for the results section
Per-member AUROC drops in-dist→OOD: M1 0.999→0.753, M2 0.974→0.517, M3 0.891→0.642
(in-dist row is in-sample for M1/M2; honest in-dist gates are M1a 0.977 / M2 0.925).
Error-correlation on eval fakes ≈ 0.037 (near-independent); oracle coverage 0.992.
Combiner (LogReg) eval: acc 0.674 / AUROC 0.733, real 0.729 / fake 0.619 — beats best
member (0.622) and baselines (mean 0.644, majority 0.594). All in `results/phase5_results.csv`.
LoRA fold (combiner with M1b): eval acc 0.798 / AUROC 0.838; eval_modern acc 0.863.

## Validation evidence (so the numbers are defensible)
- **Negative controls** (`results/negative_controls.json`): combiner real 0.733 →
  label-shuffled 0.475, noise-feature 0.49, member shuffles ≈0.5 → no leakage.
- **GenImage cross-generator** (`results/benchmark_genimage_xgen.{csv,json}`): our CLIP
  probe off-diagonal AUROC 0.845 (in-generator 0.998) — reproduces the published regime.
- **UnivFD exact reproduction** (`results/univfd_reproduction.json`): official
  fc_weights on official diffusion test set → per-domain AP within <1.1 pt of the paper
  (mean 95.66 vs 95.00). Validates the CLIP-probe family our M1 is built on.
- D3QE (published third-party detector) behaves sanely on eval; CF is an external benchmark.
