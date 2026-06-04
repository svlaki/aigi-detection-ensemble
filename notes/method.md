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
- **M1b** — CLIP ViT-L LoRA-fine-tuned (peft, rank 8–16) on `lora_train` (modern
  Flux/SD3.5/MJ slice) + a head.

The question: does adapting to modern generators via LoRA recover detection on modern
commercial generators that the frozen probe misses? Report before/after per-class on
`modern_test`. Image-disjoint split (same modern generators in train and test, different
images) — asserted leakage-free by `scripts/make_lora_split.py` and `manifest.validate`.

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
