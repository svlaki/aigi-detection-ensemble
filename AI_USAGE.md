# AI Usage Log (honor-code documentation)

This file logs all AI-assisted code/work for the CS231n final project, per the
honor-code requirement. Tool: **Claude Code** (Anthropic). The human authors
direct the work, review all code, and write the report; Claude writes code and
runs commands as logged below.

Format: dated entries, what was done, and the key commands run.

---

## 2026-06-03 (Wed evening) — Phase 0 + start of Phase 1

**Environment:** dev box is an Apple M3 Mac, 8 GB RAM, ~44 GB free disk, no
NVIDIA GPU (MPS available). Real training to run later on a rented L4/A10G.
Python 3.13.1, isolated `.venv`.

### Phase 0.3 — Repo skeleton + config
- Claude created directories: `data/{raw,normalized}`, `cache`, `models`,
  `results`, `figures`, `report`, `src`, `scripts`, `external`, `manifests`,
  `notes`.
- Wrote `src/config.py`: paths, `SEED=1337`, `set_seed()`, portable
  `get_device()` (cuda→mps→cpu), CLIP backbone choice, JPEG/size normalization
  constants, placeholder Pool A/B/C generator lists + `assert_no_pool_leakage()`,
  manifest schema.
- Wrote `.gitignore` (excludes data/cache/models/external + large binaries).

### Phase 0.1–0.2 — Python env
- `python3 -m venv .venv`
- `pip install -r requirements.txt`  (torch 2.12.0, torchvision 0.27.0,
  open_clip_torch 3.3.0, transformers 5.10.1, peft 0.19.1, scikit-learn 1.9.0,
  numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, pyarrow 24.0.0, datasets 4.8.5,
  huggingface_hub 1.17.0, pillow 12.2.0, matplotlib 3.10.9, seaborn 0.13.2,
  tqdm 4.67.3)
- Pinned exact versions to `requirements.lock.txt` (`pip freeze`).

### Phase 0.4 — CLIP smoke test  ✓ GATE PASSED
- Wrote `scripts/smoke_clip.py`: synthesizes 10 deterministic images, runs
  `open_clip` ViT-L/14 end-to-end.
- `python scripts/smoke_clip.py` → embeddings shape `(10, 768)`, no NaNs,
  ~0.23 s/img on MPS. Gate (dim==768, no errors) **PASS**.
- Noted QuickGELU mismatch warning for the `openai` tag; switched
  `CLIP_MODEL` to `ViT-L-14-quickgelu` so M1 uses the exact OpenAI activation.

### Phase 1 (start) — data source identification + M3 triage
- Searched HF Hub for datasets. Found per-generator GenImage mirrors under
  `bitmind/GenImage_*` (generator_name guaranteed), `OwensLab/CommunityForensics{,-Small,-Eval}`,
  and modern-generator candidates. Recorded all in `notes/data_sources.md`.
- Streaming schema probe (1 sample/source) surfaced 3 gotchas before any long
  download: GenImage images are **RGBA** (convert→RGB) and per-generator =
  fakes-only; CF-Eval split is **`CompEval`** not `train`; `lehduong/flux-coco-generated`
  is **gated** (dropped). Full overnight download NOT launched — awaiting OK
  (long job; check-in per project rules).

### Phase 1 — download/normalize/manifest pipeline + overnight launch
- Locked 3v3 generator-disjoint split (no SD1.4 hunt): Pool A {adm,biggan,glide},
  Pool B {midjourney,wukong,vqdm} (config.py).
- Probed all sources (`scripts/probe_sources.py`) → finalized IDs in
  `notes/data_sources.md`. Confirmed: GenImage `bitmind/GenImage_*` (RGBA, fakes),
  ImageNet reals `evanarlian/imagenet_1k_resized_256`, CF `OwensLab/CommunityForensics-Eval`
  (split `CompEval`, image as `image_data` bytes, per-row `label`+`model_name`),
  modern fakes Flux `LukasT9/Flux-1-Dev-Images-1k` / SD3.5
  `momodawoud/sd3.5_generated_examples_hard_medium` / MJ `davidmunechika/midjourney-images`,
  modern reals FFHQ `merkol/ffhq-256` + COCO `detection-datasets/coco`.
- Wrote `src/normalize.py` (RGB + center-crop + resize 256 + JPEG q95),
  `src/manifest.py` (pool assignment + Phase 1.6/2 gate validation),
  `scripts/download_data.py` (streaming, resumable, balanced, bytes/PIL decode).
- Smoke-tested 4 sources × 3 imgs end-to-end: manifest correct, gate problems NONE.
- Launched FULL streaming subsample in background (~13k imgs target):
  Pool A 1500f+1500r, Pool B 1500f+1500r, CF 1250+1250, modern 2250f+2250r.
  Env: `SSL_CERT_FILE=certifi`. Will stop at Phase 1.6 manifest gate to report.

### 2026-06-04 — Plan update: LoRA promoted to CORE + modern-slice split
- Per user: LoRA fine-tune is now a core deliverable. M1 has TWO versions —
  M1a (frozen CLIP + linear probe) and M1b (LoRA-fine-tuned CLIP); the headline
  experiment is the M1a-vs-M1b ablation on the held-out `modern_test`.
- NO Phase 1 rebuild. Annotated the EXISTING manifest in place with a new `split`
  column (`scripts/make_lora_split.py`): member_train (A) / combiner_fit (B) /
  eval (CF + modern_test) / lora_train (modern slice). Backed up prior manifest
  to `manifests/manifest.prelora.bak.csv`.
- Modern slice split image-disjoint, stratified by generator/source:
  lora_train = 1500f (flux/sd35/mj 500 each) + 1500r; modern_test = 750f + 750r.
  Asserted: no lora_train image_id in any eval row ✓ (also enforced in
  `src/manifest.validate`). Eval set now 1652r/1687f, 15 distinct generators.
- Confirmed `peft` 0.19.1 installed.
- Updated `AIGI_ensemble_build_spec.md` (THESIS, Phase 1.6 manifest, Phase 2 split
  gate, Phase 3 M1 two-version, Phase 6 CORE, figures, fallbacks), `README.md`,
  and added `notes/method.md`.
- Did NOT start Phase 2/3 modeling (per user).

### Phase 3 M3 — D3QE triage  ✓ GATE PASSED → 3-member ensemble
- Cloned `github.com/Zhangyr2022/D3QE` (ICCV 2025, arXiv:2510.05891) into `external/`.
- D3QE ships no classifier in-repo; downloaded required checkpoints:
  `Yanran21/D3QE/model_epoch_best.pth` (92MB) + `FoundationVision/LlamaGen/vq_ds16_c2i.pt`
  (288MB) into `external/D3QE/pretrained`.
- Wrote `scripts/triage_d3qe.py`: builds D3QE on CPU, loads weights, runs 10
  synthetic 256×256 images. Patched the repo's hardcoded `.cuda()` path by
  running on CPU; fixed urllib SSL (set `SSL_CERT_FILE` to certifi so the
  vendored `clip.load` could fetch the original OpenAI `ViT-L-14.pt`, 890MB).
- Result: clean state_dict load (0 unexpected, 0 missing non-backbone keys),
  no NaNs, varied logits. **VERDICT: SANE → 3-member (M1+M2+M3).**
- Observed quirk to note in report: at eval `freq_log_counter` starts at 0 so
  the codebook frequency-difference signal is zeroed (`<200` gate) — this is
  D3QE's own eval behavior, not a change we made.

---

## External repos / checkpoints pulled in (track for citations)

| Item | Source | Purpose | Notes |
|------|--------|---------|-------|
| open_clip ViT-L/14 (`openai`) | https://github.com/mlfoundations/open_clip | M1 CLIP-linear backbone | use `ViT-L-14-quickgelu` |
| D3QE code (ICCV 2025) | https://github.com/Zhangyr2022/D3QE (arXiv:2510.05891) | M3 detector | cloned to external/D3QE |
| D3QE classifier weights | https://huggingface.co/Yanran21/D3QE | M3 weights | model_epoch_best.pth, 92MB |
| LlamaGen VQ-VAE | https://huggingface.co/FoundationVision/LlamaGen | D3QE tokenizer | vq_ds16_c2i.pt, 288MB |
| CLIP ViT-L-14.pt (OpenAI) | openaipublic.azureedge.net via clip.load | D3QE CLIP branch | 890MB, ~/.cache/clip |
| GenImage subsets | https://huggingface.co/bitmind | Pool A/B data | per-generator |
| Community Forensics | https://huggingface.co/OwensLab | Pool C data | pin revision when pulled |
