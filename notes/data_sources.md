# Data sources & provenance (pin these for citations)

All identified on HuggingFace on **2026-06-03**. Subset targets in `src/config.py`.

## GenImage (Pool A/B source) — per-generator, generator_name guaranteed
`bitmind/` hosts one dataset per GenImage generator:
- `bitmind/GenImage_ADM`
- `bitmind/GenImage_BigGAN`
- `bitmind/GenImage_glide`
- `bitmind/GenImage_MidJourney`
- `bitmind/GenImage_wukong`
- `bitmind/GenImage_VQDM`

Probe (`GenImage_ADM`, streaming): column `image` only, 256×256, **mode RGBA**
(→ convert to RGB on normalize). Appears **fake-only** per generator; GenImage
pairs fakes with ImageNet reals → need a reals source (ImageNet subset) to match.
TODO: confirm whether bitmind splits include reals/labels or are fakes-only.
Alt full mirror: `jzousz/GenImage` (dl≈1000).

Pool A generators (placeholder): sd14, adm, glide, biggan
Pool B generators (placeholder): midjourney, wukong, vqdm
(NOTE: bitmind has no SD1.4 dataset visible — may swap to available generators.)

## Community Forensics (Pool C — terminal in-the-wild)
- `OwensLab/CommunityForensics`        (full)
- `OwensLab/CommunityForensics-Small`  (~278K "Small" variant)
- `OwensLab/CommunityForensics-Eval`   ← **split name is `CompEval`** (not `train`)

TODO: confirm label + generator columns; pin the revision hash in this file.

## Modern generators (Pool C + LoRA data) — self-collected slice
- Flux: `lehduong/flux-coco-generated` is **GATED** (skip). Ungated candidates:
  `LukasT9/Flux-1-Dev-Images-1k`, `LukasT9/Flux-1-Schnell-Images-1k`.
- SD3.5: `momodawoud/sd3.5_generated_examples_hard_medium`, `Sarim-Hash/sd3.5_generated_images`.
- Midjourney: `davidmunechika/midjourney-images` had only **8 images** (dead). Swapped to
  `ehristoforu/midjourney-images` (2456 imgs, ungated, `image` col 864x1728) -> 750 pulled.
- CF (`CompEval`) is 51,836 rows / **206 GB**, generators grouped by shard -> streaming the
  head gives only DFGAN/StableCascade. Fix: `scripts/repull_cf.py` reads a spread of 30 of
  the 413 shards with a per-generator cap + 12-min budget for diversity.
- Reals (COCO/FFHQ): TODO pick ungated mirrors.

## Optional
- Synthbuster: `marco-willi/synthbuster-plus` (Phase 1.4, only if time).

## External model checkpoints pulled (for citations)
| Asset | Source | Size | Local path |
|-------|--------|------|-----------|
| open_clip ViT-L/14 (openai) | https://github.com/mlfoundations/open_clip | ~1.7GB | HF cache |
| CLIP ViT-L-14.pt (orig OpenAI, for D3QE) | openaipublic.azureedge.net (via clip.load) | 890MB | ~/.cache/clip |
| D3QE classifier `model_epoch_best.pth` | https://huggingface.co/Yanran21/D3QE | 92MB | external/D3QE/pretrained |
| LlamaGen VQ-VAE `vq_ds16_c2i.pt` | https://huggingface.co/FoundationVision/LlamaGen | 288MB | external/D3QE/pretrained |
| D3QE code (ICCV 2025) | https://github.com/Zhangyr2022/D3QE — arXiv:2510.05891 | — | external/D3QE |

## Environment gotcha
python.org Python 3.13's `urllib` lacks a CA store → SSL verify fails on
azureedge downloads. Fix: `export SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())')`
before anything that uses `clip.load` / urllib. (`requests`/`huggingface_hub` already use certifi.)
