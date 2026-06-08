"""Modal app — remote GPU jobs against the cs231n-data volume.

Currently: M3 / D3QE inference (Phase 3). D3QE runs a full CLIP ViT-L/14 + a
VQ-VAE per image, which is ~10h on the dev-box CPU — so we run it on an A10G.

The volume `cs231n-data` holds:
  /data/normalized/<subdir>/<img>.jpg   (uploaded normalized images)
  /data/manifests/manifest.csv          (the manifest; `path` col = "data/normalized/...")

This app ships D3QE's code + pretrained weights into the image, reads the manifest
from the volume, runs inference on GPU, writes the logit cache back to the volume
(/data/cache/m3_d3qe_logit.npz), and returns the {image_id -> logit} dict so the
local entrypoint can drop it into ./cache/ for the rest of the pipeline.

Run:
  ./.venv/bin/modal run modal_app.py            # all manifest images
  ./.venv/bin/modal run modal_app.py --limit 64 # smoke test

Reproduction note: matches the authors' validate.py exactly — load checkpoint,
.eval(), model(x).sigmoid(); `freq_log_counter` left at its loaded default.
"""
import modal

app = modal.App("cs231n-m3-d3qe")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "numpy", "pandas", "pillow",
        "ftfy", "regex", "tqdm", "setuptools", "certifi",
    )
    # ship D3QE code + pretrained weights (vq_ds16_c2i.pt 275MB, model_epoch_best 88MB)
    .add_local_dir("external/D3QE/networks", "/d3qe/networks", copy=True)
    .add_local_dir("external/D3QE/pretrained", "/d3qe/pretrained", copy=True)
)

vol = modal.Volume.from_name("cs231n-data")


CACHE_REMOTE = "/data/cache/m3_d3qe_logit.npz"


def _load_remote_cache() -> dict:
    import os
    import numpy as np
    if not os.path.exists(CACHE_REMOTE):
        return {}
    data = np.load(CACHE_REMOTE, allow_pickle=False)
    return {str(i): float(data["logit"][k]) for k, i in enumerate(data["image_id"])}


def _save_remote_cache(logits: dict) -> None:
    import os
    import numpy as np
    os.makedirs("/data/cache", exist_ok=True)
    ids = np.array(list(logits.keys()), dtype=np.str_)
    lg = np.array([logits[i] for i in logits], dtype=np.float32)
    np.savez(CACHE_REMOTE + ".tmp.npz", image_id=ids, logit=lg)
    os.replace(CACHE_REMOTE + ".tmp.npz", CACHE_REMOTE)
    vol.commit()


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=7200)
def run_m3(batch_size: int = 64, limit: int = 0, checkpoint: int = 2000) -> dict:
    """D3QE inference over the manifest. Resumable: reloads the volume cache and
    skips already-done image_ids, checkpointing every `checkpoint` new images."""
    import os
    import sys
    import time

    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # clip.load urllib download

    import numpy as np
    import pandas as pd
    import torch
    import torchvision.transforms as T
    from PIL import Image

    sys.path.insert(0, "/d3qe")
    from networks.D3QE import D3QE

    device = "cuda"
    print(f"[m3-modal] torch {torch.__version__} | cuda={torch.cuda.is_available()}")

    model = D3QE(vqvae_path="/d3qe/pretrained/vq_ds16_c2i.pt")
    state = torch.load("/d3qe/pretrained/model_epoch_best.pth", map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    info = model.load_state_dict(sd, strict=False)
    missing_nb = [k for k in info.missing_keys
                  if not (k.startswith("vq_model") or k.startswith("clip_model"))]
    assert not missing_nb, f"missing non-backbone keys: {missing_nb[:10]}"
    assert not info.unexpected_keys, f"unexpected keys: {list(info.unexpected_keys)[:10]}"
    model = model.to(device).eval()
    print("[m3-modal] model loaded.")

    df = pd.read_csv("/data/manifests/manifest.csv")
    if limit:
        df = df.head(limit)

    logits = _load_remote_cache()
    print(f"[m3-modal] resume: {len(logits)} already cached on volume")
    todo = df[~df["image_id"].astype(str).isin(logits.keys())].reset_index(drop=True)
    ids = [str(x) for x in todo["image_id"].tolist()]
    paths = ["/data/" + p.removeprefix("data/") for p in todo["path"].tolist()]
    print(f"[m3-modal] images: total={len(df)} pending={len(paths)} | batch={batch_size}")

    to_tensor = T.Compose([T.Resize((256, 256)), T.ToTensor()])  # [0,1], model normalizes internally
    since_ckpt = 0
    t0 = time.time()
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        cids = ids[start:start + batch_size]
        batch = torch.stack([to_tensor(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        with torch.no_grad():
            lg = model(batch).float().cpu().numpy().reshape(-1)
        for c, v in zip(cids, lg):
            logits[c] = float(v)
        since_ckpt += len(chunk)
        n = start + len(chunk)
        rate = n / max(time.time() - t0, 1e-6)
        print(f"[m3-modal] {n}/{len(paths)} ({rate:.1f} img/s, "
              f"eta {(len(paths)-n)/max(rate,1e-6)/60:.1f}m)", flush=True)
        if since_ckpt >= checkpoint:
            _save_remote_cache(logits)
            print(f"[m3-modal]   ✓ checkpoint to volume ({len(logits)} total)", flush=True)
            since_ckpt = 0

    _save_remote_cache(logits)
    arr = np.array([logits[i] for i in logits], dtype=np.float32)
    print(f"[m3-modal] DONE — {len(logits)} logits in {(time.time()-t0)/60:.1f}m")
    print(f"[m3-modal] logit stats: min={arr.min():.3f} max={arr.max():.3f} "
          f"mean={arr.mean():.3f} std={arr.std():.3f} nan={bool(np.isnan(arr).any())}")
    return logits


@app.local_entrypoint()
def main(batch_size: int = 32, limit: int = 0):
    import sys
    from pathlib import Path

    logits = run_m3.remote(batch_size=batch_size, limit=limit)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src import d3qe
    path = d3qe.save_cache(logits)
    print(f"[m3-local] saved {len(logits)} logits -> {path}")


if __name__ == "__main__":
    print("Run with:  ./.venv/bin/modal run modal_app.py")
