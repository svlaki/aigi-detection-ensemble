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

import os as _os

_d3qe_base = _os.path.join(_os.path.dirname(__file__), "external", "D3QE")
_d3qe_available = (_os.path.isdir(_os.path.join(_d3qe_base, "networks"))
                   and _os.path.isdir(_os.path.join(_d3qe_base, "pretrained")))

_d3qe_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "torchvision", "numpy", "pandas", "pillow",
    "ftfy", "regex", "tqdm", "setuptools", "certifi",
)
if _d3qe_available:
    _d3qe_image = (_d3qe_image
                   .add_local_dir("external/D3QE/networks", "/d3qe/networks", copy=True)
                   .add_local_dir("external/D3QE/pretrained", "/d3qe/pretrained", copy=True))
image = _d3qe_image

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


# ===========================================================================
# Phase 6 — LoRA fine-tune: the M1a-vs-M1b ablation (headline, Figure 4)
# ===========================================================================
lora_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "open_clip_torch", "peft", "scikit-learn",
        "numpy", "pandas", "pillow", "ftfy", "regex", "certifi",
    )
)

M1B_LOGITS_REMOTE = "/data/cache/m1b_logits.npz"
LORA_TARGETS = ["attn.out_proj", "mlp.c_fc", "mlp.c_proj"]  # only peft-targetable Linears
SEED = 1337


@app.function(image=lora_image, gpu="A10G", volumes={"/data": vol}, timeout=7200)
def train_lora(epochs: int = 10, batch_size: int = 32, infer_batch: int = 128,
               r: int = 16, alpha: int = 32, dropout: float = 0.05,
               limit: int = 0) -> dict:
    """Matched ablation: train a linear head on lora_train with the CLIP visual
    encoder FROZEN (M1a) vs LoRA-adapted (M1b); everything else identical. Eval on
    modern_test (in-generator), CF (cross-generator), full eval. Returns metrics +
    M1b logits over combiner_fit ∪ eval (for the optional combiner fold)."""
    import os
    import time

    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # open_clip openai download

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import Image
    import open_clip
    from peft import LoraConfig, get_peft_model
    from sklearn.metrics import roc_auc_score, accuracy_score

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda"

    def vol_path(p: str) -> str:
        return "/data/" + (p[len("data/"):] if p.startswith("data/") else p)

    # ---- splits ----
    df = pd.read_csv("/data/manifests/manifest.csv")
    splits = {
        "lora_train":   df[df.split == "lora_train"],
        "modern_test":  df[(df.split == "eval") & (df.source_dataset == "modern_self")],
        "cf":           df[(df.split == "eval") & (df.source_dataset == "community_forensics")],
        "eval":         df[df.split == "eval"],
        "combiner_fit": df[df.split == "combiner_fit"],
    }
    if limit:  # smoke: subsample each split
        splits = {k: v.head(limit).reset_index(drop=True) for k, v in splits.items()}
    else:
        splits = {k: v.reset_index(drop=True) for k, v in splits.items()}

    lt_ids = set(splits["lora_train"].image_id.astype(str))
    for k in ("modern_test", "cf", "eval"):
        leak = lt_ids & set(splits[k].image_id.astype(str))
        assert not leak, f"LEAKAGE: {len(leak)} lora_train ids in {k}"
    print(f"[lora] splits: " + " ".join(f"{k}={len(v)}" for k, v in splits.items())
          + " | leakage gate ✓")

    # ---- model ----
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def l2norm(z):
        return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def preprocess_tensor(frame):
        xs = [preprocess(Image.open(vol_path(p)).convert("RGB")) for p in frame.path]
        return torch.stack(xs)

    # lora_train is the only set forwarded repeatedly -> preprocess once, keep on CPU
    X_lt = preprocess_tensor(splits["lora_train"])
    y_lt = torch.tensor(splits["lora_train"].label.to_numpy(np.float32))
    print(f"[lora] cached lora_train tensors: {tuple(X_lt.shape)}")

    @torch.no_grad()
    def embed(frame, backbone):
        backbone.eval()
        out = []
        for s in range(0, len(frame), infer_batch):
            xb = preprocess_tensor(frame.iloc[s:s + infer_batch]).to(device)
            out.append(l2norm(backbone(xb)).cpu())
        return torch.cat(out)

    def eval_split(frame, backbone, head):
        emb = embed(frame, backbone).to(device)
        with torch.no_grad():
            logit = head(emb).squeeze(-1).cpu().numpy()
        y = frame.label.to_numpy(int)
        p = 1 / (1 + np.exp(-np.clip(logit, -50, 50)))
        pred = (logit > 0).astype(int)
        return {
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "acc": float(accuracy_score(y, pred)),
            "real_acc": float(accuracy_score(y[y == 0], pred[y == 0])) if (y == 0).any() else float("nan"),
            "fake_acc": float(accuracy_score(y[y == 1], pred[y == 1])) if (y == 1).any() else float("nan"),
        }, dict(zip(frame.image_id.astype(str), logit.astype(float)))

    def train_head(backbone, lora: bool):
        """Train Linear(768,1) on lora_train; backbone frozen (M1a) or LoRA (M1b)."""
        torch.manual_seed(SEED)
        head = nn.Linear(768, 1).to(device)
        loss_fn = nn.BCEWithLogitsLoss()

        if lora:
            cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                             target_modules=LORA_TARGETS)
            backbone = get_peft_model(backbone, cfg)
            backbone.print_trainable_parameters()
            opt = torch.optim.AdamW(
                [{"params": head.parameters(), "lr": 1e-3},
                 {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": 1e-4}],
                weight_decay=1e-4)
        else:
            # frozen backbone -> precompute embeddings once, head-only training
            E = embed(splits["lora_train"], backbone).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

        n = len(X_lt)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                yb = y_lt[idx].to(device)
                if lora:
                    backbone.train()
                    emb = l2norm(backbone(X_lt[idx].to(device)))
                else:
                    emb = E[idx]
                logit = head(emb).squeeze(-1)
                loss = loss_fn(logit, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(idx)
            print(f"[lora] {'M1b' if lora else 'M1a'} epoch {ep+1}/{epochs} "
                  f"loss={tot/n:.4f}", flush=True)
        return backbone, head

    results, m1b_logits = {}, {}
    t0 = time.time()

    # ---- M1a (frozen) ----
    print("\n[lora] === M1a (frozen backbone + linear head) ===")
    _, head_a = train_head(model.visual, lora=False)
    for name in ("modern_test", "cf", "eval"):
        results[f"M1a/{name}"], _ = eval_split(splits[name], model.visual, head_a)

    # ---- M1b (LoRA) ----
    print("\n[lora] === M1b (LoRA backbone + linear head) ===")
    pv, head_b = train_head(model.visual, lora=True)
    for name in ("modern_test", "cf", "eval"):
        results[f"M1b/{name}"], lg = eval_split(splits[name], pv, head_b)
        if name == "eval":
            m1b_logits.update(lg)
    # combiner_fit logits too (for the optional fold)
    _, lg_b = eval_split(splits["combiner_fit"], pv, head_b)
    m1b_logits.update(lg_b)

    # ---- persist to volume ----
    os.makedirs("/data/cache", exist_ok=True)
    os.makedirs("/data/results", exist_ok=True)
    os.makedirs("/data/models", exist_ok=True)
    ids = np.array(list(m1b_logits.keys()), dtype=np.str_)
    lg = np.array([m1b_logits[i] for i in m1b_logits], dtype=np.float32)
    np.savez(M1B_LOGITS_REMOTE, image_id=ids, logit=lg)
    import json
    meta = {"epochs": epochs, "batch_size": batch_size, "r": r, "alpha": alpha,
            "dropout": dropout, "target_modules": LORA_TARGETS, "results": results}
    with open("/data/results/phase6_metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    try:
        pv.save_pretrained("/data/models/m1b_lora")
    except Exception as e:
        print(f"[lora] adapter save warning: {e}")
    vol.commit()

    print(f"\n[lora] DONE in {(time.time()-t0)/60:.1f}m")
    for k in ("modern_test", "cf"):
        a, b = results[f"M1a/{k}"], results[f"M1b/{k}"]
        print(f"[lora] {k}: M1a acc={a['acc']:.3f} auroc={a['auroc']:.3f} "
              f"(r{a['real_acc']:.2f}/f{a['fake_acc']:.2f})  ->  "
              f"M1b acc={b['acc']:.3f} auroc={b['auroc']:.3f} "
              f"(r{b['real_acc']:.2f}/f{b['fake_acc']:.2f})")
    return {"results": results, "m1b_logits": m1b_logits, "meta": meta}


@app.local_entrypoint()
def lora_main(epochs: int = 10, batch_size: int = 32, limit: int = 0,
              r: int = 16, alpha: int = 32):
    import json
    import sys
    from pathlib import Path

    out = train_lora.remote(epochs=epochs, batch_size=batch_size, limit=limit,
                            r=r, alpha=alpha)

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    import numpy as np
    (root / "results").mkdir(exist_ok=True)
    (root / "cache").mkdir(exist_ok=True)
    with open(root / "results" / "phase6_metrics.json", "w") as f:
        json.dump(out["meta"], f, indent=2)
    lg = out["m1b_logits"]
    np.savez(root / "cache" / "m1b_logits.npz",
             image_id=np.array(list(lg.keys()), dtype=np.str_),
             logit=np.array([lg[i] for i in lg], dtype=np.float32))
    print(f"[lora-local] saved phase6_metrics.json + m1b_logits.npz ({len(lg)} ids)")


@app.local_entrypoint()
def lora_sweep(limit: int = 0, batch_size: int = 32):
    """Sweep rank ∈ {4,8,16,32} × epochs ∈ {5,10}, alpha=2×rank.
    Skips r=16/e=10 (already in phase6_metrics.json). Runs 7 configs in parallel.

    Usage: ./.venv/bin/modal run modal_app.py::lora_sweep
    """
    import csv
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent

    # build grid, skip r=16/e=10
    configs = []
    for r in (4, 8, 16, 32):
        for epochs in (5, 10):
            if r == 16 and epochs == 10:
                continue
            configs.append((r, 2 * r, epochs))

    print(f"[sweep] launching {len(configs)} configs in parallel (skip r=16/e=10)")
    for r, alpha, epochs in configs:
        print(f"  r={r} alpha={alpha} epochs={epochs}")

    # starmap: each tuple -> (epochs, batch_size, infer_batch, r, alpha, dropout, limit)
    args = [(epochs, batch_size, 128, r, alpha, 0.05, limit) for r, alpha, epochs in configs]
    results_list = list(train_lora.starmap(args))

    # fold in existing r=16/e=10 result
    existing = root / "results" / "phase6_metrics.json"
    if existing.exists():
        meta = json.loads(existing.read_text())
        results_list.append({
            "results": meta["results"],
            "meta": meta,
            "m1b_logits": {},
        })
        print("[sweep] folded in existing r=16/e=10 from phase6_metrics.json")

    # normalize: extract config + results from each entry
    sweep_results = []
    for i, entry in enumerate(results_list):
        if "meta" in entry and entry["meta"]:
            m = entry["meta"]
            cfg = {"r": m["r"], "alpha": m["alpha"], "epochs": m["epochs"],
                   "dropout": m["dropout"]}
        else:
            r, alpha, epochs = configs[i]
            cfg = {"r": r, "alpha": alpha, "epochs": epochs, "dropout": 0.05}
        sweep_results.append({"config": cfg, "results": entry["results"]})

    # save full JSON
    (root / "results").mkdir(exist_ok=True)
    sweep_path = root / "results" / "lora_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"[sweep] saved {sweep_path}")

    # flatten to CSV
    rows = []
    for entry in sweep_results:
        cfg = entry["config"]
        for key, metrics in entry["results"].items():
            model_name, split_name = key.split("/")
            rows.append({
                "r": cfg["r"], "alpha": cfg["alpha"], "epochs": cfg["epochs"],
                "model": model_name, "split": split_name, **metrics,
            })
    csv_path = root / "results" / "lora_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] saved {csv_path}")

    # summary table
    print(f"\n{'r':>4s} {'alpha':>5s} {'ep':>3s} | "
          f"{'mt_acc':>7s} {'mt_d':>7s} | {'cf_acc':>7s} {'cf_d':>7s} | "
          f"{'ev_acc':>8s} {'ev_d':>7s}")
    print("-" * 75)
    for entry in sorted(sweep_results, key=lambda e: (e["config"]["r"], e["config"]["epochs"])):
        cfg, R = entry["config"], entry["results"]
        mt_a, mt_b = R["M1a/modern_test"]["acc"], R["M1b/modern_test"]["acc"]
        cf_a, cf_b = R["M1a/cf"]["acc"], R["M1b/cf"]["acc"]
        ev_a, ev_b = R["M1a/eval"]["acc"], R["M1b/eval"]["acc"]
        print(f"{cfg['r']:4d} {cfg['alpha']:5d} {cfg['epochs']:3d} | "
              f"{mt_b:7.3f} {mt_b-mt_a:+7.3f} | {cf_b:7.3f} {cf_b-cf_a:+7.3f} | "
              f"{ev_b:8.3f} {ev_b-ev_a:+7.3f}")


# ===========================================================================
# Tier-2 EXACT reproduction — UniversalFakeDetect (Ojha+ CVPR'23) official ckpt
# ===========================================================================
# Runs UnivFD's OWN code + OFFICIAL fc_weights.pth on their OFFICIAL diffusion
# test set (gdown), reproducing their per-generator AP. Our M1 IS this method, so
# matching their published table validates our pipeline against the literature.
univfd_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "ftfy", "regex", "tqdm", "scikit-learn",
                 "numpy", "pillow", "scipy", "gdown", "certifi", "setuptools")
    .add_local_dir("external/UniversalFakeDetect", "/univfd", copy=True)
)

# UnivFD diffusion test set (1k real + 1k fake per domain): Google Drive file id
UNIVFD_DIFFUSION_GDRIVE_ID = "1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t"


# UnivFD diffusion test set domain pairing (from their dataset_paths.py):
# reals live in a shared folder; each fake domain has its own 1_fake.
UNIVFD_PAIRS = [
    ("guided", "imagenet", "guided"),
    ("ldm_200", "laion", "ldm_200"),
    ("ldm_200_cfg", "laion", "ldm_200_cfg"),
    ("ldm_100", "laion", "ldm_100"),
    ("glide_100_27", "laion", "glide_100_27"),
    ("glide_50_27", "laion", "glide_50_27"),
    ("glide_100_10", "laion", "glide_100_10"),
    ("dalle", "laion", "dalle"),
]


@app.function(image=univfd_image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def validate_univfd(gdrive_id: str = UNIVFD_DIFFUSION_GDRIVE_ID) -> dict:
    import os
    import sys
    import zipfile

    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # clip.load urllib download

    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from sklearn.metrics import average_precision_score, accuracy_score

    sys.path.insert(0, "/univfd")
    from models import get_model

    # --- fetch the official test set (cached on the volume across runs) ---
    data_root = "/data/univfd_diffusion"
    if not os.path.isdir(data_root) or not os.listdir(data_root):
        import gdown
        os.makedirs(data_root, exist_ok=True)
        out = "/tmp/diffusion.zip"
        print(f"[univfd] downloading test set (gdrive id {gdrive_id}) ...", flush=True)
        gdown.download(id=gdrive_id, output=out, quiet=False)
        print(f"[univfd] downloaded {os.path.getsize(out)/1e9:.2f} GB; extracting ...", flush=True)
        with zipfile.ZipFile(out) as z:
            z.extractall(data_root)
        vol.commit()
    else:
        print("[univfd] reusing cached test set on volume")

    # locate any folder named <name> that contains 0_real or 1_fake
    found = {}  # folder name -> path
    for dp, dns, _ in os.walk(data_root):
        for d in dns:
            sub = os.path.join(dp, d)
            if os.path.isdir(os.path.join(sub, "0_real")) or os.path.isdir(os.path.join(sub, "1_fake")):
                found[d] = sub
    print(f"[univfd] domain folders found: {sorted(found)}")

    # --- official model + official weights ---
    device = "cuda"
    model = get_model("CLIP:ViT-L/14")
    sd = torch.load("/univfd/pretrained_weights/fc_weights.pth", map_location="cpu")
    model.fc.load_state_dict(sd)
    model = model.to(device).eval()

    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]
    tf = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])  # exact UnivFD transform

    def list_imgs(d):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(exts)] \
            if os.path.isdir(d) else []

    @torch.no_grad()
    def score(paths):
        out = []
        for s in range(0, len(paths), 128):
            xs = []
            for p in paths[s:s + 128]:
                try:
                    xs.append(tf(Image.open(p).convert("RGB")))
                except Exception:
                    pass
            if not xs:
                continue
            out.append(model(torch.stack(xs).to(device)).sigmoid().flatten().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    results = {}
    for key, real_folder, fake_folder in UNIVFD_PAIRS:
        if real_folder not in found or fake_folder not in found:
            print(f"[univfd] skip {key}: missing {real_folder}/{fake_folder}")
            continue
        real = list_imgs(os.path.join(found[real_folder], "0_real"))
        fake = list_imgs(os.path.join(found[fake_folder], "1_fake"))
        if not real or not fake:
            print(f"[univfd] skip {key}: real={len(real)} fake={len(fake)}")
            continue
        ps, pf = score(real), score(fake)
        y = np.r_[np.zeros(len(ps)), np.ones(len(pf))]
        p = np.r_[ps, pf]
        ap = float(average_precision_score(y, p))
        acc = float(accuracy_score(y, (p > 0.5).astype(int)))
        results[key] = {"ap": round(ap, 4), "acc": round(acc, 4),
                        "n_real": len(ps), "n_fake": len(pf)}
        print(f"[univfd] {key:14s} AP={ap:.4f} acc={acc:.4f} "
              f"(real {len(ps)}, fake {len(pf)})", flush=True)

    if results:
        mean_ap = float(np.mean([v["ap"] for v in results.values()]))
        print(f"[univfd] MEAN AP across {len(results)} domains: {mean_ap:.4f}")
        results["_mean_ap"] = round(mean_ap, 4)
    return results


@app.local_entrypoint()
def univfd_main(gdrive_id: str = UNIVFD_DIFFUSION_GDRIVE_ID):
    import json
    from pathlib import Path
    res = validate_univfd.remote(gdrive_id=gdrive_id)
    root = Path(__file__).resolve().parent
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "univfd_reproduction.json").write_text(json.dumps(res, indent=2))
    print(f"[univfd-local] saved -> results/univfd_reproduction.json ({len(res)} entries)")


if __name__ == "__main__":
    print("Run with:  ./.venv/bin/modal run modal_app.py")
