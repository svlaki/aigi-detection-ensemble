"""One-shot verification of every data source before the overnight download.
Streams 1 example per source, prints schema + splits. No bulk download.
"""
import sys
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

def first_example(name, split, **kw):
    ds = load_dataset(name, split=split, streaming=True, **kw)
    return next(iter(ds))

def describe(ex):
    out = []
    for k, v in ex.items():
        if hasattr(v, "size"):
            out.append(f"{k}:img{tuple(v.size)}/{getattr(v,'mode','?')}")
        else:
            out.append(f"{k}:{type(v).__name__}={str(v)[:40]}")
    return "  ".join(out)

def probe(name, split="train", **kw):
    print(f"\n## {name}  (split={split}) {kw if kw else ''}")
    try:
        print("   splits:", get_dataset_split_names(name, **{k:v for k,v in kw.items() if k=='name'}))
    except Exception as e:
        print("   splits: ERR", repr(e)[:120])
    try:
        print("  ", describe(first_example(name, split, **kw)))
    except Exception as e:
        print("   sample ERR:", repr(e)[:200])

print("===== GenImage fakes (bitmind) =====")
probe("bitmind/GenImage_ADM")

print("\n===== ImageNet reals candidates (need ungated, ~256px) =====")
for cand, sp in [("benjamin-paine/imagenet-1k-256x256", "train"),
                 ("evanarlian/imagenet_1k_resized_256", "train"),
                 ("mrm8488/ImageNet1K-val", "train")]:
    probe(cand, sp)

print("\n===== Community Forensics (Pool C terminal) =====")
probe("OwensLab/CommunityForensics-Eval", "CompEval")

print("\n===== Modern generators (Pool C + LoRA) =====")
probe("LukasT9/Flux-1-Dev-Images-1k", "train")
probe("momodawoud/sd3.5_generated_examples_hard_medium", "train")
probe("davidmunechika/midjourney-images", "train")

print("\n===== Reals for modern slice (COCO / FFHQ, ungated) =====")
probe("rafaelpadilla/coco2017", "val")
probe("merkol/ffhq-256", "train")
print("\nDONE")
