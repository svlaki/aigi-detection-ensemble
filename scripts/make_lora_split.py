"""Annotate the EXISTING manifest with the training/eval `split` column.

Does NOT touch images or re-download anything — pure in-place annotation.

Roles:
  member_train  : Pool A          (train frozen M1 probe + M2)
  combiner_fit  : Pool B          (fit combiner + calibration)
  eval          : CF + modern_test (held-out evaluation = "Pool C" in figures)
  lora_train    : modern slice partition used ONLY to fine-tune the LoRA

The modern slice (source_dataset == modern_self) is split image-disjoint and
stratified by generator (fakes) / source (reals) into lora_train vs modern_test.
HARD assertion: no lora_train image_id appears in any eval row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


def modern_stratum(image_id: str, label: int, generator: str) -> str:
    """Stratum key for the modern slice: fakes by generator, reals by source."""
    if label == 1:
        return f"fake:{generator}"
    # reals: recover source (ffhq/coco) from the image_id prefix
    if image_id.startswith("modern_real_ffhq"):
        return "real:ffhq"
    if image_id.startswith("modern_real_coco"):
        return "real:coco"
    return "real:other"


def main():
    config.set_seed()
    rng = np.random.default_rng(config.SEED)
    df = pd.read_csv(config.MANIFEST_CSV)
    print(f"[split] loaded manifest: {len(df)} rows")

    df["split"] = ""
    # Pools A/B map straight to roles
    df.loc[df["pool"] == "A", "split"] = config.SPLIT_MEMBER_TRAIN
    df.loc[df["pool"] == "B", "split"] = config.SPLIT_COMBINER_FIT
    # CF (pool C, not modern) is held-out eval
    cf = (df["pool"] == "C") & (df["source_dataset"] == "community_forensics")
    df.loc[cf, "split"] = config.SPLIT_EVAL

    # Modern slice: image-disjoint stratified partition
    modern = df["source_dataset"] == "modern_self"
    mdf = df[modern].copy()
    train_ids: set[str] = set()
    for stratum, grp in mdf.groupby(
        mdf.apply(lambda r: modern_stratum(r.image_id, r.label, r.generator_name), axis=1)
    ):
        ids = grp["image_id"].tolist()
        rng.shuffle(ids)
        n_train = (config.LORA_TRAIN_FAKE_PER_GEN if stratum.startswith("fake:")
                   else config.LORA_TRAIN_REAL_PER_SOURCE)
        n_train = min(n_train, len(ids))
        train_ids.update(ids[:n_train])
        print(f"[split] {stratum:28s} total={len(ids):4d} -> lora_train={n_train} "
              f"modern_test={len(ids)-n_train}")

    df.loc[modern & df["image_id"].isin(train_ids), "split"] = config.SPLIT_LORA_TRAIN
    df.loc[modern & ~df["image_id"].isin(train_ids), "split"] = config.SPLIT_EVAL

    # ---- assertions ----
    assert (df["split"] != "").all(), "some rows have no split assigned"
    eval_ids = set(df[df["split"] == config.SPLIT_EVAL]["image_id"])
    lora_ids = set(df[df["split"] == config.SPLIT_LORA_TRAIN]["image_id"])
    leak = eval_ids & lora_ids
    assert not leak, f"LEAKAGE: {len(leak)} ids in both lora_train and eval"
    # also: no lora_train id anywhere outside lora_train
    assert not (lora_ids & set(df[df["split"] != config.SPLIT_LORA_TRAIN]["image_id"])), \
        "lora_train id appears in a non-lora_train row"

    df.to_csv(config.MANIFEST_CSV, index=False)

    print(f"\n[split] no LoRA-train image appears in any eval row ✓ "
          f"(lora_train={len(lora_ids)}, eval={len(eval_ids)})")
    print("\n[split] role x label counts:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("\n[split] eval-set fake generators (modern_test + CF):")
    ev = df[(df.split == config.SPLIT_EVAL) & (df.label == 1)]
    print(f"  distinct generators: {ev.generator_name.nunique()}")
    print("\n[split] lora_train by stratum:")
    lt = df[df.split == config.SPLIT_LORA_TRAIN]
    print(lt.groupby(["generator_name", "label"]).size())
    print(f"\n[split] manifest written: {config.MANIFEST_CSV}")


if __name__ == "__main__":
    main()
