"""Manifest helpers (Phase 1.6).

One CSV: image_id, path, label, source_dataset, generator_name, pool.
Pools assigned from config generator/source lists.
"""
from __future__ import annotations

import pandas as pd

from src import config


def pool_for(source_dataset: str, generator_name: str) -> str | None:
    """Assign pool from config lists. Returns 'A'/'B'/'C' or None if unassigned."""
    if source_dataset in config.POOL_C_SOURCES:
        return "C"
    g = generator_name.lower()
    if g in config.POOL_A_GENERATORS:
        return "A"
    if g in config.POOL_B_GENERATORS:
        return "B"
    if g in config.POOL_C_GENERATORS:
        return "C"
    # reals: pool follows whatever set they were drawn for (caller sets explicitly)
    return None


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of gate violations (empty == Phase 1.6 ✓ gate passes)."""
    problems = []
    missing_cols = [c for c in config.MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        problems.append(f"missing columns: {missing_cols}")
        return problems
    if df["image_id"].duplicated().any():
        problems.append(f"{df['image_id'].duplicated().sum()} duplicate image_ids")
    fakes = df[df["label"] == 1]
    miss_gen = fakes[fakes["generator_name"].isin(["", "real", None]) |
                     fakes["generator_name"].isna()]
    if len(miss_gen):
        problems.append(f"{len(miss_gen)} fakes with missing/invalid generator_name")
    if df["pool"].isna().any() or (df["pool"] == "").any():
        problems.append(f"{(df['pool'].isna() | (df['pool']=='')).sum()} rows with no pool")
    # generator-disjointness A vs B (Phase 2 gate, checked early)
    a = set(df[df["pool"] == "A"]["generator_name"]) - {"real"}
    b = set(df[df["pool"] == "B"]["generator_name"]) - {"real"}
    if a & b:
        problems.append(f"Pool A/B generator leakage: {a & b}")
    # LoRA-train must never appear in the eval set (image-level)
    if "split" in df.columns:
        lora_ids = set(df[df["split"] == config.SPLIT_LORA_TRAIN]["image_id"])
        eval_ids = set(df[df["split"] == config.SPLIT_EVAL]["image_id"])
        if lora_ids & eval_ids:
            problems.append(f"lora_train/eval leakage: {len(lora_ids & eval_ids)} ids")
    return problems


def summary(df: pd.DataFrame) -> str:
    lines = [f"total rows: {len(df)}"]
    g = df.groupby(["pool", "label"]).size().unstack(fill_value=0)
    lines.append("per-pool real(0)/fake(1) counts:\n" + g.to_string())
    lines.append("per (pool, generator):\n" +
                 df.groupby(["pool", "generator_name"]).size().to_string())
    return "\n".join(lines)
