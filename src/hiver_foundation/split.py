"""Conversation-level train / validation / test split. No tweet leaks across splits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hiver_foundation.settings import RANDOM_SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO


def split_conversation_ids(
    conversation_ids,
    seed: int = RANDOM_SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> dict[str, np.ndarray]:
    ids = np.array(sorted(set(int(x) for x in conversation_ids if pd.notna(x))), dtype=np.int64)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def assign_split(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> pd.DataFrame:
    mapping = {}
    for name, ids in splits.items():
        for conv_id in ids:
            mapping[int(conv_id)] = name
    out = df.copy()
    out["split"] = out["conversation_id"].map(
        lambda x: mapping.get(int(x), None) if pd.notna(x) else None
    )
    return out
