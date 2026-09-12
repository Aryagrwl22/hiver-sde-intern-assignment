"""Golden-set sampling workflow.

MANUAL STEP REQUIRED: humans must fill the `intent` column.
This module never invents evaluation labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hiver_foundation.settings import (
    DATA_GOLDEN_DIR,
    GOLDEN_TEST_N,
    GOLDEN_TRAIN_N,
    GOLDEN_VAL_N,
    RANDOM_SEED,
)
from hiver_foundation.taxonomy import hint_intent


def sample_golden_candidates(
    messages: pd.DataFrame,
    taxonomy: dict,
    seed: int = RANDOM_SEED,
    n_train: int = GOLDEN_TRAIN_N,
    n_val: int = GOLDEN_VAL_N,
    n_test: int = GOLDEN_TEST_N,
) -> pd.DataFrame:
    """Sample inbound messages from each split. Test rows stay in the test split."""
    if "split" not in messages.columns:
        raise ValueError("messages must have a split column from conversation-level splitting.")

    rng = np.random.default_rng(seed)
    parts = []
    for split_name, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        pool = messages[messages["split"] == split_name]
        if pool.empty or n <= 0:
            continue
        # Unique conversations first, then one inbound tweet per conversation
        conv_ids = pool["conversation_id"].dropna().astype(int).unique()
        rng.shuffle(conv_ids)
        chosen = conv_ids[: min(n, len(conv_ids))]
        picked_rows = []
        for conv_id in chosen:
            conv_msgs = pool[pool["conversation_id"] == conv_id].sort_values(
                ["created_at", "tweet_id"]
            )
            picked_rows.append(conv_msgs.iloc[0])
        parts.append(pd.DataFrame(picked_rows))

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["keyword_hint"] = out["customer_text"].map(lambda t: hint_intent(t, taxonomy))
    out["intent"] = ""  # MANUAL STEP REQUIRED
    out["labeler"] = ""
    out["notes"] = ""
    out["is_gold"] = False
    columns = [
        "split",
        "conversation_id",
        "tweet_id",
        "brand",
        "created_at",
        "customer_text",
        "keyword_hint",
        "intent",
        "labeler",
        "notes",
        "is_gold",
    ]
    return out[columns]


def save_golden_template(df: pd.DataFrame, brand: str) -> Path:
    DATA_GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_GOLDEN_DIR / f"golden_candidates_{_safe(brand)}.csv"
    df.to_csv(dest, index=False)
    return dest


def load_labeled_golden(path: str | Path) -> pd.DataFrame:
    """Load a golden CSV. Rows with empty intent are unlabeled and dropped."""
    df = pd.read_csv(path)
    if "intent" not in df.columns or "tweet_id" not in df.columns:
        raise ValueError("Golden CSV must contain tweet_id and intent columns.")
    labeled = df[df["intent"].notna() & (df["intent"].astype(str).str.strip() != "")].copy()
    labeled["intent"] = labeled["intent"].astype(str).str.strip()
    labeled["tweet_id"] = pd.to_numeric(labeled["tweet_id"], errors="coerce")
    labeled = labeled.dropna(subset=["tweet_id"])
    labeled["tweet_id"] = labeled["tweet_id"].astype(int)
    labeled["is_gold"] = True
    return labeled


def _safe(brand: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in brand)
