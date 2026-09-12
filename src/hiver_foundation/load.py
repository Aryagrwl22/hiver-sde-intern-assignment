"""Load the Kaggle Customer Support on Twitter CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hiver_foundation.settings import DEFAULT_RAW_CSV, SAMPLE_RAW_CSV

EXPECTED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def resolve_raw_path(path: str | Path | None = None) -> Path:
    """Pick twcs.csv, then sample.csv, unless a path is passed in."""
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"CSV not found: {resolved}")
        return resolved
    if DEFAULT_RAW_CSV.exists():
        return DEFAULT_RAW_CSV
    if SAMPLE_RAW_CSV.exists():
        return SAMPLE_RAW_CSV
    raise FileNotFoundError(
        "No dataset CSV found. MANUAL STEP REQUIRED:\n"
        "1. Download https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter\n"
        "2. Place twcs.csv (or a subsample named sample.csv) in data/raw/\n"
        "A subsample is expected and encouraged by the assignment."
    )


def load_tweets(path: str | Path | None = None) -> pd.DataFrame:
    """Read tweets and normalize types. Does not drop rows except fully empty text."""
    csv_path = resolve_raw_path(path)
    df = pd.read_csv(csv_path, dtype={"author_id": str, "text": str}, low_memory=False)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns {missing}. "
            "Expected the Kaggle twcs.csv schema: " + ", ".join(EXPECTED_COLUMNS)
        )

    df = df.copy()
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    )
    df["inbound"] = df["inbound"].map(_as_bool)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["text"] = df["text"].fillna("").astype(str)
    df["author_id"] = df["author_id"].fillna("").astype(str)
    df["source_path"] = str(csv_path)
    return df


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "t"}


def inspect_tweets(df: pd.DataFrame) -> dict:
    """Small summary dict for printing and JSON reports."""
    return {
        "n_rows": int(len(df)),
        "n_unique_tweets": int(df["tweet_id"].nunique(dropna=True)),
        "n_authors": int(df["author_id"].nunique()),
        "n_inbound": int(df["inbound"].sum()),
        "n_outbound": int((~df["inbound"]).sum()),
        "date_min": str(df["created_at"].min()),
        "date_max": str(df["created_at"].max()),
        "n_missing_text": int((df["text"].str.strip() == "").sum()),
        "n_missing_tweet_id": int(df["tweet_id"].isna().sum()),
    }
