"""Brand statistics and a deterministic brand choice.

Brands are outbound authors (inbound == False) in this dataset.
"""

from __future__ import annotations

import pandas as pd

from hiver_foundation.settings import MIN_CONVERSATIONS_FOR_BRAND, PREFERRED_BRAND


def brand_statistics(tweets: pd.DataFrame) -> pd.DataFrame:
    """One row per brand: outbound tweets and unique customer authors they replied to."""
    outbound = tweets.loc[~tweets["inbound"]].copy()
    if outbound.empty:
        return pd.DataFrame(
            columns=["brand", "n_outbound_tweets", "n_customer_authors"]
        )

    stats = (
        outbound.groupby("author_id", dropna=False)
        .agg(
            n_outbound_tweets=("tweet_id", "count"),
            n_customer_authors=("in_response_to_tweet_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"author_id": "brand"})
        .sort_values(
            ["n_outbound_tweets", "brand"], ascending=[False, True], kind="mergesort"
        )
        .reset_index(drop=True)
    )
    return stats


def select_brand(
    tweets: pd.DataFrame,
    conversation_counts: pd.DataFrame | None = None,
    preferred: str | None = PREFERRED_BRAND,
    min_conversations: int = MIN_CONVERSATIONS_FOR_BRAND,
) -> str:
    """Pick one brand in a fully reproducible way.

    Order of rules:
    1. If `preferred` is set and present in the data, use it.
    2. Else keep brands with at least `min_conversations` (if any qualify).
    3. Sort remaining brands by conversation count ascending, then name.
    4. Pick the middle brand so we avoid the huge head of the distribution
       when the full Kaggle file is used. A subsample is still fine.

    This is not "the best brand". It is a documented, repeatable choice.
    Override it by setting PREFERRED_BRAND in settings.py.
    """
    outbound_brands = tweets.loc[~tweets["inbound"], "author_id"].dropna().unique()
    if len(outbound_brands) == 0:
        raise ValueError("No brand tweets found (inbound==False). Check the CSV.")

    if preferred and preferred in set(outbound_brands):
        return str(preferred)

    if conversation_counts is not None and not conversation_counts.empty:
        counts = conversation_counts.copy()
    else:
        counts = (
            tweets.loc[~tweets["inbound"]]
            .groupby("author_id")["tweet_id"]
            .count()
            .rename("n_conversations")
            .reset_index()
            .rename(columns={"author_id": "brand"})
        )

    eligible = counts[counts["n_conversations"] >= min_conversations]
    if eligible.empty:
        eligible = counts

    eligible = eligible.sort_values(
        ["n_conversations", "brand"], ascending=[True, True], kind="mergesort"
    ).reset_index(drop=True)
    mid = len(eligible) // 2
    return str(eligible.loc[mid, "brand"])
