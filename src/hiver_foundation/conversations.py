"""Rebuild conversation threads from tweet reply pointers."""

from __future__ import annotations

import pandas as pd


def reconstruct_conversations(tweets: pd.DataFrame) -> pd.DataFrame:
    """Add conversation_id = root tweet of the reply chain.

    Walks `in_response_to_tweet_id` until it hits a tweet with no parent
    (or a missing parent id). Cycles are broken by visited-set.
    """
    if tweets.empty:
        out = tweets.copy()
        out["conversation_id"] = pd.Series(dtype="float64")
        return out

    id_to_parent = {}
    for tweet_id, parent in zip(
        tweets["tweet_id"], tweets["in_response_to_tweet_id"], strict=False
    ):
        if pd.isna(tweet_id):
            continue
        tid = int(tweet_id)
        if pd.isna(parent):
            id_to_parent[tid] = None
        else:
            id_to_parent[tid] = int(parent)

    root_cache: dict[int, int] = {}

    def root_of(tweet_id: int) -> int:
        if tweet_id in root_cache:
            return root_cache[tweet_id]
        seen: set[int] = set()
        current = tweet_id
        while True:
            if current in seen:
                break
            seen.add(current)
            if current not in id_to_parent:
                break
            parent = id_to_parent[current]
            if parent is None:
                break
            current = parent
        for node in seen:
            root_cache[node] = current
        return current

    roots = []
    for tweet_id in tweets["tweet_id"]:
        if pd.isna(tweet_id):
            roots.append(pd.NA)
        else:
            roots.append(root_of(int(tweet_id)))

    out = tweets.copy()
    out["conversation_id"] = pd.array(roots, dtype="Int64")
    return out


def conversations_for_brand(tweets_with_conv: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Keep threads that contain at least one outbound tweet from `brand`."""
    brand_convs = set(
        tweets_with_conv.loc[
            (~tweets_with_conv["inbound"]) & (tweets_with_conv["author_id"] == brand),
            "conversation_id",
        ].dropna()
    )
    return tweets_with_conv[
        tweets_with_conv["conversation_id"].isin(brand_convs)
    ].copy()


def conversation_table(brand_tweets: pd.DataFrame, brand: str) -> pd.DataFrame:
    """One row per conversation with first customer text and a brand reply flag."""
    if brand_tweets.empty:
        return pd.DataFrame()

    rows = []
    grouped = brand_tweets.sort_values(["conversation_id", "created_at", "tweet_id"])
    for conv_id, group in grouped.groupby("conversation_id", dropna=True):
        inbound = group[group["inbound"]]
        outbound_brand = group[(~group["inbound"]) & (group["author_id"] == brand)]
        first_inbound = inbound.iloc[0] if not inbound.empty else None
        first_brand = outbound_brand.iloc[0] if not outbound_brand.empty else None
        rows.append(
            {
                "conversation_id": int(conv_id),
                "brand": brand,
                "n_tweets": int(len(group)),
                "n_customer_tweets": int(len(inbound)),
                "n_brand_tweets": int(len(outbound_brand)),
                "started_at": group["created_at"].min(),
                "ended_at": group["created_at"].max(),
                "first_customer_tweet_id": (
                    int(first_inbound["tweet_id"]) if first_inbound is not None else pd.NA
                ),
                "first_customer_text": (
                    str(first_inbound["text"]) if first_inbound is not None else ""
                ),
                "first_brand_text": (
                    str(first_brand["text"]) if first_brand is not None else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def inbound_message_table(brand_tweets: pd.DataFrame, brand: str) -> pd.DataFrame:
    """One row per customer (inbound) tweet — the unit we classify later."""
    inbound = brand_tweets[brand_tweets["inbound"]].copy()
    inbound["brand"] = brand
    inbound["customer_text"] = inbound["text"]
    return inbound[
        [
            "conversation_id",
            "tweet_id",
            "brand",
            "created_at",
            "author_id",
            "customer_text",
        ]
    ].reset_index(drop=True)
