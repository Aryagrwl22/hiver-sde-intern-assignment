"""Light text cleaning for traditional ML features. Original text is kept separately."""

from __future__ import annotations

import re

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WHITESPACE_RE = re.compile(r"\s+")
MASK_RE = re.compile(r"__(email|phone|url)__", flags=re.IGNORECASE)


def clean_text(text: str) -> str:
    """Lowercase, strip urls/mentions, keep hashtag words, squeeze space."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    out = str(text)
    out = URL_RE.sub(" ", out)
    out = MENTION_RE.sub(" ", out)
    out = HASHTAG_RE.sub(r"\1", out)
    out = MASK_RE.sub(" ", out)
    out = out.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    out = out.lower()
    out = WHITESPACE_RE.sub(" ", out).strip()
    return out


def add_cleaned_text(df: pd.DataFrame, source_col: str, dest_col: str = "text_clean") -> pd.DataFrame:
    out = df.copy()
    out[dest_col] = out[source_col].map(clean_text)
    return out
