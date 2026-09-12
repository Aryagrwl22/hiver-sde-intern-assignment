from pathlib import Path

from hiver_foundation.clean import clean_text
from hiver_foundation.conversations import reconstruct_conversations
from hiver_foundation.load import load_tweets
from hiver_foundation.split import assign_split, split_conversation_ids

FIXTURE = Path(__file__).parent / "fixtures" / "sample_tweets.csv"


def test_clean_text_strips_urls_and_mentions():
    raw = "Hello @AcmeSupport see https://example.com/a #Help &amp; thanks"
    assert clean_text(raw) == "hello see help & thanks"


def test_reconstruct_groups_reply_chain():
    df = load_tweets(FIXTURE)
    out = reconstruct_conversations(df)
    chain = out[out["tweet_id"].isin([1000, 1001])]
    assert set(chain["conversation_id"].unique()) == {1000}


def test_split_has_no_conversation_leakage():
    df = load_tweets(FIXTURE)
    df = reconstruct_conversations(df)
    inbound = df[df["inbound"]]
    splits = split_conversation_ids(inbound["conversation_id"], seed=42)
    labeled = assign_split(inbound, splits)
    overlap_tv = set(splits["train"]) & set(splits["val"])
    overlap_tt = set(splits["train"]) & set(splits["test"])
    overlap_vt = set(splits["val"]) & set(splits["test"])
    assert not overlap_tv
    assert not overlap_tt
    assert not overlap_vt
    mixed = labeled.groupby("conversation_id")["split"].nunique()
    assert (mixed == 1).all()
    assert set(labeled["split"].dropna()) <= {"train", "val", "test"}
