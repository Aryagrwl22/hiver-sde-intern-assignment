from pathlib import Path

import pandas as pd

from hiver_foundation.baselines import fit_majority_baseline, fit_tfidf_logreg
from hiver_foundation.brands import select_brand
from hiver_foundation.evaluate import compute_metrics
from hiver_foundation.load import load_tweets
from hiver_foundation.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_tweets.csv"
GOLDEN = ROOT / "tests" / "fixtures" / "golden_labeled.csv"


def test_majority_baseline_picks_mode():
    labels = pd.Series(["a", "a", "b"])
    model = fit_majority_baseline(labels)
    preds = model.predict(["x", "y"])
    assert list(preds) == ["a", "a"]


def test_tfidf_logreg_fits_and_predicts():
    texts = pd.Series(
        [
            "cannot login password",
            "account locked username",
            "charged twice card invoice",
            "payment failing bill",
        ]
    )
    labels = pd.Series(["account_access", "account_access", "billing_payment", "billing_payment"])
    model = fit_tfidf_logreg(texts, labels)
    pred = model.predict(["my password login is broken"])[0]
    assert pred in {"account_access", "billing_payment"}


def test_metrics_perfect_when_identical():
    m = compute_metrics(["a", "b"], ["a", "b"])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_select_brand_prefers_configured_name():
    df = load_tweets(FIXTURE)
    brand = select_brand(df, preferred="AcmeSupport")
    assert brand == "AcmeSupport"


def test_pipeline_demo_runs_baselines(tmp_path):
    result = run_pipeline(
        csv_path=FIXTURE,
        preferred_brand="AcmeSupport",
        golden_labeled_path=GOLDEN,
        reports_dir=tmp_path,
    )
    assert result["selected_brand"] == "AcmeSupport"
    assert result["n_conversations"] >= 18
    assert result["baselines"]["status"] == "ok"
    assert "majority" in result["baselines"]["models"]
    assert "tfidf_logreg" in result["baselines"]["models"]
    # Test split must not be empty in this fixture after 70/15/15
    split_counts = result["split_counts"]
    assert split_counts.get("train", 0) > 0
    assert split_counts.get("test", 0) > 0
