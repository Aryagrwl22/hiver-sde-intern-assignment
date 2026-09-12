import json
from pathlib import Path

import pandas as pd
import pytest

from hiver_foundation.taxonomy import load_taxonomy, hint_intent
from hiver_foundation.golden import load_labeled_golden

def test_taxonomy_loading(tmp_path):
    tax_data = {
        "brand": "AppleSupport",
        "intents": [
            {"intent": "software_issue", "keyword_hints": ["bug", "glitch"]},
            {"intent": "other"}
        ]
    }
    p = tmp_path / "tax.json"
    p.write_text(json.dumps(tax_data))
    loaded = load_taxonomy(p)
    assert loaded["brand"] == "AppleSupport"
    assert len(loaded["intents"]) == 2

def test_hint_intent():
    tax_data = {
        "brand": "AppleSupport",
        "intents": [
            {"intent": "software_issue", "keyword_hints": ["bug", "glitch"]},
            {"intent": "hardware_issue", "keyword_hints": ["screen", "button"]},
            {"intent": "other"}
        ]
    }
    assert hint_intent("my screen is broken", tax_data) == "hardware_issue"
    assert hint_intent("i found a bug", tax_data) == "software_issue"
    assert hint_intent("hello", tax_data) == "other"

def test_golden_label_validation(tmp_path):
    # Mock a golden labeled file
    p = tmp_path / "golden.csv"
    pd.DataFrame({
        "tweet_id": [1, 2, 3],
        "split": ["train", "val", "test"],
        "conversation_id": [10, 11, 12],
        "intent": ["software_issue", "hardware_issue", ""]
    }).to_csv(p, index=False)
    
    labeled = load_labeled_golden(p)
    # The empty intent should be dropped
    assert len(labeled) == 2
    assert set(labeled["intent"]) == {"software_issue", "hardware_issue"}
    assert labeled["is_gold"].all()

def test_actual_golden_set_leakage():
    # If the actual golden candidates file exists, check for leakage
    golden_path = Path("c:/Hiver1/data/golden/golden_candidates_AppleSupport.csv")
    if not golden_path.exists():
        pytest.skip("Golden candidates file not generated yet")
    
    df = pd.read_csv(golden_path)
    
    # Check that a single conversation ID doesn't appear in multiple splits
    split_counts = df.groupby("conversation_id")["split"].nunique()
    assert (split_counts == 1).all(), "Leakage detected: conversation spans multiple splits!"

def test_actual_golden_labels_in_taxonomy():
    golden_path = Path("c:/Hiver1/data/golden/golden_candidates_AppleSupport.csv")
    tax_path = Path("c:/Hiver1/configs/taxonomy_AppleSupport.json")
    
    if not golden_path.exists() or not tax_path.exists():
        pytest.skip("Golden or Taxonomy files not generated yet")
        
    df = pd.read_csv(golden_path)
    tax = load_taxonomy(tax_path)
    valid_intents = {item["intent"] for item in tax.get("intents", [])}
    
    labeled = df[df["intent"].notna() & (df["intent"].astype(str).str.strip() != "")]
    if len(labeled) > 0:
        invalid_labels = set(labeled["intent"]) - valid_intents
        assert not invalid_labels, f"Found invalid intents in golden set: {invalid_labels}"
