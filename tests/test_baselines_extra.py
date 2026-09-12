import numpy as np
import pandas as pd
from hiver_foundation.baselines import fit_tfidf_logreg

def test_tfidf_logreg_deterministic():
    texts = pd.Series(["hello world", "goodbye world", "hello again", "goodbye again"])
    labels = pd.Series(["hello_intent", "goodbye_intent", "hello_intent", "goodbye_intent"])
    
    model1 = fit_tfidf_logreg(texts, labels)
    model2 = fit_tfidf_logreg(texts, labels)
    
    test_texts = ["hello", "goodbye"]
    preds1 = model1.predict(test_texts)
    preds2 = model2.predict(test_texts)
    
    assert list(preds1) == list(preds2)
    assert isinstance(preds1, np.ndarray)

def test_tfidf_logreg_prediction_format():
    texts = pd.Series(["hello world", "goodbye text"])
    labels = pd.Series(["intent_a", "intent_b"])
    model = fit_tfidf_logreg(texts, labels)
    preds = model.predict(["a c"])
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 1
    assert isinstance(preds[0], str)
