"""Traditional ML baselines. Fit only on labeled TRAIN rows. Never on test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hiver_foundation.settings import (
    LOGREG_MAX_ITER,
    RANDOM_SEED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)


@dataclass
class MajorityBaseline:
    label: str

    def predict(self, texts) -> np.ndarray:
        n = len(list(texts))
        return np.array([self.label] * n)


def fit_majority_baseline(train_labels: pd.Series) -> MajorityBaseline:
    if train_labels.empty:
        raise ValueError("No labeled train rows. MANUAL STEP REQUIRED: label the golden train set.")
    mode = train_labels.mode(dropna=True)
    if mode.empty:
        raise ValueError("Train labels are empty.")
    return MajorityBaseline(label=str(mode.iloc[0]))


def fit_tfidf_logreg(train_texts: pd.Series, train_labels: pd.Series) -> Pipeline:
    if train_texts.empty or train_labels.empty:
        raise ValueError("No labeled train rows. MANUAL STEP REQUIRED: label the golden train set.")
    n_classes = train_labels.nunique()
    if n_classes < 2:
        raise ValueError(
            "Logistic regression needs at least 2 intent classes in the labeled train set."
        )
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    min_df=1,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=LOGREG_MAX_ITER,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipe.fit(train_texts.tolist(), train_labels.tolist())
    return pipe
