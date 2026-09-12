"""Accuracy, macro-F1, per-class scores, and a confusion-matrix CSV + PNG."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from hiver_foundation.settings import REPORTS_DIR


def compute_metrics(y_true, y_pred, labels=None) -> dict:
    labels = labels or sorted(set(list(y_true) + list(y_pred)))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = []
    for i, label in enumerate(labels):
        per_class.append(
            {
                "intent": label,
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "n_examples": int(len(list(y_true))),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    }


def confusion_matrix_df(y_true, y_pred, labels=None) -> pd.DataFrame:
    labels = labels or sorted(set(list(y_true) + list(y_pred)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def save_eval_artifacts(
    name: str,
    metrics: dict,
    cm: pd.DataFrame,
    reports_dir: Path | None = None,
) -> dict[str, Path]:
    reports_dir = reports_dir or REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / f"{name}_metrics.json"
    cm_csv = reports_dir / f"{name}_confusion_matrix.csv"
    cm_png = reports_dir / f"{name}_confusion_matrix.png"

    payload = {k: v for k, v in metrics.items() if k != "classification_report"}
    payload["classification_report"] = metrics.get("classification_report", "")
    import json

    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cm.to_csv(cm_csv)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks(range(len(cm.columns)))
        ax.set_yticks(range(len(cm.index)))
        ax.set_xticklabels(cm.columns, rotation=45, ha="right")
        ax.set_yticklabels(cm.index)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(cm_png, dpi=120)
        plt.close(fig)
    except Exception:
        cm_png = Path()

    return {"metrics": metrics_path, "confusion_csv": cm_csv, "confusion_png": cm_png}
