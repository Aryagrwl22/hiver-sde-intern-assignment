"""End-to-end foundation pipeline: inspect -> brand -> threads -> split -> baselines."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hiver_foundation.brands import brand_statistics, select_brand
from hiver_foundation.baselines import fit_majority_baseline, fit_tfidf_logreg
from hiver_foundation.clean import add_cleaned_text
from hiver_foundation.conversations import (
    conversation_table,
    conversations_for_brand,
    inbound_message_table,
    reconstruct_conversations,
)
from hiver_foundation.evaluate import compute_metrics, confusion_matrix_df, save_eval_artifacts
from hiver_foundation.golden import load_labeled_golden, sample_golden_candidates, save_golden_template
from hiver_foundation.load import inspect_tweets, load_tweets
from hiver_foundation.settings import DATA_PROCESSED_DIR, PREFERRED_BRAND, RANDOM_SEED, REPORTS_DIR
from hiver_foundation.split import assign_split, split_conversation_ids
from hiver_foundation.taxonomy import save_taxonomy, taxonomy_for_brand


def run_pipeline(
    csv_path: str | Path | None = None,
    preferred_brand: str | None = PREFERRED_BRAND,
    golden_labeled_path: str | Path | None = None,
    reports_dir: Path | None = None,
) -> dict:
    reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tweets = load_tweets(csv_path)
    inspection = inspect_tweets(tweets)

    tweets = reconstruct_conversations(tweets)
    conv_counts = (
        tweets.loc[~tweets["inbound"]]
        .groupby("author_id")["conversation_id"]
        .nunique()
        .rename("n_conversations")
        .reset_index()
        .rename(columns={"author_id": "brand"})
    )
    brand_stats = brand_statistics(tweets)
    brand_stats = brand_stats.merge(conv_counts, on="brand", how="left")
    brand_stats["n_conversations"] = brand_stats["n_conversations"].fillna(0).astype(int)
    try:
        brand_stats.to_csv(reports_dir / "brand_statistics.csv", index=False)
    except PermissionError:
        print("Warning: could not write brand_statistics.csv (Permission denied)")


    brand = select_brand(
        tweets, conversation_counts=conv_counts, preferred=preferred_brand
    )
    brand_tweets = conversations_for_brand(tweets, brand)
    convs = conversation_table(brand_tweets, brand)
    messages = inbound_message_table(brand_tweets, brand)
    messages = add_cleaned_text(messages, "customer_text", "text_clean")
    messages = messages[messages["text_clean"].str.len() > 0].copy()

    splits = split_conversation_ids(messages["conversation_id"].dropna().unique())
    messages = assign_split(messages, splits)
    convs = assign_split(convs, splits) if not convs.empty else convs

    messages.to_csv(DATA_PROCESSED_DIR / "inbound_messages.csv", index=False)
    convs.to_csv(DATA_PROCESSED_DIR / "conversations.csv", index=False)

    taxonomy = taxonomy_for_brand(brand)
    taxonomy_path = save_taxonomy(brand)
    candidates = sample_golden_candidates(messages, taxonomy)
    golden_path = save_golden_template(candidates, brand)

    result = {
        "seed": RANDOM_SEED,
        "source_csv": tweets["source_path"].iloc[0] if len(tweets) else None,
        "inspection": inspection,
        "selected_brand": brand,
        "n_brand_tweets": int(len(brand_tweets)),
        "n_conversations": int(convs.shape[0]),
        "n_inbound_messages": int(len(messages)),
        "split_counts": messages["split"].value_counts(dropna=False).to_dict(),
        "taxonomy_path": str(taxonomy_path),
        "golden_candidates_path": str(golden_path),
        "n_golden_candidates": int(len(candidates)),
        "manual_step": (
            "MANUAL STEP REQUIRED: open the golden CSV, fill the intent column using "
            f"{taxonomy_path}, then rerun with --golden path/to/labeled.csv"
        ),
        "baselines": {},
    }

    labeled = pd.DataFrame()
    if golden_labeled_path:
        gold = load_labeled_golden(golden_labeled_path)
        labeled = messages.merge(
            gold[["tweet_id", "intent"]].drop_duplicates("tweet_id"),
            on="tweet_id",
            how="inner",
        )
    elif "intent" in messages.columns:
        labeled = messages[
            messages["intent"].notna() & (messages["intent"].astype(str).str.strip() != "")
        ].copy()

    result["baselines"] = maybe_run_baselines(labeled, reports_dir)
    (reports_dir / "foundation_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2), encoding="utf-8"
    )
    return result


def maybe_run_baselines(labeled: pd.DataFrame, reports_dir: Path) -> dict:
    if labeled.empty or "intent" not in labeled.columns:
        return {
            "status": "skipped",
            "reason": "MANUAL STEP REQUIRED: no labeled intents yet. Baselines were not trained.",
        }
    if "split" not in labeled.columns:
        return {
            "status": "skipped",
            "reason": "Labeled file needs a split column (train/val/test).",
        }
    if "text_clean" not in labeled.columns:
        from hiver_foundation.clean import add_cleaned_text

        text_col = "customer_text" if "customer_text" in labeled.columns else "text"
        labeled = add_cleaned_text(labeled, text_col, "text_clean")

    train = labeled[labeled["split"] == "train"]
    val = labeled[labeled["split"] == "val"]
    test = labeled[labeled["split"] == "test"]
    if train.empty:
        return {
            "status": "skipped",
            "reason": "MANUAL STEP REQUIRED: label at least some train-split golden rows.",
        }

    # Leakage guard: conversation ids must not cross splits among labeled rows.
    _assert_no_split_leakage(labeled)

    majority = fit_majority_baseline(train["intent"])
    out = {
        "status": "ok",
        "n_labeled_train": int(len(train)),
        "n_labeled_val": int(len(val)),
        "n_labeled_test": int(len(test)),
        "majority_label": majority.label,
        "models": {},
    }

    for split_name, split_df in (("val", val), ("test", test)):
        if split_df.empty:
            continue
        preds = majority.predict(split_df["text_clean"])
        metrics = compute_metrics(split_df["intent"], preds)
        cm = confusion_matrix_df(split_df["intent"], preds)
        paths = save_eval_artifacts(f"majority_{split_name}", metrics, cm, reports_dir)
        out["models"].setdefault("majority", {})[split_name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "n_examples": metrics["n_examples"],
            "artifacts": {k: str(v) for k, v in paths.items()},
        }

    try:
        model = fit_tfidf_logreg(train["text_clean"], train["intent"])
    except ValueError as exc:
        out["tfidf_logreg_error"] = str(exc)
        return out

    for split_name, split_df in (("val", val), ("test", test)):
        if split_df.empty:
            continue
        preds = model.predict(split_df["text_clean"].tolist())
        metrics = compute_metrics(split_df["intent"], preds)
        cm = confusion_matrix_df(split_df["intent"], preds)
        paths = save_eval_artifacts(f"tfidf_logreg_{split_name}", metrics, cm, reports_dir)
        out["models"].setdefault("tfidf_logreg", {})[split_name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "n_examples": metrics["n_examples"],
            "artifacts": {k: str(v) for k, v in paths.items()},
        }
    return out


def _assert_no_split_leakage(df: pd.DataFrame) -> None:
    grouped = df.dropna(subset=["conversation_id", "split"]).groupby("conversation_id")["split"]
    mixed = grouped.nunique()
    leaked = mixed[mixed > 1]
    if len(leaked):
        raise ValueError(
            f"Conversation-level leakage detected for ids: {list(leaked.index[:10])}"
        )


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)
