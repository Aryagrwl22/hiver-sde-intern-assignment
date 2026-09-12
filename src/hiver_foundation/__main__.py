"""CLI: python -m hiver_foundation --demo   or   python -m hiver_foundation --csv data/raw/twcs.csv"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hiver_foundation.pipeline import run_pipeline
from hiver_foundation.settings import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hiver foundation pipeline (data + traditional ML only)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to twcs.csv or a subsample. Default: data/raw/twcs.csv then sample.csv",
    )
    parser.add_argument(
        "--brand",
        type=str,
        default=None,
        help="Force a brand author_id (e.g. AppleSupport). Must exist in the CSV.",
    )
    parser.add_argument(
        "--golden",
        type=str,
        default=None,
        help="Path to a HAND-labeled golden CSV (intent column filled in).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on tests/fixtures/sample_tweets.csv (tiny synthetic file, not Kaggle).",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if args.demo:
        csv_path = ROOT / "tests" / "fixtures" / "sample_tweets.csv"
        golden = ROOT / "tests" / "fixtures" / "golden_labeled.csv"
        result = run_pipeline(csv_path=csv_path, preferred_brand=args.brand, golden_labeled_path=golden)
    else:
        result = run_pipeline(
            csv_path=csv_path,
            preferred_brand=args.brand,
            golden_labeled_path=args.golden,
        )

    print(json.dumps(_preview(result), indent=2))


def _preview(result: dict) -> dict:
    keep = dict(result)
    return keep


if __name__ == "__main__":
    main()
