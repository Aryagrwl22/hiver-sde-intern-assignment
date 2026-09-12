# Golden set

MANUAL STEP REQUIRED

The pipeline writes `golden_candidates_<brand>.csv` here.

1. Read `configs/taxonomy_<brand>.json` and `LABELING.md`.
2. Fill the `intent` column. Leave `keyword_hint` as a hint only — do not copy it blindly.
3. Put your name in `labeler`.
4. Save as `golden_labeled_<brand>.csv`.
5. Rerun:

```
python scripts/run_foundation.py --csv data/raw/twcs.csv --brand YOUR_BRAND --golden data/golden/golden_labeled_YOUR_BRAND.csv
```

Do not invent labels. If you are unsure, use `other` and write a note.
