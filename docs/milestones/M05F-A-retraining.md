# M05F-A — Retraining and persisted feature contract

## Purpose

Retrain the intent LightGBM model with the existing M03A training rows and the dedicated M05F-A early-stopping partition. Save a new model and its train-fitted categorical contract. Do not use calibration or final evaluation rows for model fitting or early stopping.

## Run

```bash
python -m sponsored_ads.m05fa_retrain \
  --input-dir artifacts/m03a \
  --split-dir artifacts/m05fa/splits \
  --output-dir artifacts/m05fa/model \
  --seed 2026 --n-estimators 250
```

Outputs: `lightgbm_intent.txt`, `lightgbm_intent.contract.json`, and `retrain_report.json`. The report records input and output SHA-256 hashes, training settings, and early-stopping metrics. The saved bundle is reloaded and checked for feature-count and feature-name consistency.

## Evaluation boundaries

The early-stopping metrics are **not** an unbiased final evaluation. Calibration and final evaluation files are reserved for M05F-B. The original development sampling is row-based, not chronological or user-disjoint; historical feature point-in-time validity is unverified. Published `test.csv` remains unused. No live-auction effectiveness claim is supported.

## Acceptance

Run `pytest -q`; inspect `retrain_report.json`; confirm a contract file is present and that `published_test_used` is false. Do not commit until results are reviewed.
