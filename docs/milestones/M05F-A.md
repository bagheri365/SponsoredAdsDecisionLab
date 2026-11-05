# M05F-A — Independent calibration data roles

**Status:** Split implementation ready; no real run or model retraining claimed.

## Goal

Retain M03A `train_features.csv` for model fitting and split M03A `validation_features.csv` into three mutually exclusive row partitions:

- `early_stopping_features.csv`: model iteration selection only
- `calibration_features.csv`: calibrator fitting only
- `evaluation_features.csv`: final comparison only

Default proportions of the *existing validation file* are 34%, 33%, and 33%, respectively. Assignment uses a deterministic BLAKE2b row-index hash with a separate M05F-A namespace. All three roles are separate by row, **not** by user or time. Existing model artifacts must not be treated as independently evaluated, since their early stopping used the full M03A validation partition.

## Run

```bash
python -m sponsored_ads.m05fa_split \
  --input-dir artifacts/m03a \
  --output-dir artifacts/m05fa/splits \
  --seed 2026
```

The command refuses to overwrite an existing output. The manifest includes source and output SHA-256 hashes, row/click counts, split fractions, and limitations. The original M03A artifacts are not modified. Do not commit the local derived CSVs or model artifacts.

## Acceptance criteria

1. Full project test suite passes.
2. Three disjoint output files exist, with both click classes in each.
3. Total partition counts match the original validation manifest (219,398 rows and 11,373 clicks for the recorded run).
4. Manifest records deterministic provenance and SHA-256 hashes.
5. Published test data remains untouched.

## Next increment

Retrain LightGBM with M03A training data and **only** the new early-stopping partition. Save a new model plus its train-fitted categorical feature contract. Then obtain untouched calibration/evaluation predictions for M05F-B. Do not claim real calibration improvement in this increment.

## Limitations

The source split is deterministic row-based sampling, not chronological or user-disjoint. Source histories have no verified point-in-time semantics; users and ads may appear across partitions. Offline click labels are observed impressions, not counterfactual outcomes or an unbiased auction evaluation.
