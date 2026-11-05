# M05F-B — Real CTR calibration experiment

**Scope:** Score the M05F-A calibration and evaluation partitions using the M05F-A LightGBM booster and exact feature contract. Fit sigmoid (Platt on clipped logit) and isotonic regression **only** on calibration labels/scores. Compare raw, sigmoid, and isotonic probabilities **only** on the evaluation partition.

## Reproduction

Prerequisites: `artifacts/m05fa/splits/` and `artifacts/m05fa/model/` from the verified M05F-A commands. The runner verifies split, model, contract, and input file hashes against the M05F-A provenance reports and checks label counts. No published test data is opened.

```bash
pytest -q
python -m sponsored_ads.m05fb_experiment \
  --split-dir artifacts/m05fa/splits \
  --model-dir artifacts/m05fa/model \
  --output artifacts/m05fb/calibration_report.json \
  --bins 10 --batch-size 8192
```

The report contains Brier score, log loss, ROC-AUC, ECE, and equal-width reliability bins for all three prediction streams, with observed-impression counts and SHA-256 provenance. The runner refuses to overwrite an existing report.

## Interpretation

These are **logged-impression** diagnostics, not policy value or live-auction measurements. M03A and M05F-A partitions are row-hash based, not temporal or user-disjoint. Historical features are not verified point-in-time. Method comparison on the evaluation set is exploratory; selecting the best method based on these metrics and then calling the same set a final unbiased test would introduce selection bias. No calibrator is promoted or persisted for serving in this milestone.
