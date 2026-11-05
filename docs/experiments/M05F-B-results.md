# M05F-B — Real CTR calibration experiment log

**Status:** Executed locally on 2026-11-05; observed-impression calibration results recorded. **Not committed at the time of this run.** M05F-A baseline commit: `cd4f0b5` (`main`, synchronized with `origin/main`). Record the eventual M05F-B commit after committing.

## Reproduction

The following command was executed from the repository root with the project virtual environment active:

```bash
python -m sponsored_ads.m05fb_experiment \
  --split-dir artifacts/m05fa/splits \
  --model-dir artifacts/m05fa/model \
  --output artifacts/m05fb/calibration_report.json \
  --bins 10 \
  --batch-size 8192
```

- Report: `artifacts/m05fb/calibration_report.json` (local experiment artifact).
- Platform: macOS (MacBook Pro); exact macOS version not recorded.
- Python, NumPy, scikit-learn, and LightGBM versions for this run: **not captured in the shared experiment output**. Record these from the execution environment for full reproducibility.
- Reliability bins: 10 equal-width probability bins; full per-bin values are in the local JSON report and were not included in the shared output. In particular, inspect counts in high-probability bins before drawing conclusions about local calibration.

## Data and evaluation protocol

| Partition | Impressions | Role |
|---|---:|---|
| Calibration | 72,227 | Fit sigmoid and isotonic calibration mappings |
| Evaluation | 72,505 | Compare raw and calibrated probabilities |

The M05F-A model was trained on a separate training partition and used a separate early-stopping partition. The calibration-fit and evaluation partitions were not used for LightGBM fitting or early stopping. The published TaobaoAd test set was not used. Calibration and evaluation click counts were **3,727** and **3,699**, respectively, as recorded in the M05F-A split manifest.

**Important:** Partitions are deterministic row-hash splits, not chronological or user-disjoint. Historical features have not been verified as point-in-time safe. The evaluation represents observed logged impressions, not counterfactual candidate-ranking or auction performance.

## Measured results on 72,505 held-out impressions

| Metric | Raw LightGBM | Sigmoid | Isotonic | Direction |
|---|---:|---:|---:|---|
| ROC-AUC | **0.580896927** | **0.580896927** | 0.579238677 | Higher |
| Log loss | 0.199793148 | 0.199816185 | **0.199580650** | Lower |
| Brier score | 0.048253815 | 0.048249526 | **0.048229739** | Lower |
| ECE (10 equal-width bins) | 0.001174877 | **0.000472873** | 0.000486723 | Lower |

Compared with raw LightGBM, isotonic calibration reduced log loss by **0.000212497** and Brier score by **0.000024076**, while decreasing ROC-AUC by **0.001658251**. Sigmoid calibration improved ECE and slightly improved Brier score but increased log loss by approximately **0.000023037**.

These are small differences. No confidence intervals or repeated-split uncertainty estimates were reported. ECE is sensitive to the binning scheme, and sparse high-probability bins can make bin-level conclusions unreliable. This experiment is **exploratory**: it does not establish a statistically significant improvement or justify selecting a production calibrator from these same evaluation results.

## Artifact provenance (SHA-256)

| Input artifact | SHA-256 |
|---|---|
| Calibration feature CSV | `affc73e87c6b6d7f0097b2e7ef77dcf13ae2b537101caff7dc18027d4bb7b0fa` |
| Evaluation feature CSV | `31827ab978ebca614eaea7ee63754cacbf9cbe8644f846c163f8fdfdcee6b7c7` |
| LightGBM model | `47d9611078d1f79fd4a25b1a00c45ba7cb920762d22e56e08182112ca2aee02c` |
| Feature contract | `89497d6396079b9cf1d6d8353180a6757857e6df3be11235e7f45f3886c89daa` |
| M05F-A retraining report | `785904991dccf181c857e2d669afb0d9ab7de7e93fa825935713ab5de6f5284d` |
| M05F-A split manifest | `d8e917ee3f0e98e450f4c5af2c14e783199531b602be0f5fbf0e6bf700564b7e` |

## Validation

The complete test suite passed after applying M05F-B: **181 passed, 1 warning in 3.46 seconds**. The warning was the existing scikit-learn/LightGBM feature-name warning in `tests/test_ctr_baselines.py::test_lightgbm_smoke`. `git diff --check` reported no whitespace errors. Four expected new M05F-B files were untracked prior to staging.

## Decision and limitations

**No production calibrator is selected or deployed by M05F-B.** Choosing isotonic because it has the lowest evaluation log loss and then treating that same evaluation set as an untouched final test would introduce method-selection bias. M05F-C must establish an explicit selection policy or independent confirmation strategy before promoting a calibrated serving bundle.

The results describe probability quality on logged impressions only. They do not establish temporal generalization, user-disjoint generalization, unbiased relevance, causal lift, auction revenue, or online effectiveness. The local JSON report should be retained for complete reliability-bin details.
