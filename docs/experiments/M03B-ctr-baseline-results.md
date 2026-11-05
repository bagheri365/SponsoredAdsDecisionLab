# M03B — CTR baseline results (corrected run)

**Status:** Corrected full-sample training run completed and all 38 tests passed on the user’s Mac (Python 3.10.15). Results documented; serving-time preprocessing and additional slice/calibration evaluations remain open.

## Research question

Do the six behavioral-intent features improve observed-impression click prediction beyond the same static user/ad features with LightGBM?

## Dataset and protocol

- Prepared by M03A from the published TaobaoAd_x1 training split; approximately 876,719 development-training and 219,398 validation impressions.
- Deterministic row-index sampling and validation assignment. **Neither chronological nor user-disjoint.** Source histories are not verified point-in-time safe.
- Published `test.csv` was not used in this experiment.
- M0: hashed-feature SGD logistic regression (static). M1: LightGBM (static). M2: LightGBM (static + six history features).
- Primary metric: binary log loss. Secondary: ROC-AUC, average precision (PR-AUC), Brier score.

## Observed results — after categorical-encoding correction

These values come from the user’s local `python -m sponsored_ads.ctr_train_cli --input-dir artifacts/m03a --output-dir artifacts/m03b` run, followed by reading `artifacts/m03b/metrics.json`. They have **not** been independently rerun in this documentation environment.

| Model | Log loss ↓ | ROC-AUC ↑ | PR-AUC ↑ | Brier ↓ |
|---|---:|---:|---:|---:|
| M0 logistic, static | 0.32812 | 0.53976 | 0.05944 | 0.05160 |
| M1 LightGBM, static | 0.20309 | 0.55386 | 0.06298 | 0.04907 |
| M2 LightGBM, intent | **0.20244** | **0.57489** | **0.06703** | **0.04901** |

**M2 minus M1:** ROC-AUC +0.02103 (+3.80% relative); PR-AUC +0.00405 (+6.43% relative); log loss −0.00065 (0.32% relative reduction); Brier score −0.00006 (0.12% relative reduction). The intent model improves all four aggregate metrics on this validation split, but the absolute discrimination is modest and the probability-quality gains are small. These results do not establish causal advertising lift.

**Correction impact:** Compared with the historical run below, M1 ROC-AUC changed from 0.55278 to 0.55386, and M2 from 0.57482 to 0.57489. M0 remained unchanged. The corrected encoding therefore did not materially change the main qualitative conclusion on this split.

**Run evidence:** `pytest` reported **38 passed in 3.08s**. The full training CLI completed and printed all three models’ metrics without the previously reported LightGBM deprecation or sparse-categorical warnings. The user did not provide a runtime duration or model artifact checksums; those remain unrecorded.

## Observed results — before categorical-encoding correction

Values below are copied from the user's local CLI output, not reproduced in this environment.

| Model | Log loss ↓ | ROC-AUC ↑ | PR-AUC ↑ | Brier ↓ |
|---|---:|---:|---:|---:|
| M0 logistic, static | 0.32812 | 0.53976 | 0.05944 | 0.05160 |
| M1 LightGBM, static | 0.20312 | 0.55278 | 0.06269 | 0.04908 |
| M2 LightGBM, intent | 0.20245 | 0.57482 | 0.06695 | 0.04901 |

Relative to M1, M2 improved ROC-AUC by **0.02204** (about 3.99% relative), average precision by **0.00426** (about 6.80% relative), and log loss by **0.00067** (about 0.33% relative). These are observed associations on this development validation split, not evidence of causal lift.

## Training diagnostics and corrective action

The initial training completed and the user reported **37 passing tests**, with one `LGBMDeprecationWarning` concerning `eval_set`. LightGBM also warned about sparse categorical codes. The original LightGBM pipeline used a stable 20-bit hash of categorical IDs, which produces sparse codes and permits collisions.

The applied follow-up patch replaces the LightGBM categorical hashing with **training-fitted contiguous codes**, and encodes unseen validation categories as `-1`. It also uses `eval_X`/`eval_y` when supported by the installed LightGBM API, with an older-version fallback. **The historical table above belongs to the old representation; the corrected table belongs to the updated representation.**

## Remaining work

1. Capture exact LightGBM/scikit-learn versions, runtime, and model artifact checksums.
2. Add calibration and history-length slice analyses, then document limitations.
3. Persist training-fitted category maps before using LightGBM model files for inference; model text files alone cannot reproduce preprocessing.
4. Evaluate stability with an additional defensible holdout strategy if a timestamped or user-disjoint source becomes available.

## Limitations

- CTR labels are conditioned on historical ad exposure; non-exposed candidates are not negatives.
- User/ad overlap and unverified history timing may inflate offline metrics.
- No temporal holdout, unbiased policy evaluation, advertiser-budget outcomes, or causal lift estimates.
- Validation was used for LightGBM early stopping; repeated experimentation on the same validation split may overfit model selection.
