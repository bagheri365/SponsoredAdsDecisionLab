# M03B — CTR baseline results (initial run)

**Status:** Initial full-sample run recorded; categorical-encoding correction requires rerun before finalizing this milestone.

## Research question

Do the six behavioral-intent features improve observed-impression click prediction beyond the same static user/ad features with LightGBM?

## Dataset and protocol

- Prepared by M03A from the published TaobaoAd_x1 training split; approximately 876,719 development-training and 219,398 validation impressions.
- Deterministic row-index sampling and validation assignment. **Neither chronological nor user-disjoint.** Source histories are not verified point-in-time safe.
- Published `test.csv` was not used in this experiment.
- M0: hashed-feature SGD logistic regression (static). M1: LightGBM (static). M2: LightGBM (static + six history features).
- Primary metric: binary log loss. Secondary: ROC-AUC, average precision (PR-AUC), Brier score.

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

The follow-up patch replaces the LightGBM categorical hashing with **training-fitted contiguous codes**, and encodes unseen validation categories as `-1`. It also uses `eval_X`/`eval_y` when supported by the installed LightGBM API, with an older-version fallback. **This changes the feature representation; the results above must not be attributed to the corrected implementation.**

## Remaining work

1. Rerun `pytest` and the full `ctr_train_cli` experiment with the corrected code.
2. Record the corrected metrics and compare against the historical figures above.
3. Capture Python/LightGBM/scikit-learn versions, runtime, and model artifact checksums.
4. Add calibration and history-length slice analyses, then document limitations.
5. Persist training-fitted category maps before using LightGBM model files for inference; model text files alone cannot reproduce preprocessing.

## Limitations

- CTR labels are conditioned on historical ad exposure; non-exposed candidates are not negatives.
- User/ad overlap and unverified history timing may inflate offline metrics.
- No temporal holdout, unbiased policy evaluation, advertiser-budget outcomes, or causal lift estimates.
- Validation was used for LightGBM early stopping; repeated experimentation on the same validation split may overfit model selection.
