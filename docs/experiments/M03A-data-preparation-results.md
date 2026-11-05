# M03A — TaobaoAd_x1 development data preparation: run report

**Status:** Data preparation completed; model training not started.
**Run date:** 2026-11-05 (reported artifact date).
**Repository:** `SponsoredAdsDecisionLab`
**Source:** `reczoo/TaobaoAd_x1`, published `train.csv`
**Evidence:** User-provided `m03a.zip` (`sample_manifest.json`, `train_features.csv`, `validation_features.csv`) and the previously supplied full-dataset audit manifest. Values below are reported by these manifests; no independent full-file recount was performed for this report.

## 1. Objective and outcome

Create a manageable, deterministic development dataset for CTR prediction, without loading the ~15 GiB source training CSV into memory and without accessing the published `test.csv`. The provided manifest reports a completed full scan of **21,929,911** training impressions, yielding **1,096,117** sampled impressions. The development split contains **876,719** training rows and **219,398** validation rows. No CTR model metrics exist yet.

## 2. Source and provenance

- Dataset: processed Taobao advertising CTR benchmark, `reczoo/TaobaoAd_x1`.
- Input: `data/TaobaoAd_x1/train.csv`; published `test.csv` is reserved for later final evaluation.
- Label: `clk`, binary observed click for a logged impression; not a conversion label.
- Original full-audit train size: **21,929,911** rows and **1,129,076** clicks.
- Original full-audit test size: **3,099,515** rows and **155,437** clicks.
- Original full-audit training SHA-256: `50c0862043f62aa8b44bdab46a54242d2bf1ae5d2d1e541923a7790e6a847408`.
- Original full-audit test SHA-256: `ff55e3d06c3b1c50c3a17da06459081151340b99193e3ec8291116a5ebcac111`.

These hashes identify the local source files audited earlier, not the derived feature CSVs. Derived-file checksums have not yet been recorded.

## 3. Reproducible preparation protocol

Implementation: `src/sponsored_ads/ctr_features.py`, invoked through `python -m sponsored_ads.ctr_features_cli`.

| Setting | Value |
|---|---|
| Source split | Published training data only |
| Sampling | Uniform deterministic row-index hash |
| Sampling rate | `0.05` |
| Validation assignment | Independent deterministic row-index hash |
| Validation fraction | `0.20` of selected rows |
| Seed | `2026` |
| Source row limit | `null` (full training scan) |
| Split guarantees | Row assignment only; **not** chronological or user-disjoint |

The code uses BLAKE2b-derived stable pseudo-random values from `(seed, zero-based source row index, salt)`; `sample` and `split` use separate salts. This is reproducible for the same ordered input and implementation, but shuffling the source file changes membership. Sampling is not stratified by label or user.

Reproduction command:

```bash
python -m sponsored_ads.ctr_features_cli --sample-rate 0.05
```

Expected outputs (under the default artifact directory):

```text
artifacts/m03a/train_features.csv
artifacts/m03a/validation_features.csv
artifacts/m03a/sample_manifest.json
```

A `--limit` run is only a prefix-based smoke test; it is not representative random sampling of the full source.

## 4. Observed counts and sanity checks

| Measure | Development train | Development validation | Combined |
|---|---:|---:|---:|
| Impressions | 876,719 | 219,398 | 1,096,117 |
| Clicks | 44,590 | 11,373 | 55,963 |
| Non-clicks | 832,129 | 208,025 | 1,040,154 |
| Observed CTR | 5.0860% | 5.1837% | 5.1056% |

The sampled fraction is **4.9983%** of source training rows. Validation comprises approximately **20.02%** of selected rows. Similar click rates are a descriptive sanity check only: they do not prove absence of distribution shift, user overlap, label leakage, or representative coverage of rare entities.

## 5. Feature contract (24 output columns)

**Label:** `clk`.

**Static categorical fields (16):** `userid`, `cms_segid`, `cms_group_id`, `final_gender_code`, `age_level`, `pvalue_level`, `shopping_level`, `occupation`, `new_user_class_level`, `adgroup_id`, `cate_id`, `campaign_id`, `customer`, `brand`, `pid`, `btag`.

**Numeric fields (7):** `price`, `history_length`, `category_history_count`, `brand_history_count`, `behavior_history_count`, `last_category_matches`, `last_brand_matches`.

Feature meanings from the implementation:

- `history_length`: number of tokens in `cate_his`.
- `category_history_count`: occurrences of the candidate `cate_id` in `cate_his`.
- `brand_history_count`: occurrences of the candidate `brand` in `brand_his`.
- `behavior_history_count`: occurrences of the candidate `btag` in `btag_his`.
- `last_category_matches`: whether the **last supplied** category token equals `cate_id`.
- `last_brand_matches`: whether the **last supplied** brand token equals `brand`.
- `price`: source numeric value, retained without rescaling.

The word **last** describes array position only. Without event timestamps, it is not verified to mean most recent in time. Category, brand, and behavior histories are source-provided sequences; no separate raw behavioral log was joined.

Categorical IDs are nominal tokens, **not continuous quantities**. Model pipelines must use categorical encoding or a suitable categorical treatment, and fit any learned encoders using development training rows only.

## 6. Risks, limits, and permitted claims

1. **Point-in-time uncertainty:** Source histories are not independently proven to precede the target impression. Do not assert leakage-free causal or chronological features.
2. **Split uncertainty:** There is no timestamp in the processed 21-column input schema. Development validation is neither chronological nor user-disjoint. Repeated users, ads, and campaigns may occur across splits.
3. **Exposure/selection bias:** Click labels are observed for historically displayed ads. Missing/unshown candidates are not verified negative examples. Offline CTR prediction is not unbiased policy evaluation.
4. **No auction ground truth:** Bids, budgets, auction payments, pacing, and policy outcomes must be explicitly simulated later; none is observed here.
5. **No purchase-attribution label:** `clk` is a click label; these files do not establish ad-attributed conversions.
6. **Sampling scope:** Row-index sampling is reproducible but can underrepresent rare categories, users, or campaigns. Analyze coverage before interpreting slice metrics.
7. **Artifact verification:** The supplied manifest records counts and configuration, but this report has not independently recomputed CSV checksums or scanned every derived row.

## 7. Validation evidence and acceptance

- Previous user terminal output: `pytest` passed **17 tests** after Milestone 2.
- Milestone 3A patch was previously described as having **29 tests** in an isolated verification environment; the user's local Milestone 3A `pytest` output has **not** been provided in this conversation.
- The uploaded `m03a.zip` contains the two feature CSVs and the manifest, demonstrating that preparation outputs were produced.
- The manifest reports a full training-file scan and preserves `history_point_in_time_verified: false`.

**M03A data-generation acceptance:** Met based on supplied artifacts and manifest.
**M03A local test-suite acceptance:** Pending user confirmation.
**M03B model-quality acceptance:** Not applicable yet; no models trained.

## 8. Next-step gate (M03B)

Before modeling: run `pytest` on the current checkout; commit this report with the implementation; record source and derived file hashes for future experiment provenance. In M03B, compare static-feature logistic regression, static-feature LightGBM, and LightGBM with history-derived features on the fixed development split. Report log loss, ROC-AUC, PR-AUC, calibration, class prevalence, resource usage, and slice performance where sample sizes permit. Reserve the published test split for a single final model-selection checkpoint, not iterative tuning. Any observed improvements are offline predictive associations, not causal ad-performance lift.
