# M04A — Observed-Ad Catalog Build Results

**Project:** SponsoredAdsDecisionLab
**Milestone:** M04A — Retrieval foundation
**Status:** Full-data build and local tests passed; documentation prepared for commit
**Source:** `data/TaobaoAd_x1/train.csv` (RecZoo TaobaoAd_x1 published training split)

## 1. Objective and scope

Build a deterministic, impression-count-based retrieval baseline from ad IDs observed in the published training impressions. The catalog is a **reconstructed observed-ad catalog**, not a production inventory or a historical snapshot of eligible ads.

## 2. Reproduction commands

Run from the repository root with the project environment activated:

```bash
pytest
python -m sponsored_ads.retrieval_cli --limit 10000 --top-k 5
python -m sponsored_ads.retrieval_cli --top-k 10
```

The full run writes `artifacts/m04a/catalog.json` and `artifacts/m04a/manifest.json`. The `artifacts/` directory is generated output and should remain outside Git.

## 3. Validation evidence

The user-reported local run on macOS / Python 3.10.15 passed **46 tests** (`46 passed in 2.89s`). Both limited and full catalog-building CLI runs completed.

| Measure | 10,000-row smoke run | Full training run |
|---|---:|---:|
| Rows scanned | 10,000 | 21,929,911 |
| Clicks scanned | 517 | 1,129,076 |
| Distinct observed ad IDs | 7,485 | 788,275 |
| Retrieval top-K requested | 5 | 10 |

These are observed counts, not estimates of an eligible-ad inventory.

## 4. Full-run top-10 by historical impressions

| Rank | `adgroup_id` | Historical impressions |
|---:|---|---:|
| 1 | `710164` | 63,012 |
| 2 | `747157` | 23,023 |
| 3 | `736349` | 20,416 |
| 4 | `659289` | 18,949 |
| 5 | `725105` | 18,938 |
| 6 | `632644` | 16,725 |
| 7 | `659211` | 15,465 |
| 8 | `711852` | 15,098 |
| 9 | `632432` | 14,503 |
| 10 | `656398` | 13,943 |

**Scoring rule:** historical impression count. This is a frequency/exposure baseline, **not** a learned relevance score, CTR estimate, or propensity-corrected statistic.

## 5. Manifest and provenance

The full-run manifest reports:

```text
source: data/TaobaoAd_x1/train.csv
source_split: published train only
rows_scanned: 21929911
clicks_scanned: 1129076
catalog_ads: 788275
row_limit: None
score: historical impression count
catalog_scope: ads observed in sampled/published training impressions only
point_in_time_verified: False
eligibility: NOT VERIFIED; query exclusions are explicit simulator controls
warning: No test data used. Counts cannot be treated as online point-in-time features.
```

The source is the processed TaobaoAd_x1 dataset. The full training scan used the published training split only; the published test split was not used to construct the catalog.

## 6. Interpretation

The full scan demonstrates that the streaming aggregation can process approximately 21.9 million training impressions and produce a deterministic catalog of 788,275 distinct observed ad IDs. The most frequently observed ad has about 2.74 times the impression count of the second-ranked ad. This describes historical exposure concentration; it does not establish intrinsic ad popularity or shopper relevance.

## 7. Limitations and validity threats

1. **Exposure/selection bias:** only ads shown under the historical policy are observed. Unshown ads are not verified negatives.
2. **No historical eligibility:** inventory, budget, auction constraints, and request-time availability cannot be reconstructed from these CSV columns.
3. **No verified point-in-time correctness:** the processed dataset lacks impression timestamps. Aggregate impression counts built from the full training split must not be presented as request-time features for historical impressions.
4. **In-sample contamination risk:** evaluating retrieval on impressions used to construct the same catalog can overstate performance. Future offline evaluation must use a disjoint held-out portion and rebuild statistics only from its designated training portion; this still cannot remove historical exposure bias.
5. **No causal claims:** Recall@K on observed clicked ads is conditional on the logging policy and is not an unbiased measure of online candidate recall or revenue lift.
6. **Resource usage:** the builder streams rows but retains one aggregate per distinct observed ad ID in memory; this memory footprint grows with catalog cardinality.

## 8. Milestone acceptance and next actions

- [x] Automated retrieval tests passed locally (46 total repository tests).
- [x] Smoke run succeeded on 10,000 rows.
- [x] Full training-split catalog build succeeded.
- [x] Manifest reports scope, score definition, and point-in-time caveats.
- [x] Exact observed results and methodological limitations documented.
- [ ] Documentation and M04A implementation committed and pushed (pending user action).

**M04B direction:** add shopper-history category/brand retrieval and deterministic fusion, with explicit train/evaluation separation. Preserve popularity as a baseline. **Before serving CTR predictions**, persist and reload the fitted categorical preprocessing mappings with model artifacts.
