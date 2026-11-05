# M04C — Exposure-conditioned observed-click retrieval experiment

**Status:** Completed exploratory offline evaluation; not an estimate of unbiased online recall or causal advertising lift.

## Dataset and protocol

- Training-derived observed-ad catalog: **788,275** ad groups.
- Published TaobaoAd_x1 test split: **155,437** valid clicked impressions.
- Deterministic bottom-k BLAKE2b sample of **1,000** clicked impressions, seed **2026**.
- Pool size **100**; K = **10, 50, 100**.
- Compared popularity, category/brand affinity, and reciprocal-rank fusion.

## Full-test catalog coverage

- Valid clicked impressions: **155,437**.
- Clicked impressions whose ad appears in the training catalog: **152,777**.
- Coverage: **98.29%**.
- Unrecoverable clicked impressions due to missing catalog membership: **2,660**.

## Observed-click Recall@K

| K | Popularity | Affinity | Fusion |
|---|---:|---:|---:|
| 10 | 1.00% | 1.40% | 1.50% |
| 50 | 2.80% | 4.20% | 4.80% |
| 100 | 4.90% | 6.00% | 7.10% |

## Paired recall differences

Differences and percentile bootstrap confidence intervals are expressed in percentage points.

| K | Comparison | Difference | 95% bootstrap CI | Fusion-only hits | Comparator-only hits |
|---|---|---:|---|---:|---:|
| 10 | Fusion − popularity | +0.50 pp | [-0.20, +1.20] pp | 9 | 4 |
| 10 | Fusion − affinity | +0.10 pp | [-0.60, +0.80] pp | 6 | 5 |
| 50 | Fusion − popularity | +2.00 pp | [+0.90, +3.30] pp | 28 | 8 |
| 50 | Fusion − affinity | +0.60 pp | [-0.60, +1.80] pp | 21 | 15 |
| 100 | Fusion − popularity | +2.20 pp | [+0.70, +3.70] pp | 42 | 20 |
| 100 | Fusion − affinity | +1.10 pp | [-0.20, +2.40] pp | 28 | 17 |

The intervals use **2,000 deterministic impression-level bootstrap replicates**. They assume independent sampled impressions and are unadjusted for multiple comparisons. Repeated shoppers or ads may violate independence.

At K=50 and K=100, the fusion-minus-popularity intervals exclude zero. The fusion-minus-affinity intervals include zero at every evaluated K. These observations do not establish online lift or generalize beyond the evaluation assumptions.

## Runtime

- CSV scan: **39.2434 s**.
- Index build: **6.8742 s**.
- Median combined retrieval latency: **96.6414 ms**.

## Earlier 100-click exploratory sample

Popularity recovered **0/2/3**, affinity **2/3/4**, and fusion **1/4/5** clicked impressions at K=10/50/100. The 100-click and 1,000-click samples use the same seed and are not independent replications.

## Limitations

- Logged clicked impressions are exposure-conditioned observations, not counterfactual relevance labels.
- The published train/test split has not been verified as temporal or user-disjoint.
- Shopper-history point-in-time validity and event ordering are unverified.
- The observed training catalog is not a request-time eligible-ad inventory.
- Bootstrap intervals do not account for shopper clustering, exposure bias, or multiple comparisons.
- The experiment uses one deterministic sample; retrieval parameters must not be tuned on the published test split.
- Query latency is exploratory and does not establish production serving performance.

## Reproduction

```bash
python -m sponsored_ads.retrieval_evaluation_runner \
  --test-csv data/TaobaoAd_x1/test.csv \
  --catalog artifacts/m04a/catalog.json \
  --sample-size 1000 --seed 2026 --pool-size 100 \
  --output artifacts/m04c/evaluation-1000-paired.json
```
