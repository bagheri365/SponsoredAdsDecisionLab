# ADR-004: Exposure-conditioned clicked-ad retrieval evaluation

**Status:** Accepted for M04C evaluation design; empirical validation pending.

## Context

TaobaoAd_x1 contains observed impressions and click labels, but no complete per-request eligible-ad set, auction log, or verified point-in-time history. Training-catalog retrieval can therefore be checked against an observed clicked ad, but not evaluated as a counterfactual serving policy.

## Decision

Evaluate each held-out clicked impression as a positive lookup task. Compare popularity, affinity, and fused retrieval with the same fixed K and candidate-pool configuration. Report both all-click and catalog-covered-click denominators, with explicit missing-catalog coverage. Track provenance for successful fused hits without treating it as causal source contribution.

Keep test.csv strictly out of catalog construction, indexing, popularity estimation, and tuning. Label all results exploratory until temporal and history integrity can be verified.

## Alternatives rejected

- **Treat unclicked impressions as negatives:** unshown alternatives are missing and exposed nonclicks do not establish irrelevant candidates.
- **Report precision@K or unbiased recall:** true per-request relevant/eligible sets are unknown.
- **Filter missing-catalog clicks:** hides an important system coverage limitation.
- **Use test-set clicks to tune fusion weights:** contaminates the final evaluation.

## Consequences

The metric is useful for a controlled retrieval comparison, not proof of user benefit, auction performance, online CTR, or incremental ad revenue. Follow-up work must include streaming evaluation, overlap checks, and explicit temporal uncertainty.
