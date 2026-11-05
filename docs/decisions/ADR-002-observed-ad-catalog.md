# ADR-002 — Reconstruct a catalog from observed training impressions

**Status:** Accepted for M04A baseline; not a production inventory design.

## Context

TaobaoAd_x1 supplies CTR impression rows and precomputed shopper histories,
but not the eligible ads available for each historical request. There is no
request timestamp to support point-in-time catalog reconstruction.

## Decision

Aggregate distinct `adgroup_id` values from published `train.csv` with stable
category, brand, and customer metadata. Rank by observed impression count,
with deterministic ID tie-breaking. Keep synthetic eligibility as explicit
query exclusions; do not infer it from unshown ads. Reject metadata conflicts.

## Alternatives considered

- **Clicks / smoothed CTR:** conflate exposure selection with relevance;
  potentially useful later as an explicitly biased benchmark.
- **Random unseen negatives:** falsely assert unshown ads were eligible.
- **Raw Taobao four-table inventory:** not present in the downloaded release.
- **External ad catalog:** would break the single-dataset project constraint.

## Consequences

Simple reproducible retrieval is available, but its popularity is historical
exposure frequency, not intrinsic shopper relevance. Memory is O(unique ads).
Future evaluation must distinguish offline diagnostics from counterfactual
claims, and future serving must use a versioned catalog and eligibility source.
