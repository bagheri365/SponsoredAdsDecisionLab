# ADR-003 — Heuristic affinity and rank-based candidate fusion

**Status:** Accepted for M04B standalone foundation; empirical evaluation pending.

## Context

TaobaoAd_x1 contains historical category/brand sequences and observed ad metadata, but no verified per-request eligible inventory, timestamps, or unbiased exposure labels. M04A provides an observed-impression popularity catalog.

## Decision

Use an inverted index keyed by ad category and brand. Compute deterministic sequence-position-weighted affinity and combine independently ranked affinity and popularity lists with reciprocal rank fusion (RRF). Maintain explicit source attribution and exclusions. Keep the implementation separate from M04A's storage contract until its JSON structure is verified.

## Alternatives

- Raw-score addition: rejected because affinity and impression counts have incomparable scales.
- Learned retrieval embeddings: deferred until a defensible training and evaluation protocol is established.
- Item-item collaborative filtering: deferred; the processed dataset does not provide a verified item-interaction graph or unbiased negatives.
- Event-type weighting: deferred pending a documented mapping of behavior-tag semantics.

## Consequences

Pros: interpretable, deterministic, simple to unit test, no model training required. Cons: heuristic recency assumption, observed-exposure bias, in-memory index growth, no validated eligibility or point-in-time correctness. RRF is a fusion heuristic, not a probability or revenue estimate.
