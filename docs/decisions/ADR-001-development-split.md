# ADR-001: Deterministic row-index development split

**Status:** Accepted for M03A; revisit when event timestamps or source split provenance become available.

**Context:** TaobaoAd_x1 supplies published train/test files, with no timestamps. A training-only validation partition is required for model selection.

**Decision:** Sample training rows using a stable seeded BLAKE2b row-index hash; assign sampled rows to train or validation using a separate hash salt. Retain original label prevalence in expectation and reserve published test data.

**Alternatives:** Chronological split (not supported by columns); user-disjoint split (could better test new-user generalization but changes the evaluation target); stratified split (not necessary at dataset scale and complicates streaming); naive first-N slice (may be ordered and unrepresentative).

**Consequences:** Fully reproducible, constant-memory, easy to audit; **not** independent across repeated users/ads and **not** temporally leakage-safe. Report this limitation beside all downstream metrics.
