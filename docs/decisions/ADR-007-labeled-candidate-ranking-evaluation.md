# ADR-007: Explicit labeled-candidate pre-ranking evaluation

Status: Accepted for M05C.

## Context

M05B scores complete candidate feature rows but the observed TaobaoAd_x1 test split does not provide complete request-time eligible inventories, counterfactual outcomes, or a verified point-in-time feature store. A retrieved candidate without a logged click is **not** thereby a valid negative.

## Decision

- Accept only caller-provided candidate groups with an explicit binary relevance label for **every** candidate and at least one positive per group.
- Report Hit@K and reciprocal rank of the first positive (MRR@K), macro-averaged over labeled groups. Multiple positives are permitted.
- Retain unscorable candidates and positives in denominators; report candidate-level feature coverage before top-K truncation.
- Use the M05B feature assembler and M05A inference contract unchanged. Fail closed on identity conflicts and invalid model probabilities.
- Profile assembly, model prediction, and sorting as separate wall-clock phases. These are diagnostic measurements, not SLA guarantees.

## Consequences

Synthetic labeled sets exercise evaluation correctness but do **not** establish ranking effectiveness on real traffic. Any future empirical report must identify label provenance, candidate-set construction, sampling, feature availability, and leakage risks. This API does not generate negatives or infer counterfactual labels from unexposed ads.
