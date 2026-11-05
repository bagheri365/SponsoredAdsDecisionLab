# ADR-005 — Persist train-fitted CTR preprocessing for serving

**Status:** Accepted for M05A.

## Context

M03B LightGBM uses contiguous category IDs fitted on M03A training rows, with
unseen validation categories mapped to -1. The existing LightGBM `.txt` booster
files omit the external category dictionaries. The unused `_category_code`
BLAKE2b function is not equivalent. Logistic regression instead uses feature
hashing and log1p transforms and is outside this contract.

## Decision

Persist a versioned JSON contract beside each newly trained LightGBM booster.
It records exact ordered features, train-fitted mappings, unknown=-1, direct
float32 numeric conversion, and error-on-missing policy. The same categorical
encoding helper is used by training and serving. Bundle loading checks model
feature count and feature names; models without a matching contract fail closed.
Use complete M03A feature rows only. Do not invent user/history features from
an M04A ad-only catalog.

## Consequences

Old M03B booster artifacts without their fitted mappings **cannot be served
faithfully**; rerun training with M05A persistence enabled. JSON maps may be
large for high-cardinality fields and expose pseudonymous identifiers: keep
artifacts outside Git and apply appropriate access controls. Exact parity is
relative to M03A feature rows, not to unverifiable historical point-in-time
truth. Model and contract should be deployed atomically; count/name checks do
not authenticate the contents of a contract.
