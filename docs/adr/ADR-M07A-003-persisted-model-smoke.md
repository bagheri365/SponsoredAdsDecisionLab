# ADR-M07A-003: Persisted model-to-auction smoke experiment

Status: Accepted (locally verified).

## Decision

Run the M07A-2 integration with the persisted M05F-A LightGBM model, versioned
feature contract, and the first three complete evaluation-feature CSV rows.
Use M05F-C raw inference by default; sigmoid requires an explicitly supplied,
hash-matched calibrator. Use deterministic synthetic bids (2.00, 1.50, 1.00
CPC), distinct synthetic advertisers with 500-cent budgets, and one
score-second-price auction. Record artifact provenance, feature-file hash,
CTR outputs, winner, and post-reservation ledger snapshots in canonical JSON.

## Boundaries

The sampled rows are logged impression features, not a counterfactual eligible
candidate set. Synthetic advertisers/bids/budgets are unrelated to logged ads.
This experiment is not an offline policy evaluation, a measurement of revenue,
or evidence of online lift. The report contains no labels and does not perform
click settlement. Repeatability is expected within a fixed local environment;
there is no claim of cross-version numerical identity. Artifacts must not be
modified between loading and inference. Do not commit large generated reports.
