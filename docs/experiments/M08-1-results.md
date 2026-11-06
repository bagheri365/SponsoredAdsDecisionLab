# M08-1 — Deterministic Synthetic Evaluation Results

## Verification

- Baseline commit: `4404d6a` (M07B-4); M08-1 changes not yet committed.
- Focused regression suite: **9 passed**.
- Full regression suite: **388 passed, 1 warning**.
- Warning: pre-existing LightGBM/scikit-learn feature-name warning in `test_ctr_baselines.py::test_lightgbm_smoke`.
- `git diff --check`: passed.
- Command: `python -m sponsored_ads.m08_synthetic_evaluation --output artifacts/m08/m08_1_report.json`.
- CLI reported `allocation_agreement: true`.
- Report file size: **8,189 bytes** (as reported by the local run).

## Provenance

- Input SHA-256: `a2db11807b5dbdf22b62bbc34c14b6159390054623d27c666ec3f41b48ab0446`.
- Embedded report SHA-256: `896cf301d14b606d3e1e80b5696e43bd072adf6395441d2b1b07cd45d39fd54b`.
- These are hashes emitted by the M08-1 harness; no independent full-file SHA-256 comparison was performed.

## Aggregate results

| Metric | `first_price` | `score_second_price` |
|---|---:|---:|
| Scenarios | 6 | 6 |
| Filled | 3 | 3 |
| Fill rate | 0.5 | 0.5 |
| Relevance agreement on filled | 2/3 | 2/3 |
| Mean winning score | 0.30000000000000004 | 0.30000000000000004 |
| Mean conditional CPC if clicked | 2.0 | 0.8333333333333334 |
| Total initial budget (cents) | 2499 | 2499 |
| Total reserved (cents) | 600 | 600 |
| Reserved fraction | 0.24009603841536614 | 0.24009603841536614 |

Each policy had one `no_candidates`, one `no_eligible_or_above_reserve`, and one `insufficient_budget` no-fill outcome. No-fill scenarios emitted zero ledger events.

## Scenario findings

| Scenario | Winner (both policies) | Key finding |
|---|---|---|
| `score_vs_relevance` | `a` | Synthetic relevance oracle prefers `b`; relevance disagreement |
| `budget_exclusion` | `b` | `a` excluded as unaffordable; repriced conditional CPC is 3.0 first-price versus 0.0 score-second-price; winner reserves 300 cents |
| `tie_break` | `a` | Equal auction scores resolve lexically; oracle also selects `a` |
| `no_candidates` | none | `no_candidates` |
| `below_reserve` | none | `no_eligible_or_above_reserve` |
| `insufficient_budget` | none | `insufficient_budget` |

In `score_vs_relevance`, conditional CPC is 2.0 (`first_price`) versus approximately 1.5 (`score_second_price`); in `tie_break`, both are 1.0. Across both policies, winner reservations are 200, 300, and 100 cents for the three filled scenarios, respectively, and every reported account has zero spent cents.

## Interpretation and limitations

All scores, bids, relevance labels, reserves, and budgets are synthetic and fixed. Scenarios use independent ledgers per policy; this is **not** a shared sequential budget or pacing experiment. Relevance agreement is evaluated only on filled scenarios, against the fixture's synthetic oracle. Conditional CPC is a hypothetical payment **if clicked**, not realized revenue; the harness reserves maximum bids and does not settle or simulate clicks. Results do not establish online relevance, observed clicks, causal lift, revenue, production budget utilization, concurrency, or platform behavior.

## Remaining release checks

- Review the staged diff and confirm only the intended M08-1 source, tests, ADR, milestone, and this results document are staged.
- Keep local review notes and downloaded patches untracked.
- Do not claim a Git commit or push until terminal evidence confirms them.
