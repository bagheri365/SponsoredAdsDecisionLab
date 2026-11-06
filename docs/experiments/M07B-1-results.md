# M07B-1: Single-pass scoring diagnostics — results

Status: Locally verified.

## Verification

- Focused tests: 11 passed.
- Full suite: 357 passed, 1 existing LightGBM warning.
- Raw persisted-model experiment: completed twice.
- Sigmoid persisted-model experiment: completed once.
- Raw complete-file SHA-256 identical across both runs.

## Baseline comparison

M07B-1 was compared against the accepted M07A-3 persisted-model
smoke experiment using the same model, feature rows, synthetic
auction configuration, and budget inputs.

| Check | Raw | Sigmoid |
|---|---|---|
| Report SHA-256 unchanged | True | True |
| Predicted CTRs unchanged | True | True |
| Auction decision unchanged | True | True |
| Ledger snapshots unchanged | True | True |
| Ledger event count unchanged | True | True |

## Artifact hashes

Raw embedded report SHA-256:
d23bc6a4d5857dd0ab320f4d3afa74d4a63a044206e4ae06cf6d7659ccb7a2f0

Raw complete-file SHA-256 (both runs):
f31db9166e4f13c19180f1bc84761590cb42ccaa6bab58794e5324bb7c006d53

Sigmoid embedded report SHA-256:
0a52e656ae3da19d2ea0911f7c4e27400a9ee34e591e8a7b2910d92ebf323576

## Interpretation

M07B-1 exposes scored candidates from the integrated model-auction
service, allowing the smoke runner to reuse inference results rather
than scoring the same feature rows twice.

The regression comparison confirms unchanged decisions, predictions,
and ledger effects for the tested persisted-model scenarios.

These experiments use synthetic bids, advertisers, and budgets.
They do not establish online performance, revenue lift, or
cross-environment reproducibility.
