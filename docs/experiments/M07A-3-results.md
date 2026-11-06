# M07A-3: Persisted-model auction smoke results

Status: Locally verified.

## Verification

- Focused tests: 7 passed.
- Full suite: 353 passed, 1 existing LightGBM warning.
- Raw experiment: two runs with identical complete-file SHA-256.
- Sigmoid experiment: completed successfully.

## Inputs

- Feature sample: first 3 rows of evaluation_features.csv.
- Feature file SHA-256: 31827ab978ebca614eaea7ee63754cacbf9cbe8644f846c163f8fdfdcee6b7c7
- Model SHA-256: 47d9611078d1f79fd4a25b1a00c45ba7cb920762d22e56e08182112ca2aee02c
- Contract SHA-256: 89497d6396079b9cf1d6d8353180a6757857e6df3be11235e7f45f3886c89daa
- Sigmoid artifact SHA-256: b2d6c5ff6ed077d090bc3461005fa4af12bda84d29259d029b271c6083f1a2b9
- Synthetic bids: $2.00, $1.50, $1.00.
- Synthetic advertiser budgets: 500 cents each.
- Auction policy: score_second_price.

## Observed results

| Metric | Raw | Sigmoid |
|---|---:|---:|
| Ad 1 CTR | 0.050736117828688206 | 0.05193289940843874 |
| Ad 2 CTR | 0.050044697375752345 | 0.05130378312542433 |
| Ad 3 CTR | 0.0598733313903252 | 0.06016614832582516 |
| Winner | synthetic-ad-1 | synthetic-ad-1 |
| Conditional CPC | 1.479558335880059 | 1.4818289670850098 |
| Reserved cents | 200 | 200 |
| Spent cents | 0 | 0 |
| Ledger events | 1 | 1 |

Raw embedded report SHA-256:
d23bc6a4d5857dd0ab320f4d3afa74d4a63a044206e4ae06cf6d7659ccb7a2f0

Raw complete-file SHA-256 (both runs):
f31db9166e4f13c19180f1bc84761590cb42ccaa6bab58794e5324bb7c006d53

Sigmoid embedded report SHA-256:
0a52e656ae3da19d2ea0911f7c4e27400a9ee34e591e8a7b2910d92ebf323576

## Interpretation and limitations

The persisted model and feature contract successfully produced CTR
predictions for complete feature rows. Both calibration policies
produced a filled synthetic auction and a correct budget reservation.

Bids, advertisers, budgets, and the auction scenario are synthetic.
This is not an observed auction replay, an online experiment, or
evidence of revenue lift.

The smoke runner currently scores rows separately for diagnostics
and again inside the model-auction service. A later milestone should
provide single-pass scoring diagnostics and replay verification.
