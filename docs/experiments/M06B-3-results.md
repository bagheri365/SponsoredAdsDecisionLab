# M06B-3 — Synthetic budget-aware auction experiment results

Status: Local full-suite tests, synthetic experiment, and deterministic replay verified.

## Declared input

- Initial budgets: `adv-a=250¢`, `adv-b=350¢`, `adv-c=0¢`.
- Six sequential synthetic auctions per policy, with fixed click flags; each policy uses a fresh ledger.
- Reference input SHA-256: `3d129f3420c4f73542cda0a352c47eb7802e9ead1999569de1c9803d906f00b6`.
- Reference embedded report SHA-256: `8eb2fa59100d50a88a25ef99b8c85b07ef863ac9b5fe2c641483915532285487`.

## Reference outcomes (synthetic only)

| Policy | Fills | No-fills | Ledger events | Settled cents | Replay |
|---|---:|---:|---:|---:|---|
| First price | 4 | 2 | 8 | 300¢ | Equal snapshots and event stream |
| Score second price | 4 | 2 | 8 | 151¢ | Equal snapshots and event stream |

The first click settles 200¢ under first price and 151¢ under score-second-price (the latter includes the documented conservative ceiling of `1.5000000000000002`). In the second auction, `a` is excluded for insufficient available budget and `b` wins; a simulated nonclick releases its entire 300¢ hold. A bidder with zero budget and a zero bid is admitted with zero settlement. Two steps produce no fill. The final single-bidder click settles 100¢ under first price and 0¢ under score-second-price with zero reserve.

**Do not interpret the 300¢ versus 151¢ comparison as observed revenue, lift, or production auction performance.** All bids, budgets, and click flags are invented scenario inputs. The policies' allocation sequences happen to match here; that is not a general guarantee.

## Local verification

- Git baseline: `2a8a80f` (M06B-2 committed; M06B-3 uncommitted).
- Full suite: **315 passed, 1 existing LightGBM feature-name warning**, in 3.52 seconds.
- Input SHA-256: `3d129f3420c4f73542cda0a352c47eb7802e9ead1999569de1c9803d906f00b6`.
- Embedded report SHA-256: `8eb2fa59100d50a88a25ef99b8c85b07ef863ac9b5fe2c641483915532285487`.
- First local JSON file SHA-256: `eea05615fadbe6924588ba3a7a1ec5ff05cd97f4a5bbd4ebee6ba543ccaee583`.
- Second local JSON file SHA-256: `eea05615fadbe6924588ba3a7a1ec5ff05cd97f4a5bbd4ebee6ba543ccaee583`.
- Deterministic output: **verified across two consecutive local executions**.
- Ledger replay: **verified by the experiment's snapshot and event-stream equality checks**.

These checks establish reproducibility for the tested local environment, not cross-platform bitwise reproducibility or real auction validity.
