# M06A-3 — Synthetic auction experiment results

Status: Local tests, synthetic experiment, and deterministic replay verified.

## Reproduction

```bash
python -m sponsored_ads.auction_experiment --output artifacts/m06a3/synthetic_auction_report.json
pytest -q
shasum -a 256 artifacts/m06a3/synthetic_auction_report.json
```

## Observed execution

- Date: 2026-11-05 (conversation date; local execution date not independently recorded)
- Git baseline: `0d92f0e` (M06A-2 committed; M06A-3 patch uncommitted)
- Platform: macOS; Python 3.10 virtual environment (exact patch versions not captured)
- Tests: **245 passed, 1 existing LightGBM feature-name warning**, in 3.59 seconds
- `git diff --check`: clean
- Synthetic scenarios: **8**
- Allocation agreement across first-price and score-based second-price: **true**
- Synthetic input SHA-256: `571beb3a82bc23ab97bb505a40fc2dae42cbcf3a2eb6d067da78b841a55ce8d3`
- First-run JSON report SHA-256: `736668b07f1a5030511188279f2c731c9518545e329d3acf87fdc526800fbda2`
- Deterministic replay: **verified; two consecutive local runs produced identical report SHA-256 checksums**

## Scenario outcomes

All conditional CPC and proxy values are simulated, not realized charges or revenue. Values are shown rounded for readability; the JSON artifact retains floating-point outputs.

| Scenario | Winner | First-price conditional CPC | First-price CTR × CPC proxy | Score-second-price conditional CPC | Score-second-price CTR × CPC proxy |
|---|---|---:|---:|---:|---:|
| `two_bidders` | a | 2.00 | 0.400 | 1.50 | 0.300 |
| `reserve_zero` | a | 2.00 | 0.400 | 1.50 | 0.300 |
| `reserve_2_5` | b | 3.00 | 0.300 | 2.50 | 0.250 |
| `exact_tie` | a | 1.00 | 0.200 | 1.00 | 0.200 |
| `sole_bidder` | solo | 1.50 | 0.375 | 0.50 | 0.125 |
| `zero_ctr` | a | 2.00 | 0.000 | 0.00 | 0.000 |
| `no_admitted_bidder` | none | N/A | N/A | N/A | N/A |
| `ineligible_high_score` | b | 1.00 | 0.100 | 0.00 | 0.000 |

The reserve of 2.50 in `reserve_2_5` excludes candidate `a` and changes the allocation to `b`. Both policies allocate identically for each scenario because they share the same admission and ranking rules. A sole admitted bidder pays the reserve under the defined score-second-price convention. A zero-CTR winner yields zero proxy revenue; a non-admitted auction has no conditional CPC.

## Deterministic replay check

Two consecutive local executions produced identical report SHA-256 checksums:

- First run: `736668b07f1a5030511188279f2c731c9518545e329d3acf87fdc526800fbda2`
- Second run: `736668b07f1a5030511188279f2c731c9518545e329d3acf87fdc526800fbda2`

**Result: deterministic replay verified in the same local environment.**

## Caveats

All scenario candidates, CTRs, bids, and reserves are explicitly synthetic. Conditional CPC is not a realized charge. Predicted CTR × conditional CPC is an expected-revenue **proxy**, not observed revenue, profit, welfare, or causal lift. The experiment does not reconstruct TaobaoAd auction inventory, bids, payment rules, or counterfactual outcomes. The eight scenarios are targeted correctness checks, not representative traffic or a statistical performance estimate.
