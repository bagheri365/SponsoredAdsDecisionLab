# Milestone 2 — RecZoo TaobaoAd_x1 data audit

Input: `data/TaobaoAd_x1/train.csv` and `test.csv` (excluded from Git).

The published dataset has 21 columns, including three aligned history sequences:
`btag_his`, `cate_his`, and `brand_his`, separated by `^`. The `clk` column
is a binary click label. The files do not expose an impression timestamp.

## Smoke audit

```bash
python -m sponsored_ads.taobao_cli --limit 10000
```

This scans and validates the first 10,000 rows **per split**, but hashes both
entire files. Expect substantial disk I/O for hashing ~17 GiB. The manifest
will say `scanned_row_limit_per_file: 10000`; those row counts are not full
corpus totals.

## Full audit

```bash
python -m sponsored_ads.taobao_cli
```

The full audit streams both CSVs, so it uses bounded memory but may take time.
Outputs `artifacts/taobao_manifest.json` (ignored by Git).

## Boundaries

- No assumption that the published split is chronological.
- No guarantee that histories are point-in-time safe without dataset provenance.
- No raw behavior event timestamps or standalone purchase labels in this export.
- No bids, budgets, or auction logs; those remain simulated.
- A CTR label on logged impressions is not unbiased evidence for unseen ads.
- Full audit may reveal missing-value conventions not present in sample rows;
  adjust validation only after inspecting representative failures.
