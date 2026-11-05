# M04B — Real-catalog integration and smoke benchmark

## Objective

Adapt the M04A observed-ad catalog to the M04B behavioral index without changing either contract. The input is a JSON **list** of records with `adgroup_id`, `customer`, `cate_id`, `brand`, `impressions`, and `clicks`. Identifiers remain strings. Counts are nonnegative integers and clicks cannot exceed impressions. Duplicate ad IDs are rejected.

## Reproduce

```bash
pytest
python -m sponsored_ads.retrieval_integration_cli --queries 10
python -m sponsored_ads.retrieval_integration_cli --cate-his '6261^4281' --brand-his '39211^74909' --btag-his '1^1' --queries 10
```

The first run measures popularity fallback with empty history. The second demonstrates behavioral matching with a *manually supplied* history; it is not a representative sample. Both overwrite `artifacts/m04b/smoke.json` unless `--output` is supplied. Record machine specs, output, and input catalog hash before publishing measurements.

## Measurements

The CLI records catalog size, index construction wall time, Python-traced peak allocation, and min/median/max wall time across repeated identical top-K queries. Python tracing is **not** process RSS, and a few identical requests do **not** establish production p95/p99 latency. The full catalog is loaded as JSON before index construction, so memory usage may be substantial.

## Scientific boundaries

- The catalog is built from historical training impressions, not verified active eligible ads.
- Exposure counts are biased by the logging policy and are not a relevance or unbiased popularity estimate.
- The catalog has no timestamps; it is not safe to claim point-in-time retrieval correctness.
- This integration is a smoke/performance probe, **not** Recall@K evaluation.
- Any later evaluation must avoid catalog construction using its evaluation impressions, and report the unknown eligible-inventory denominator and historical-policy selection bias.
- Behavioral sequence ordering and event-type semantics are not independently verified. Recency weights remain assumptions.

## Acceptance

- [ ] Full repository tests pass locally.
- [ ] Real 788,275-ad catalog loads and query completes.
- [ ] Runtime and memory results captured with environment details.
- [ ] Experiment results documented before closing M04B.
