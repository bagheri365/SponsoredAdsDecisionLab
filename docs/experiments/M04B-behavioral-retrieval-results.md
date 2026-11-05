# M04B — Behavioral retrieval: real-catalog smoke results and profiling

## Status and provenance

**Functional integration and profiling validated; exact top-K optimization awaiting local benchmark.** These values were reported from a local Mac running Python 3.10.15, pytest 9.2.0. Full repository suite: **69 passed in 3.32 s**. Exact CPU, RAM, OS release, catalog checksum, and run-to-run variation have not yet been recorded. Results are not independently reproduced in this patch build environment.

## Observed results (2026-11-05)

| Measurement | Popularity-only, empty history | Behavioral, supplied history |
|---|---:|---:|
| Catalog ads | 788,275 | 788,275 |
| Index build wall time | 19.758 s | 18.6545 s |
| Median query wall time | 0.202 ms | 69.3712 ms |
| Minimum query wall time | Not supplied | 68.6586 ms |
| Maximum query wall time | Not supplied | 107.2098 ms |
| Query repetitions | 10 | 10 |
| History length | 0 | 2 |
| Peak Python-traced allocations | Not supplied | 730,055,708 bytes (~696.2 MiB) |

The behavioral-to-popularity median ratio is approximately **343.4x** for these *different workloads*. Do not interpret this as a general speedup/slowdown factor for equivalent traffic. Tracemalloc does not report complete process RSS. No warm-up, concurrency, representative query distribution, or p95/p99 measurement was provided.

The supplied behavioral query used `cate_his=6261^4281`, `brand_his=39211^74909`, `btag_his=1^1`, top-K 10, default pool size 100. Its ten returned candidates were five affinity-only and five popularity-only; no dual-source candidate appeared in the returned top ten. These are heuristic RRF scores, not calibrated CTR or click relevance.

## Observed profiling (2026-11-05; supplied Mac run)

The user ran 20 repetitions, history length 2, pool size 100, top-K 10, on the 788,275-ad catalog. All 74 repository tests passed in 3.49 seconds.

| Phase | Median (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|
| Affinity | 63.4295 | 60.7422 | 105.5526 |
| Popularity | 0.2279 | 0.2194 | 0.2587 |
| Fusion | 0.2763 | 0.2688 | 0.3028 |

The affinity cProfile run reported 100,774 function calls and 0.093 s total, including `sorted` with 0.070 s cumulative time and approximately 50,325 key evaluations. These instrumented timings cannot be directly compared with the uninstrumented medians. Sorting the scored candidate pool is the primary measured opportunity for optimization; score accumulation is still required for exact results.

**Decision:** replace full sorting with `heapq.nsmallest(top_k, ..., key=lambda item: (-score, ad_id))`. This preserves the existing total ordering and exact results for valid finite scores, while selecting only the required top-K from a potentially large candidate set. It does not cap candidate enumeration or guarantee sublinear retrieval. Add randomized full-sort equivalence tests. No latency improvement is claimed until the optimized version is benchmarked locally.

## Reproduce profiling

```bash
pytest
python -m sponsored_ads.retrieval_profile \
  --catalog artifacts/m04a/catalog.json \
  --cate-his '6261^4281' \
  --brand-his '39211^74909' \
  --btag-his '1^1' \
  --top-k 10 --pool-size 100 --repetitions 20 \
  --output artifacts/m04b/profile.json
python -m json.tool artifacts/m04b/profile.json
```

The profiler measures affinity, popularity, and fusion phases individually, and records cProfile's top 20 affinity functions. It does not modify the retrieval algorithm. **Phase timings are not identical to end-to-end timings**, and cProfile instrumentation affects measured duration. The JSON is ignored by Git and should be summarized here after running it.

## Post-optimization benchmark — pending local run

Re-run the identical profiling command above after applying the exact top-K patch. Record affinity median/min/max, end-to-end median/min/max (using the integration CLI with the same history), Python-traced memory and catalog SHA-256. Compare against the **pre-optimization** affinity median of 63.4295 ms and end-to-end median of 69.3712 ms. Different repetition counts and profiling instrumentation limit strict comparisons. Retain the raw JSON under ignored `artifacts/m04b/` and record measured values in this report before closing M04B.

## Validity limitations

- Catalog derived from historical exposed impressions, not complete or current eligible inventory.
- Historical popularity is logging-policy biased; unexposed ads are not known negatives.
- No verified point-in-time catalog or shopper-history chronology; recency ordering is an assumption.
- Synthetic/manual history is not representative workload sampling.
- No observed-click Recall@K or independent held-out evaluation has been run.
- No production throughput, concurrency, memory RSS, or latency SLO has been established.

**M04B remains open** until profile results and decision are recorded. Do not commit an unsupported optimization or claim retrieval quality improvements.
