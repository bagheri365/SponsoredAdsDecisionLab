"""Reproducible M04B real-catalog smoke test and local performance probe."""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

from sponsored_ads.behavioral_retrieval import ShopperHistory, retrieve_fused
from sponsored_ads.retrieval_integration import load_observed_catalog


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("artifacts/m04a/catalog.json"))
    parser.add_argument("--cate-his", default="")
    parser.add_argument("--brand-his", default="")
    parser.add_argument("--btag-his", default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--queries", type=int, default=10,
                        help="Repeated identical queries for a smoke benchmark, not a latency SLO")
    parser.add_argument("--output", type=Path, default=Path("artifacts/m04b/smoke.json"))
    args = parser.parse_args(argv)
    if args.queries < 1:
        parser.error("--queries must be positive")
    history = ShopperHistory.from_delimited(args.cate_his, args.brand_his, args.btag_his)
    tracemalloc.start()
    started = time.perf_counter()
    index = load_observed_catalog(args.catalog)
    build_seconds = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    durations_ms = []
    results = []
    for _ in range(args.queries):
        started = time.perf_counter()
        results = retrieve_fused(index, history, args.top_k, pool_size=args.pool_size)
        durations_ms.append((time.perf_counter() - started) * 1000)
    ordered = sorted(durations_ms)
    report = {
        "catalog_path": str(args.catalog), "catalog_ads": len(index.ads),
        "index_build_seconds": round(build_seconds, 4),
        "peak_python_traced_bytes": peak_bytes,
        "query_count": args.queries,
        "query_latency_ms_min": round(ordered[0], 4),
        "query_latency_ms_median": round(ordered[len(ordered) // 2], 4),
        "query_latency_ms_max": round(ordered[-1], 4),
        "history_length": len(history.cate_his),
        "results": [{"adgroup_id": r.adgroup_id, "score": r.score, "sources": list(r.sources)}
                    for r in results],
        "limitations": ["Not a point-in-time catalog", "No verified ad eligibility",
                        "Tracemalloc excludes some native memory", "Repeated queries are not representative traffic"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ads={len(index.ads)} build_s={build_seconds:.3f} "
          f"query_median_ms={report['query_latency_ms_median']:.3f}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
