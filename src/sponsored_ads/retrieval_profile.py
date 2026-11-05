"""Isolated, repeatable phase profiling for M04B retrieval.

This measures repeated identical queries, not production traffic or tail SLOs.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import statistics
import time
from pathlib import Path

from sponsored_ads.behavioral_retrieval import ShopperHistory, fuse_ranked
from sponsored_ads.retrieval_integration import load_observed_catalog


def profile_phases(index, history: ShopperHistory, *, top_k: int = 10,
                   pool_size: int = 100, repetitions: int = 20) -> dict:
    """Time affinity, popularity, fusion separately, then profile affinity calls."""
    if repetitions < 1 or top_k < 1 or pool_size < top_k:
        raise ValueError("repetitions and top_k must be positive; pool_size >= top_k")
    samples = {"affinity": [], "popularity": [], "fusion": []}
    output = []
    for _ in range(repetitions):
        start = time.perf_counter()
        affinity = index.affinity(history, pool_size)
        samples["affinity"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        popularity = index.popularity(pool_size)
        samples["popularity"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        output = fuse_ranked(affinity, popularity, top_k)
        samples["fusion"].append((time.perf_counter() - start) * 1000)
    profiler = cProfile.Profile()
    profiler.enable()
    index.affinity(history, pool_size)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(20)
    return {
        "catalog_ads": len(index.ads),
        "history_length": len(history.cate_his),
        "top_k": top_k, "pool_size": pool_size, "repetitions": repetitions,
        "phase_latency_ms": {
            name: {"min": round(min(values), 4),
                   "median": round(statistics.median(values), 4),
                   "max": round(max(values), 4)}
            for name, values in samples.items()
        },
        "affinity_cprofile_top20": stream.getvalue(),
        "result_ids": [result.adgroup_id for result in output],
        "limitations": [
            "Repeated identical queries are not representative traffic",
            "Sequential phase timings are not equivalent to full end-to-end timing",
            "cProfile instruments calls and changes execution time",
            "No verified point-in-time inventory or shopper history ordering",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("artifacts/m04a/catalog.json"))
    parser.add_argument("--cate-his", default="")
    parser.add_argument("--brand-his", default="")
    parser.add_argument("--btag-his", default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m04b/profile.json"))
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.top_k < 1 or args.pool_size < args.top_k:
        parser.error("repetitions and top-k must be positive; pool-size >= top-k")
    history = ShopperHistory.from_delimited(args.cate_his, args.brand_his, args.btag_his)
    index = load_observed_catalog(args.catalog)
    report = profile_phases(index, history, top_k=args.top_k,
                            pool_size=args.pool_size, repetitions=args.repetitions)
    report["catalog_path"] = str(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, stats in report["phase_latency_ms"].items():
        print(f"{name}: median={stats['median']:.3f} ms")
    print(f"Profile: {args.output}")


if __name__ == "__main__":
    main()
