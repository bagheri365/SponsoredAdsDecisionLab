"""Streaming, deterministic bottom-hash sample of clicked Taobao impressions.

Evaluation is exposure-conditioned and may suffer from history/time leakage.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from sponsored_ads.behavioral_retrieval import ShopperHistory, fuse_ranked
from sponsored_ads.retrieval_evaluation import ClickCase, RetrievalObservation, summarize_observed_clicks
from sponsored_ads.retrieval_integration import load_observed_catalog
from sponsored_ads.retrieval_paired_stats import paired_recall_differences

REQUIRED = frozenset({"clk", "adgroup_id", "cate_his", "brand_his", "btag_his"})
KS = (10, 50, 100)


def stream_sample_clicked(path: str | Path, *, sample_size: int, seed: int,
                          catalog_ids=None, coverage_counts=None):
    """Select lowest hash keys from all clicked rows; memory O(sample_size).

    Row number makes repeated impressions distinct; tie-break deterministic.
    A malformed clicked history is retained for accounting, not silently replaced.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    heap = []
    clicked = 0
    invalid_clicks = 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"test CSV missing columns: {sorted(REQUIRED - set(reader.fieldnames or []))}")
        for row_number, row in enumerate(reader, start=2):
            if row["clk"] not in ("0", "1"):
                raise ValueError(f"invalid clk at CSV row {row_number}")
            if row["clk"] != "1":
                continue
            clicked += 1
            if not row["adgroup_id"]:
                invalid_clicks += 1
                continue
            try:
                ShopperHistory.from_delimited(row["cate_his"] or "", row["brand_his"] or "", row["btag_his"] or "")
            except ValueError:
                invalid_clicks += 1
                continue
            if catalog_ids is not None and coverage_counts is not None:
                coverage_counts["valid_clicked_impressions"] += 1
                coverage_counts["clicked_ads_in_catalog"] += int(row["adgroup_id"] in catalog_ids)
            digest = hashlib.blake2b(f"{seed}:{row_number}".encode(), digest_size=8).digest()
            key = int.from_bytes(digest, "big")
            entry = (-key, -row_number, row_number, row)
            if len(heap) < sample_size:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
    selected = sorted(heap, key=lambda item: (item[2]))
    return [(item[2], item[3]) for item in selected], clicked, invalid_clicks


def evaluate(test_csv: str | Path, catalog: str | Path, *, sample_size: int = 100,
             seed: int = 2026, pool_size: int = 100) -> dict:
    if pool_size < max(KS):
        raise ValueError("pool_size must be >= 100 for K=100")
    build_start = time.perf_counter()
    index = load_observed_catalog(catalog)
    build_seconds = time.perf_counter() - build_start
    coverage_counts = {"valid_clicked_impressions": 0, "clicked_ads_in_catalog": 0}
    started = time.perf_counter()
    sample, total_clicks, invalid_clicks = stream_sample_clicked(
        test_csv, sample_size=sample_size, seed=seed,
        catalog_ids=index.ads, coverage_counts=coverage_counts)
    sample_seconds = time.perf_counter() - started
    observations = []
    paired_outcomes = {str(k): [] for k in KS}
    query_times = []
    for _, row in sample:
        history = ShopperHistory.from_delimited(row["cate_his"] or "", row["brand_his"] or "", row["btag_his"] or "")
        query_start = time.perf_counter()
        affinity = index.affinity(history, pool_size)
        popularity = index.popularity(pool_size)
        fused = fuse_ranked(affinity, popularity, pool_size)
        query_times.append((time.perf_counter() - query_start) * 1000)
        rankings = {"affinity": [x.adgroup_id for x in affinity],
                    "popularity": [x.adgroup_id for x in popularity],
                    "fused": [x.adgroup_id for x in fused]}
        for k in KS:
            paired_outcomes[str(k)].append({
                method: row["adgroup_id"] in ids[:k] for method, ids in rankings.items()
            })
        observations.append(RetrievalObservation(
            ClickCase(row["adgroup_id"], row["adgroup_id"] in index.ads),
            rankings,
            {x.adgroup_id: x.sources for x in fused},
        ))
    query_times.sort()
    summaries = {str(k): asdict(summarize_observed_clicks(observations, k=k)) for k in KS}
    return {
        "protocol": "exposure-conditioned clicked-impression recovery; exploratory only",
        "test_csv": str(test_csv), "catalog_path": str(catalog),
        "catalog_ads": len(index.ads), "seed": seed, "requested_sample_size": sample_size,
        "sampled_clicked_impressions": len(sample), "full_test_clicked_impressions": total_clicks,
        "invalid_clicked_rows": invalid_clicks, "valid_clicked_rows": total_clicks - invalid_clicks,
        "sample_selection": "bottom-k blake2b(seed:CSV_row_number) among valid clicked rows",
        "pool_size": pool_size, "k_values": list(KS), "summaries": summaries,
        "full_test_catalog_coverage": {
            **coverage_counts,
            "coverage": (coverage_counts["clicked_ads_in_catalog"] / coverage_counts["valid_clicked_impressions"]
                         if coverage_counts["valid_clicked_impressions"] else None),
            "denominator": "valid clicked impressions in full published test CSV",
        },
        "paired_recall_differences": {
            str(k): paired_recall_differences(paired_outcomes[str(k)], seed=seed) for k in KS
        },
        "sample_scan_seconds": round(sample_seconds, 4), "index_build_seconds": round(build_seconds, 4),
        "query_latency_ms": {"min": round(query_times[0], 4),
                             "median": round((query_times[(len(query_times)-1)//2] + query_times[len(query_times)//2])/2, 4),
                             "max": round(query_times[-1], 4)} if query_times else None,
        "python_version": platform.python_version(),
        "limitations": ["Published split not verified temporal or user-disjoint",
                        "History point-in-time validity and ordering unverified",
                        "Catalog is observed train ads, not request-time eligible inventory",
                        "Test clicks are logged exposures, not counterfactual relevance",
                        "One fixed-seed sample; bootstrap assumes independent clicked impressions despite possible repeated users/ads",
                        "Bootstrap intervals describe sampling variation only; not exposure or temporal bias",
                        "No tuning on published test"],
    }


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", default="data/TaobaoAd_x1/test.csv")
    parser.add_argument("--catalog", default="artifacts/m04a/catalog.json")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--output", default="artifacts/m04c/evaluation.json")
    args = parser.parse_args(argv)
    result = evaluate(args.test_csv, args.catalog, sample_size=args.sample_size,
                      seed=args.seed, pool_size=args.pool_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"clicked={result['full_test_clicked_impressions']} sampled={result['sampled_clicked_impressions']} "
          f"catalog_ads={result['catalog_ads']} output={output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
