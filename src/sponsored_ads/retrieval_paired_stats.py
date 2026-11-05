"""Paired, impression-level descriptive uncertainty for observed-click recovery.

Percentile bootstrap is conditional on the observed sample and assumes sampled
impressions are independent; repeated users/ads can violate that assumption.
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence


def paired_recall_differences(
    outcomes: Sequence[Mapping[str, bool]], *,
    comparisons: Sequence[tuple[str, str]] = (("fused", "popularity"), ("fused", "affinity")),
    repetitions: int = 2000,
    seed: int = 2026,
) -> dict:
    """Return paired difference (first minus second) and percentile bootstrap CI.

    Outcomes are one Boolean hit indicator per sampled clicked impression.
    The bootstrap reuses the SAME resampled impression indices for both methods.
    """
    if repetitions < 2:
        raise ValueError("repetitions must be >= 2")
    if not outcomes:
        return {f"{a}_minus_{b}": None for a, b in comparisons}
    n = len(outcomes)
    result = {}
    for a, b in comparisons:
        if any(a not in row or b not in row for row in outcomes):
            raise ValueError(f"missing paired outcome for {a} or {b}")
        diffs = [int(row[a]) - int(row[b]) for row in outcomes]
        rng = random.Random(seed)
        draws = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                       for _ in range(repetitions))
        low = draws[int(0.025 * (repetitions - 1))]
        high = draws[int(0.975 * (repetitions - 1))]
        result[f"{a}_minus_{b}"] = {
            "difference": sum(diffs) / n,
            "ci95_percentile": [low, high],
            "first_only_hits": diffs.count(1),
            "second_only_hits": diffs.count(-1),
            "both_hit": sum(bool(row[a]) and bool(row[b]) for row in outcomes),
            "neither_hit": sum(not row[a] and not row[b] for row in outcomes),
            "n": n,
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": seed,
        }
    return result
