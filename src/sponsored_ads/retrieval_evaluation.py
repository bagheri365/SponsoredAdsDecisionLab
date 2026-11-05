"""Exposure-conditioned observed-click retrieval evaluation primitives.

No eligible-ad universe, counterfactual labels, or point-in-time guarantees are
inferred from TaobaoAd_x1. This module deliberately performs no data loading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ClickCase:
    """One held-out clicked impression, not a unique shopper or ad."""

    adgroup_id: str
    in_catalog: bool

    def __post_init__(self) -> None:
        if not self.adgroup_id:
            raise ValueError("clicked adgroup_id must be nonempty")


@dataclass(frozen=True)
class RetrievalObservation:
    """Predictions for one ClickCase; sources are supplied by the retriever."""

    case: ClickCase
    rankings: Mapping[str, Sequence[str]]
    fused_sources: Mapping[str, Sequence[str]] | None = None


@dataclass(frozen=True)
class MethodMetrics:
    clicks: int
    eligible_catalog_clicks: int
    hits: int
    hits_in_catalog: int
    observed_click_recall: float | None
    catalog_conditional_recall: float | None


@dataclass(frozen=True)
class EvaluationSummary:
    k: int
    clicked_impressions: int
    catalog_clicks: int
    catalog_coverage: float | None
    methods: dict[str, MethodMetrics]
    fused_hit_sources: dict[str, int]


def summarize_observed_clicks(
    observations: Iterable[RetrievalObservation],
    *,
    k: int,
    methods: Sequence[str] = ("popularity", "affinity", "fused"),
) -> EvaluationSummary:
    """Micro-average per clicked impression; never infer unobserved negatives.

    Each method must return a unique ordered ID list. Hits outside the catalog
    are rejected, since that indicates a catalog/prediction consistency bug.
    Empty evaluation sets yield None rather than silently reporting zero recall.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not methods or len(set(methods)) != len(methods) or any(not m for m in methods):
        raise ValueError("methods must be distinct nonempty names")
    count = covered = 0
    hits = {method: 0 for method in methods}
    covered_hits = {method: 0 for method in methods}
    sources = {"affinity_only": 0, "popularity_only": 0, "both": 0, "other": 0}
    for obs in observations:
        if set(obs.rankings) != set(methods):
            raise ValueError("each observation must supply exactly the requested methods")
        count += 1
        covered += int(obs.case.in_catalog)
        for method in methods:
            ranked = obs.rankings[method]
            if len(set(ranked)) != len(ranked):
                raise ValueError(f"duplicate IDs in {method} ranking")
            hit = obs.case.adgroup_id in ranked[:k]
            if hit and not obs.case.in_catalog:
                raise ValueError("hit for clicked ad absent from training catalog")
            hits[method] += int(hit)
            covered_hits[method] += int(hit and obs.case.in_catalog)
        if "fused" in methods and obs.case.adgroup_id in obs.rankings["fused"][:k]:
            labels = set((obs.fused_sources or {}).get(obs.case.adgroup_id, ()))
            if labels == {"affinity"}:
                sources["affinity_only"] += 1
            elif labels == {"popularity"}:
                sources["popularity_only"] += 1
            elif labels == {"affinity", "popularity"}:
                sources["both"] += 1
            else:
                sources["other"] += 1
    metrics = {
        method: MethodMetrics(
            clicks=count,
            eligible_catalog_clicks=covered,
            hits=hits[method],
            hits_in_catalog=covered_hits[method],
            observed_click_recall=hits[method] / count if count else None,
            catalog_conditional_recall=covered_hits[method] / covered if covered else None,
        )
        for method in methods
    }
    return EvaluationSummary(
        k=k,
        clicked_impressions=count,
        catalog_clicks=covered,
        catalog_coverage=covered / count if count else None,
        methods=metrics,
        fused_hit_sources=sources,
    )
