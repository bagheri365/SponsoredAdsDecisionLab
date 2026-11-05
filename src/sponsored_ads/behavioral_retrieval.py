"""Deterministic shopper-affinity retrieval over an observed-ad catalog.

This module does not infer historical eligibility or point-in-time availability.
All scores are heuristic, not calibrated probabilities.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from heapq import nsmallest
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ObservedAd:
    adgroup_id: str
    cate_id: str
    brand: str
    impressions: int

    def __post_init__(self) -> None:
        if not self.adgroup_id or self.impressions < 0:
            raise ValueError("adgroup_id must be nonempty and impressions nonnegative")


@dataclass(frozen=True)
class ShopperHistory:
    cate_his: tuple[str, ...]
    brand_his: tuple[str, ...]
    btag_his: tuple[str, ...]

    @classmethod
    def from_delimited(cls, cate_his: str, brand_his: str, btag_his: str) -> "ShopperHistory":
        def parse(value: str) -> tuple[str, ...]:
            return tuple(value.split("^")) if value else ()
        result = cls(parse(cate_his), parse(brand_his), parse(btag_his))
        if len(result.cate_his) != len(result.brand_his) or len(result.cate_his) != len(result.btag_his):
            raise ValueError("history sequences must be aligned")
        return result


@dataclass(frozen=True)
class RetrievedAd:
    adgroup_id: str
    score: float
    sources: tuple[str, ...]


class BehavioralIndex:
    """In-memory category/brand inverted index; each ad appears at most once."""

    def __init__(self, ads: Iterable[ObservedAd]):
        self.ads: dict[str, ObservedAd] = {}
        by_category: dict[str, list[str]] = defaultdict(list)
        by_brand: dict[str, list[str]] = defaultdict(list)
        for ad in ads:
            if ad.adgroup_id in self.ads:
                raise ValueError(f"duplicate adgroup_id: {ad.adgroup_id}")
            self.ads[ad.adgroup_id] = ad
            if ad.cate_id:
                by_category[ad.cate_id].append(ad.adgroup_id)
            if ad.brand:
                by_brand[ad.brand].append(ad.adgroup_id)
        self.by_category = {k: tuple(v) for k, v in by_category.items()}
        self.by_brand = {k: tuple(v) for k, v in by_brand.items()}
        self.popular = tuple(sorted(self.ads, key=lambda a: (-self.ads[a].impressions, a)))

    def affinity(self, history: ShopperHistory, top_k: int, *, excluded: Iterable[str] = (),
                 category_weight: float = 1.0, brand_weight: float = 1.0) -> list[RetrievedAd]:
        """Sum recency-weighted category/brand matches; no popularity in affinity score.

        Later sequence positions are assumed more recent; ordering is not independently
        verified in TaobaoAd_x1. btag_his is retained for future event weighting.
        """
        if top_k < 0 or category_weight < 0 or brand_weight < 0:
            raise ValueError("top_k and weights must be nonnegative")
        if not (len(history.cate_his) == len(history.brand_his) == len(history.btag_his)):
            raise ValueError("history sequences must be aligned")
        if top_k == 0:
            return []
        excluded_ids = set(excluded)
        n = len(history.cate_his)
        cat_weights: Counter[str] = Counter()
        brand_weights: Counter[str] = Counter()
        for i, (cat, brand) in enumerate(zip(history.cate_his, history.brand_his)):
            recency = (i + 1) / n
            if cat:
                cat_weights[cat] += recency * category_weight
            if brand:
                brand_weights[brand] += recency * brand_weight
        scores: dict[str, float] = defaultdict(float)
        for cat, weight in cat_weights.items():
            for ad_id in self.by_category.get(cat, ()):
                scores[ad_id] += weight
        for brand, weight in brand_weights.items():
            for ad_id in self.by_brand.get(brand, ()):
                scores[ad_id] += weight
        ranked = nsmallest(top_k, ((ad_id, score) for ad_id, score in scores.items()
                                   if score > 0 and ad_id not in excluded_ids),
                           key=lambda item: (-item[1], item[0]))
        return [RetrievedAd(ad_id, score, ("affinity",)) for ad_id, score in ranked]

    def popularity(self, top_k: int, *, excluded: Iterable[str] = ()) -> list[RetrievedAd]:
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        excluded_ids = set(excluded)
        output = []
        for ad_id in self.popular:
            if ad_id in excluded_ids:
                continue
            output.append(RetrievedAd(ad_id, float(self.ads[ad_id].impressions), ("popularity",)))
            if len(output) == top_k:
                break
        return output if top_k else []


def fuse_ranked(affinity: Sequence[RetrievedAd], popularity: Sequence[RetrievedAd],
                top_k: int, *, excluded: Iterable[str] = (), rrf_k: int = 60) -> list[RetrievedAd]:
    """Reciprocal-rank fusion (RRF), source attribution and stable ID tie-breaks.

    Input scores may have different units; only within-source ranks are used.
    """
    if top_k < 0 or rrf_k <= 0:
        raise ValueError("top_k must be nonnegative and rrf_k positive")
    excluded_ids = set(excluded)
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, set[str]] = defaultdict(set)
    for name, ranked in (("affinity", affinity), ("popularity", popularity)):
        seen: set[str] = set()
        for rank, result in enumerate(ranked, start=1):
            ad_id = result.adgroup_id
            if ad_id in seen:
                raise ValueError(f"duplicate {name} result: {ad_id}")
            seen.add(ad_id)
            if ad_id in excluded_ids:
                continue
            scores[ad_id] += 1.0 / (rrf_k + rank)
            sources[ad_id].add(name)
    ordered = sorted(scores, key=lambda ad_id: (-scores[ad_id], ad_id))
    return [RetrievedAd(ad_id, scores[ad_id], tuple(sorted(sources[ad_id])))
            for ad_id in ordered[:top_k]]


def retrieve_fused(index: BehavioralIndex, history: ShopperHistory, top_k: int,
                   *, pool_size: int = 100, excluded: Iterable[str] = ()) -> list[RetrievedAd]:
    if top_k < 0 or pool_size < top_k:
        raise ValueError("pool_size must be >= top_k >= 0")
    excluded_ids = frozenset(excluded)
    return fuse_ranked(index.affinity(history, pool_size, excluded=excluded_ids),
                       index.popularity(pool_size, excluded=excluded_ids),
                       top_k, excluded=excluded_ids)
