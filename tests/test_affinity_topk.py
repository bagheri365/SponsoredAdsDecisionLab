"""Exact top-K selection must preserve full-sort affinity semantics."""
from random import Random

from sponsored_ads.behavioral_retrieval import BehavioralIndex, ObservedAd, ShopperHistory


def _full_sort_reference(index, history, k, excluded=(), category_weight=1.0, brand_weight=1.0):
    """Independent brute-force reference: score every ad and sort the full result."""
    n = len(history.cate_his)
    excluded = set(excluded)
    scores = {ad.adgroup_id: 0.0 for ad in index.ads.values()}
    for field, values, weight in (("cate_id", history.cate_his, category_weight),
                                  ("brand", history.brand_his, brand_weight)):
        aggregated = {}
        for i, value in enumerate(values):
            if value:
                aggregated[value] = aggregated.get(value, 0.0) + (i + 1) / n * weight
        for value, contribution in aggregated.items():
            for ad in index.ads.values():
                if getattr(ad, field) == value:
                    scores[ad.adgroup_id] += contribution
    ranked = [(ad_id, score) for ad_id, score in scores.items() if score > 0 and ad_id not in excluded]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:k]


def test_exact_topk_matches_full_sort_randomized_catalog():
    rng = Random(20261105)
    ads = [ObservedAd(f"ad-{i:04d}", f"c{rng.randrange(6)}", f"b{rng.randrange(5)}", rng.randrange(100))
           for i in range(250)]
    index = BehavioralIndex(ads)
    for _ in range(20):
        cats = tuple(f"c{rng.randrange(7)}" for _ in range(5))
        brands = tuple(f"b{rng.randrange(6)}" for _ in range(5))
        history = ShopperHistory(cats, brands, ("1",) * 5)
        excluded = {f"ad-{i:04d}" for i in rng.sample(range(250), 12)}
        for k in (0, 1, 3, 10, 100, 300):
            expected = _full_sort_reference(index, history, k, excluded)
            actual = index.affinity(history, k, excluded=excluded)
            assert [(r.adgroup_id, r.score) for r in actual] == expected


def test_exact_topk_tie_breaks_by_ad_id_and_excludes():
    index = BehavioralIndex([ObservedAd(name, "c", "b", 1) for name in ("z", "a", "m", "b")])
    history = ShopperHistory(("c",), ("b",), ("1",))
    assert [r.adgroup_id for r in index.affinity(history, 2)] == ["a", "b"]
    assert [r.adgroup_id for r in index.affinity(history, 2, excluded={"a"})] == ["b", "m"]


def test_exact_topk_weighted_scoring_and_empty_matches():
    index = BehavioralIndex([ObservedAd("a", "c", "b", 1), ObservedAd("b", "x", "b", 2)])
    history = ShopperHistory(("c", "x"), ("b", "b"), ("1", "1"))
    expected = _full_sort_reference(index, history, 2, category_weight=2.0, brand_weight=0.5)
    assert [(r.adgroup_id, r.score) for r in index.affinity(history, 2, category_weight=2.0, brand_weight=0.5)] == expected
    assert index.affinity(ShopperHistory(("unknown",), ("unknown",), ("1",)), 10) == []
