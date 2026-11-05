import pytest

from sponsored_ads.behavioral_retrieval import (
    BehavioralIndex, ObservedAd, RetrievedAd, ShopperHistory, fuse_ranked, retrieve_fused,
)


def index():
    return BehavioralIndex([
        ObservedAd("a", "c1", "b1", 100),
        ObservedAd("b", "c2", "b2", 50),
        ObservedAd("c", "c1", "b2", 20),
    ])


def history():
    return ShopperHistory.from_delimited("c2^c1", "b2^b1", "1^1")


def test_history_alignment():
    with pytest.raises(ValueError):
        ShopperHistory.from_delimited("c1^c2", "b1", "1^1")


def test_affinity_prefers_recent_category_brand_match():
    assert index().affinity(history(), 3)[0].adgroup_id == "a"


def test_affinity_empty_history():
    assert index().affinity(ShopperHistory((), (), ()), 10) == []


def test_affinity_exclusions_and_topk():
    assert [x.adgroup_id for x in index().affinity(history(), 2, excluded={"a"})] == ["c", "b"]
    assert index().affinity(history(), 0) == []


def test_popularity_order():
    assert [x.adgroup_id for x in index().popularity(3)] == ["a", "b", "c"]


def test_fusion_deduplicates_and_attributes():
    results = fuse_ranked([RetrievedAd("b", 9, ("affinity",)), RetrievedAd("a", 5, ("affinity",))],
                          [RetrievedAd("a", 100, ("popularity",)), RetrievedAd("c", 20, ("popularity",))], 3)
    assert len(results) == 3
    assert results[0].adgroup_id == "a"
    assert results[0].sources == ("affinity", "popularity")


def test_fusion_exclusion():
    assert fuse_ranked([RetrievedAd("a", 1, ("affinity",))], [], 3, excluded={"a"}) == []


def test_fusion_duplicate_source_rejected():
    with pytest.raises(ValueError):
        fuse_ranked([RetrievedAd("a", 1, ("affinity",)), RetrievedAd("a", 1, ("affinity",))], [], 2)


def test_duplicate_catalog_rejected():
    with pytest.raises(ValueError):
        BehavioralIndex([ObservedAd("a", "c", "b", 1), ObservedAd("a", "c", "b", 2)])


def test_end_to_end_fallback_and_exclusion():
    results = retrieve_fused(index(), ShopperHistory((), (), ()), 2, pool_size=3, excluded={"a"})
    assert [x.adgroup_id for x in results] == ["b", "c"]
    assert all(x.sources == ("popularity",) for x in results)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        retrieve_fused(index(), history(), 3, pool_size=2)
    with pytest.raises(ValueError):
        fuse_ranked([], [], 1, rrf_k=0)
