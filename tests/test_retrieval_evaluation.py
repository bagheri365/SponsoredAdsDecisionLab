import pytest

from sponsored_ads.retrieval_evaluation import (
    ClickCase, RetrievalObservation, summarize_observed_clicks,
)


def case(ad, covered, popularity=(), affinity=(), fused=(), sources=None):
    return RetrievalObservation(
        ClickCase(ad, covered),
        {"popularity": popularity, "affinity": affinity, "fused": fused},
        sources,
    )


def test_micro_average_and_catalog_conditional_denominator():
    rows = [
        case("a", True, ("a",), (), ("a",), {"a": ("popularity",)}),
        case("b", True, (), ("b",), ("b",), {"b": ("affinity",)}),
        case("missing", False, ("a",), ("b",), ("a",)),
    ]
    summary = summarize_observed_clicks(rows, k=1)
    assert summary.clicked_impressions == 3
    assert summary.catalog_clicks == 2
    assert summary.catalog_coverage == pytest.approx(2 / 3)
    assert summary.methods["popularity"].observed_click_recall == pytest.approx(1 / 3)
    assert summary.methods["popularity"].catalog_conditional_recall == 0.5
    assert summary.methods["fused"].catalog_conditional_recall == 1
    assert summary.fused_hit_sources["affinity_only"] == 1
    assert summary.fused_hit_sources["popularity_only"] == 1


def test_top_k_cutoff_and_order():
    row = case("a", True, ("x", "a"), ("a",), ("x", "a"))
    assert summarize_observed_clicks([row], k=1).methods["popularity"].hits == 0
    assert summarize_observed_clicks([row], k=2).methods["popularity"].hits == 1


def test_repeated_clicked_impressions_are_weighted_separately():
    row = case("a", True, ("a",), (), ("a",))
    summary = summarize_observed_clicks([row, row], k=1)
    assert summary.clicked_impressions == 2
    assert summary.methods["popularity"].hits == 2


def test_empty_set_uses_null_metrics():
    summary = summarize_observed_clicks([], k=10)
    assert summary.catalog_coverage is None
    assert summary.methods["fused"].observed_click_recall is None
    assert summary.methods["fused"].catalog_conditional_recall is None


def test_zero_catalog_coverage_uses_null_conditional_recall():
    summary = summarize_observed_clicks([case("absent", False)], k=1)
    assert summary.methods["fused"].catalog_conditional_recall is None
    assert summary.methods["fused"].observed_click_recall == 0


def test_rejects_hit_outside_catalog():
    with pytest.raises(ValueError, match="absent"):
        summarize_observed_clicks([case("a", False, ("a",))], k=1)


def test_rejects_duplicate_rankings():
    with pytest.raises(ValueError, match="duplicate"):
        summarize_observed_clicks([case("a", True, ("a", "a"))], k=1)


def test_rejects_missing_methods_and_invalid_k():
    with pytest.raises(ValueError, match="exactly"):
        summarize_observed_clicks([RetrievalObservation(ClickCase("a", True), {"fused": ("a",)})], k=1)
    with pytest.raises(ValueError, match="positive"):
        summarize_observed_clicks([], k=0)
    with pytest.raises(ValueError, match="distinct"):
        summarize_observed_clicks([], k=1, methods=("fused", "fused"))


def test_fused_source_attribution_both_and_unknown():
    rows = [case("a", True, fused=("a",), sources={"a": ("affinity", "popularity")}),
            case("b", True, fused=("b",))]
    summary = summarize_observed_clicks(rows, k=1)
    assert summary.fused_hit_sources["both"] == 1
    assert summary.fused_hit_sources["other"] == 1


def test_invalid_empty_clicked_id():
    with pytest.raises(ValueError, match="nonempty"):
        ClickCase("", True)
