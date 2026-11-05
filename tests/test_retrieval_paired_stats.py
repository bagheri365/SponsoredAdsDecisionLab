import pytest

from sponsored_ads.retrieval_paired_stats import paired_recall_differences


def test_discordance_and_exact_difference():
    rows = [
        {"fused": True, "popularity": False, "affinity": True},
        {"fused": False, "popularity": True, "affinity": False},
        {"fused": True, "popularity": True, "affinity": False},
        {"fused": False, "popularity": False, "affinity": False},
    ]
    result = paired_recall_differences(rows, repetitions=200, seed=8)
    pop = result["fused_minus_popularity"]
    assert pop["difference"] == 0
    assert (pop["first_only_hits"], pop["second_only_hits"], pop["both_hit"], pop["neither_hit"]) == (1, 1, 1, 1)
    assert result["fused_minus_affinity"]["difference"] == 0.25
    assert result == paired_recall_differences(rows, repetitions=200, seed=8)


def test_identical_outcomes_have_zero_interval():
    rows = [{"fused": True, "popularity": True, "affinity": True}] * 5
    assert paired_recall_differences(rows)["fused_minus_popularity"]["ci95_percentile"] == [0, 0]


def test_empty_and_invalid_inputs():
    assert paired_recall_differences([])["fused_minus_popularity"] is None
    with pytest.raises(ValueError, match="repetitions"):
        paired_recall_differences([], repetitions=1)
    with pytest.raises(ValueError, match="missing paired"):
        paired_recall_differences([{"fused": True}])
