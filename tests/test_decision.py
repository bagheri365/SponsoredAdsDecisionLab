import pytest

from sponsored_ads import AdCandidate, AdRequest, choose_ad


def test_selects_max_bid_weighted_ctr():
    request = AdRequest("r1", "u1")
    ads = [
        AdCandidate("a", "adv1", 0.1, 1.0),
        AdCandidate("b", "adv2", 0.08, 2.0),
    ]
    result = choose_ad(request, ads)
    assert result is not None
    assert result.ad_id == "b"
    assert result.score == pytest.approx(0.16)


def test_excludes_ineligible_ads():
    request = AdRequest("r1", "u1")
    ads = [
        AdCandidate("a", "adv1", 0.9, 10, eligible=False),
        AdCandidate("b", "adv2", 0.1, 1),
    ]
    assert choose_ad(request, ads).ad_id == "b"


def test_empty_or_ineligible_returns_none():
    request = AdRequest("r1", "u1")
    assert choose_ad(request, []) is None
    assert choose_ad(request, [AdCandidate("a", "adv", 0.5, 1, False)]) is None


def test_ties_resolve_by_ad_id():
    request = AdRequest("r1", "u1")
    ads = [
        AdCandidate("z", "adv1", 0.1, 1),
        AdCandidate("a", "adv2", 0.1, 1),
    ]
    assert choose_ad(request, ads).ad_id == "a"


@pytest.mark.parametrize("ctr,bid", [(-0.1, 1), (1.1, 1), (float("nan"), 1), (0.1, -1), (0.1, float("inf"))])
def test_invalid_scores_rejected(ctr, bid):
    with pytest.raises(ValueError):
        AdCandidate("a", "adv", ctr, bid)


def test_duplicate_ad_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        choose_ad(
            AdRequest("r1", "u1"),
            [AdCandidate("a", "adv1", 0.1, 1), AdCandidate("a", "adv2", 0.2, 1)],
        )
