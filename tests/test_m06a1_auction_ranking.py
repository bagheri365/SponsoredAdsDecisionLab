"""M06A-1 deterministic ranking and M01 compatibility tests."""

from dataclasses import FrozenInstanceError
from itertools import permutations

import pytest

from sponsored_ads.auction_ranking import rank_eligible
from sponsored_ads.decision import choose_ad
from sponsored_ads.models import AdCandidate, AdRequest


def ad(ad_id, ctr, bid, eligible=True):
    return AdCandidate(ad_id, 'advertiser-' + ad_id, ctr, bid, eligible)


def test_score_order_and_full_ranking():
    result = rank_eligible([ad('a', .1, 1), ad('b', .08, 2), ad('c', .2, 1)])
    assert [r.ad_id for r in result] == ['c', 'b', 'a']
    assert [r.score for r in result] == pytest.approx([.2, .16, .1])


def test_ties_use_lexical_ad_id_independent_of_input_order():
    candidates = [ad('z', .1, 1), ad('a', .2, .5), ad('m', .05, 2)]
    for order in permutations(candidates):
        assert [r.ad_id for r in rank_eligible(order)] == ['a', 'm', 'z']


def test_excludes_ineligible():
    assert [r.ad_id for r in rank_eligible([ad('x', 1, 100, False), ad('a', .1, 1)])] == ['a']


def test_empty_and_all_ineligible():
    assert rank_eligible([]) == ()
    assert rank_eligible([ad('a', .1, 1, False)]) == ()


def test_zero_bid_and_zero_ctr_are_valid():
    assert [r.score for r in rank_eligible([ad('b', 0, 2), ad('a', .2, 0)])] == [0, 0]


def test_duplicate_rejected_even_when_ineligible():
    with pytest.raises(ValueError, match='unique'):
        rank_eligible([ad('a', .1, 1, False), ad('a', .2, 2)])


def test_generator_is_supported():
    assert [r.ad_id for r in rank_eligible(ad(str(i), .1, i) for i in range(3))] == ['2', '1', '0']


def test_non_candidate_rejected():
    with pytest.raises(TypeError, match='AdCandidate'):
        rank_eligible([object()])


def test_large_finite_bid_remains_finite():
    assert rank_eligible([ad('a', 1.0, 1e308)])[0].score == 1e308


def test_ranking_is_immutable():
    result = rank_eligible([ad('a', .1, 1)])
    with pytest.raises(FrozenInstanceError):
        result[0].score = 3


def test_choose_ad_agrees_with_first_rank():
    candidates = [ad('z', .1, 1), ad('a', .2, 2), ad('x', .8, 1, False)]
    first = rank_eligible(candidates)[0]
    decision = choose_ad(AdRequest('r', 'u'), candidates)
    assert (decision.ad_id, decision.advertiser_id, decision.score) == (
        first.ad_id, first.advertiser_id, first.score)


def test_choose_ad_preserves_empty_result():
    assert choose_ad(AdRequest('r', 'u'), []) is None
