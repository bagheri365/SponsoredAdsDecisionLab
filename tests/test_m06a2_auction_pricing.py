import math

import pytest

from sponsored_ads.auction_pricing import price_single_slot
from sponsored_ads.models import AdCandidate


def ad(name, ctr, bid, eligible=True):
    return AdCandidate(name, 'advertiser-' + name, ctr, bid, eligible)


def price(ads, policy='score_second_price', reserve=0):
    return price_single_slot(ads, policy=policy, reserve_cpc=reserve)


def test_first_price_charges_winning_bid():
    outcome = price([ad('a', .2, 2), ad('b', .1, 3)], 'first_price')
    assert outcome.ad_id == 'a'
    assert outcome.charged_cpc_if_clicked == 2
    assert outcome.runner_up_score == pytest.approx(.3)


def test_score_second_price_uses_runner_score_not_runner_bid():
    outcome = price([ad('a', .2, 2), ad('b', .1, 3)])
    assert outcome.ad_id == 'a'
    assert outcome.charged_cpc_if_clicked == pytest.approx(1.5)


def test_single_bidder_pays_reserve_in_second_price():
    assert price([ad('a', .2, 3)], reserve=.5).charged_cpc_if_clicked == .5


def test_single_bidder_first_price_pays_bid():
    assert price([ad('a', .2, 3)], 'first_price', reserve=.5).charged_cpc_if_clicked == 3


def test_below_reserve_cannot_win_or_set_price():
    outcome = price([ad('a', .9, .4), ad('b', .2, 2)], reserve=.5)
    assert outcome.ad_id == 'b'
    assert outcome.runner_up_score is None
    assert outcome.charged_cpc_if_clicked == .5


def test_all_below_reserve_returns_none():
    assert price([ad('a', .9, .4)], reserve=.5) is None


def test_empty_and_ineligible_return_none():
    assert price([]) is None
    assert price([ad('a', .9, 4, False)]) is None


def test_equal_scores_use_lexical_id_and_pay_bid():
    outcome = price([ad('z', .1, 2), ad('a', .2, 1)])
    assert outcome.ad_id == 'a'
    assert outcome.charged_cpc_if_clicked == pytest.approx(1)


def test_zero_ctr_all_zero_scores_pays_reserve():
    outcome = price([ad('a', 0, 2), ad('b', 0, 3)], reserve=.5)
    assert outcome.ad_id == 'a'
    assert outcome.charged_cpc_if_clicked == .5


def test_zero_bid_zero_reserve_allowed():
    outcome = price([ad('a', .5, 0)])
    assert outcome.charged_cpc_if_clicked == 0


def test_ineligible_high_score_does_not_set_price():
    outcome = price([ad('a', .5, 10, False), ad('b', .2, 2)])
    assert outcome.ad_id == 'b'
    assert outcome.charged_cpc_if_clicked == 0


@pytest.mark.parametrize('reserve', [-1, float('nan'), float('inf'), True, '1', None])
def test_invalid_reserve_rejected(reserve):
    with pytest.raises(ValueError, match='reserve_cpc'):
        price([], reserve=reserve)


@pytest.mark.parametrize('policy', ['second_price', '', None, 'FIRST_PRICE'])
def test_unknown_policy_rejected(policy):
    with pytest.raises(ValueError, match='policy'):
        price([], policy=policy)


def test_duplicate_ids_rejected_even_below_reserve():
    with pytest.raises(ValueError, match='unique'):
        price([ad('a', .2, 0), ad('a', .1, 0)], reserve=1)


def test_very_large_finite_bid_remains_finite():
    outcome = price([ad('a', 1, 1e308)], reserve=1)
    assert outcome.ad_id == 'a'
    assert outcome.charged_cpc_if_clicked == 1


def test_permutation_invariance():
    ads = [ad('c', .2, 3), ad('b', .1, 5), ad('a', .5, 2)]
    assert price(ads) == price(list(reversed(ads)))


def test_payment_bounds_for_multiple_cases():
    for ctr1 in (0, .01, .1, 1):
        for ctr2 in (0, .01, .1, 1):
            for reserve in (0, .5, 1):
                outcome = price([ad('a', ctr1, 3), ad('b', ctr2, 2)], reserve=reserve)
                assert outcome is not None
                assert math.isfinite(outcome.charged_cpc_if_clicked)
                assert reserve <= outcome.charged_cpc_if_clicked <= outcome.bid_cpc
