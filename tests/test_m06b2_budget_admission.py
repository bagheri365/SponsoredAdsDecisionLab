import pytest

from sponsored_ads.budget_admission import admit_budget_aware
from sponsored_ads.budget_ledger import BudgetLedger, finalize_auction_winner
from sponsored_ads.models import AdCandidate


def ads():
    return [AdCandidate('a', 'adv-a', .2, 2.0), AdCandidate('b', 'adv-b', .1, 3.0)]


def run(ledger, candidates=None, **kwargs):
    return admit_budget_aware(ledger, ads() if candidates is None else candidates,
                              policy='score_second_price', event_id='e1',
                              reservation_id='r1', **kwargs)


def test_funded_winner_and_price():
    ledger = BudgetLedger({'adv-a': 200, 'adv-b': 300})
    result = run(ledger)
    assert result.outcome.ad_id == 'a'
    assert result.outcome.charged_cpc_if_clicked == pytest.approx(1.5)
    assert result.reservation.max_charge_cents == 200
    assert ledger.snapshot('adv-a').available_cents == 0


def test_retry_excludes_unaffordable_and_reprices():
    ledger = BudgetLedger({'adv-a': 199, 'adv-b': 300})
    result = run(ledger)
    assert result.excluded_ad_ids == ('a',)
    assert result.outcome.ad_id == 'b'
    assert result.outcome.runner_up_score is None
    assert result.outcome.charged_cpc_if_clicked == 0
    assert ledger.snapshot('adv-b').reserved_cents == 300


def test_first_price_on_fallback():
    ledger = BudgetLedger({'adv-a': 0, 'adv-b': 300})
    result = admit_budget_aware(ledger, ads(), policy='first_price', event_id='e', reservation_id='r')
    assert result.outcome.ad_id == 'b'
    assert result.outcome.charged_cpc_if_clicked == 3


def test_no_fill_no_event():
    ledger = BudgetLedger({'adv-a': 0, 'adv-b': 0})
    result = run(ledger)
    assert result.outcome is None and result.reservation is None
    assert result.excluded_ad_ids == ('a', 'b')
    assert ledger.events == ()


def test_empty_no_fill():
    ledger = BudgetLedger({})
    assert run(ledger, []).outcome is None
    assert ledger.events == ()


def test_below_reserve_not_reported_unaffordable():
    ledger = BudgetLedger({'adv-a': 0, 'adv-b': 300})
    result = run(ledger, reserve_cpc=2.5)
    assert result.excluded_ad_ids == ()
    assert result.outcome.ad_id == 'b'
    assert result.outcome.charged_cpc_if_clicked == 2.5


def test_ineligible_not_budget_checked():
    ledger = BudgetLedger({'adv-b': 300})
    candidates = [AdCandidate('a', 'missing', .9, 10, False), ads()[1]]
    assert run(ledger, candidates).outcome.ad_id == 'b'


def test_duplicate_id_rejected_without_mutation():
    ledger = BudgetLedger({'adv-a': 1000})
    with pytest.raises(ValueError, match='unique'):
        run(ledger, [ads()[0], ads()[0]])
    assert ledger.events == ()


@pytest.mark.parametrize('policy,reserve', [('bad', 0), ('first_price', -1), ('first_price', float('nan'))])
def test_invalid_policy_or_reserve_fails_without_mutation(policy, reserve):
    ledger = BudgetLedger({'adv-a': 300, 'adv-b': 300})
    with pytest.raises(ValueError):
        admit_budget_aware(ledger, ads(), policy=policy, reserve_cpc=reserve,
                           event_id='e', reservation_id='r')
    assert ledger.events == ()


def test_missing_advertiser_fails_without_mutation():
    ledger = BudgetLedger({'adv-a': 200})
    with pytest.raises(ValueError, match='unknown advertiser'):
        run(ledger)
    assert ledger.events == ()


def test_repeated_auction_uses_available_not_initial_budget():
    ledger = BudgetLedger({'adv-a': 200, 'adv-b': 600})
    first = run(ledger)
    assert first.outcome.ad_id == 'a'
    second = admit_budget_aware(ledger, ads(), policy='score_second_price',
                                event_id='e2', reservation_id='r2')
    assert second.outcome.ad_id == 'b'
    assert second.excluded_ad_ids == ('a',)
    assert ledger.snapshot('adv-a').reserved_cents == 200
    assert ledger.snapshot('adv-b').reserved_cents == 300


def test_release_allows_future_admission():
    ledger = BudgetLedger({'adv-a': 200, 'adv-b': 300})
    first = run(ledger)
    finalize_auction_winner(ledger, first.outcome, event_id='release', reservation_id='r1', clicked=False)
    second = admit_budget_aware(ledger, ads(), policy='first_price', event_id='e2', reservation_id='r2')
    assert second.outcome.ad_id == 'a'


def test_settlement_and_replay():
    ledger = BudgetLedger({'adv-a': 200, 'adv-b': 300})
    result = run(ledger)
    finalize_auction_winner(ledger, result.outcome, event_id='settle', reservation_id='r1', clicked=True)
    replay = BudgetLedger.replay({'adv-a': 200, 'adv-b': 300}, ledger.events)
    assert replay.snapshot('adv-a') == ledger.snapshot('adv-a')
    assert ledger.snapshot('adv-a').spent_cents == 151


def test_event_and_reservation_reuse_rejected():
    ledger = BudgetLedger({'adv-a': 400, 'adv-b': 600})
    run(ledger)
    with pytest.raises(ValueError, match='event_id already used'):
        run(ledger)
    with pytest.raises(ValueError, match='reservation_id already used'):
        admit_budget_aware(ledger, ads(), policy='first_price', event_id='e2', reservation_id='r1')
    assert len(ledger.events) == 1


def test_zero_bid_zero_budget_admitted():
    ledger = BudgetLedger({'adv': 0})
    result = run(ledger, [AdCandidate('zero', 'adv', 0, 0)])
    assert result.outcome.ad_id == 'zero'
    assert result.reservation.max_charge_cents == 0


def test_order_independence():
    first = run(BudgetLedger({'adv-a': 199, 'adv-b': 300}))
    second = run(BudgetLedger({'adv-a': 199, 'adv-b': 300}), list(reversed(ads())))
    assert first == second


def test_ceil_rounding_blocks_near_budget():
    ledger = BudgetLedger({'adv': 150})
    result = run(ledger, [AdCandidate('a', 'adv', .5, 1.5000000000000002)])
    assert result.outcome is None
    assert result.excluded_ad_ids == ('a',)


def test_invalid_event_ids_do_not_mutate():
    ledger = BudgetLedger({'adv-a': 300, 'adv-b': 300})
    with pytest.raises(ValueError, match='event_id'):
        admit_budget_aware(ledger, ads(), policy='first_price', event_id='', reservation_id='r')
    assert ledger.events == ()
