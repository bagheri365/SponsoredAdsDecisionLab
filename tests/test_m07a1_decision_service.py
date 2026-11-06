from dataclasses import FrozenInstanceError

import pytest

from sponsored_ads.decision_service import decide
from sponsored_ads.budget_ledger import BudgetLedger
from sponsored_ads.models import AdCandidate, AdRequest


REQUEST = AdRequest('request-1', 'shopper-1')


def run(ledger, candidates, **kwargs):
    return decide(REQUEST, candidates, ledger, policy='score_second_price',
                  event_id='event-1', reservation_id='reservation-1', **kwargs)


def test_filled_reserves_but_does_not_spend():
    ledger = BudgetLedger({'adv-a': 300, 'adv-b': 300})
    response = run(ledger, [AdCandidate('a', 'adv-a', .2, 2), AdCandidate('b', 'adv-b', .1, 3)])
    assert response.request_id == 'request-1'
    assert response.status == 'filled' and response.no_fill_reason is None
    assert response.outcome.ad_id == 'a'
    assert response.trace.reservation_id == 'reservation-1'
    assert response.trace.reservation_event_id == 'event-1'
    assert ledger.snapshot('adv-a').reserved_cents == 200
    assert ledger.snapshot('adv-a').spent_cents == 0
    assert len(ledger.events) == 1


def test_budget_exclusion_reprices():
    ledger = BudgetLedger({'adv-a': 199, 'adv-b': 300})
    response = run(ledger, [AdCandidate('a', 'adv-a', .2, 2), AdCandidate('b', 'adv-b', .1, 3)])
    assert response.outcome.ad_id == 'b'
    assert response.outcome.charged_cpc_if_clicked == 0
    assert response.trace.excluded_ad_ids == ('a',)
    assert ledger.snapshot('adv-b').reserved_cents == 300


@pytest.mark.parametrize('candidates,budgets,reason', [
    ([], {}, 'no_candidates'),
    ([AdCandidate('a', 'adv-a', .2, 1, eligible=False)], {}, 'no_eligible_or_above_reserve'),
    ([AdCandidate('a', 'adv-a', .2, 1)], {'adv-a': 0}, 'insufficient_budget'),
])
def test_no_fill_reason_and_no_mutation(candidates, budgets, reason):
    ledger = BudgetLedger(budgets)
    response = run(ledger, candidates)
    assert response.status == 'no_fill'
    assert response.no_fill_reason == reason
    assert response.outcome is None
    assert response.trace.reservation_event_id is None
    assert response.trace.reservation_id is None
    assert ledger.events == ()


def test_below_reserve_is_distinct_from_budget_exhaustion():
    ledger = BudgetLedger({})
    response = run(ledger, [AdCandidate('a', 'adv-a', .2, 1)], reserve_cpc=2)
    assert response.no_fill_reason == 'no_eligible_or_above_reserve'
    assert ledger.events == ()


def test_zero_bid_with_zero_budget_can_fill():
    ledger = BudgetLedger({'adv-a': 0})
    response = run(ledger, [AdCandidate('a', 'adv-a', 0, 0)])
    assert response.status == 'filled'
    assert ledger.snapshot('adv-a').reserved_cents == 0
    assert len(ledger.events) == 1


def test_invalid_request_does_not_mutate():
    ledger = BudgetLedger({'adv-a': 100})
    with pytest.raises(TypeError):
        decide('not-request', [AdCandidate('a', 'adv-a', .2, 1)], ledger,
               policy='first_price', event_id='e', reservation_id='r')
    assert ledger.events == ()


def test_duplicate_ad_ids_fail_without_mutation():
    ledger = BudgetLedger({'adv-a': 100})
    with pytest.raises(ValueError):
        run(ledger, [AdCandidate('a', 'adv-a', .2, 1), AdCandidate('a', 'adv-a', .1, 1)])
    assert ledger.events == ()


def test_unknown_advertiser_fails_closed():
    ledger = BudgetLedger({})
    with pytest.raises(ValueError, match='unknown advertiser'):
        run(ledger, [AdCandidate('a', 'missing', .2, 1)])
    assert ledger.events == ()


def test_invalid_policy_fails_without_mutation():
    ledger = BudgetLedger({'adv-a': 100})
    with pytest.raises(ValueError):
        decide(REQUEST, [AdCandidate('a', 'adv-a', .2, 1)], ledger,
               policy='bad', event_id='e', reservation_id='r')
    assert ledger.events == ()


def test_reused_event_id_fails_closed():
    ledger = BudgetLedger({'adv-a': 300})
    candidate = [AdCandidate('a', 'adv-a', .2, 1)]
    run(ledger, candidate)
    with pytest.raises(ValueError, match='event_id already used'):
        run(ledger, candidate)
    assert len(ledger.events) == 1


def test_deterministic_across_candidate_order_on_fresh_ledgers():
    candidates = [AdCandidate('b', 'adv-b', .1, 2), AdCandidate('a', 'adv-a', .2, 1)]
    left = run(BudgetLedger({'adv-a': 200, 'adv-b': 200}), candidates)
    right = run(BudgetLedger({'adv-a': 200, 'adv-b': 200}), reversed(candidates))
    assert left == right
    assert left.outcome.ad_id == 'a'


def test_response_is_frozen():
    response = run(BudgetLedger({}), [])
    with pytest.raises(FrozenInstanceError):
        response.status = 'filled'
    with pytest.raises(FrozenInstanceError):
        response.trace.input_count = 3


def test_generator_consumed_once():
    ledger = BudgetLedger({'adv-a': 200})
    response = run(ledger, (x for x in [AdCandidate('a', 'adv-a', .2, 1)]))
    assert response.trace.input_count == 1
    assert response.status == 'filled'
