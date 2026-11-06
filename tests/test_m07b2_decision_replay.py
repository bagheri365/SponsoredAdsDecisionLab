"""M07B-2 deterministic scored replay, integrity, and budget invariants."""
from copy import deepcopy

import pytest

from sponsored_ads.budget_ledger import BudgetLedger
from sponsored_ads.decision_replay import capture_replay_record, replay_decision, _digest
from sponsored_ads.decision_service import decide
from sponsored_ads.model_auction_service import ModelProvenance
from sponsored_ads.models import AdCandidate, AdRequest


PROVENANCE = ModelProvenance('a' * 64, 'b' * 64, 'raw', None, 1)


def _record(candidates=None, budgets=None, *, policy='score_second_price', reserve=0.0):
    if candidates is None:
        candidates = (AdCandidate('ad-1', 'adv-1', .2, 2.),
                      AdCandidate('ad-2', 'adv-2', .1, 1.5))
    if budgets is None:
        budgets = {'adv-1': 500, 'adv-2': 500}
    ledger = BudgetLedger(budgets)
    request = AdRequest('request-1', 'shopper-1')
    prior = ledger.events
    decision = decide(request, candidates, ledger, policy=policy,
                      event_id='event-1', reservation_id='reservation-1',
                      reserve_cpc=reserve)
    return capture_replay_record(
        request=request, scored_candidates=candidates, provenance=PROVENANCE,
        initial_budgets_cents=budgets, prior_ledger_events=prior,
        policy=policy, event_id='event-1', reservation_id='reservation-1',
        reserve_cpc=reserve, decision=decision, final_ledger=ledger)


def test_filled_replay_and_roundtrip():
    import json
    record = json.loads(json.dumps(_record(), sort_keys=True))
    decision, ledger = replay_decision(record, expected_provenance=PROVENANCE)
    assert decision.status == 'filled'
    assert decision.outcome.ad_id == 'ad-1'
    assert ledger.snapshot('adv-1').reserved_cents == 200
    assert ledger.snapshot('adv-1').spent_cents == 0
    assert len(ledger.events) == 1
    assert replay_decision(record)[0] == decision


@pytest.mark.parametrize('candidates,budgets,reason', [
    ((), {'adv-1': 500}, 'no_candidates'),
    ((AdCandidate('ad-1', 'adv-1', .2, 1., False),), {'adv-1': 500},
     'no_eligible_or_above_reserve'),
    ((AdCandidate('ad-1', 'adv-1', .2, 2.),), {'adv-1': 0},
     'insufficient_budget'),
])
def test_no_fill_replay_has_no_new_event(candidates, budgets, reason):
    record = _record(candidates, budgets)
    result, ledger = replay_decision(record)
    assert result.status == 'no_fill'
    assert result.no_fill_reason == reason
    assert ledger.events == ()


def test_prior_ledger_events_are_reconstructed():
    budgets = {'adv-1': 500, 'adv-2': 500}
    ledger = BudgetLedger(budgets)
    ledger.reserve(event_id='old-event', reservation_id='old-res',
                   advertiser_id='adv-1', max_charge_cents=400)
    prior = ledger.events
    request = AdRequest('req-2', 'shopper')
    candidates = (AdCandidate('ad-1', 'adv-1', .3, 2.),
                  AdCandidate('ad-2', 'adv-2', .2, 1.))
    result = decide(request, candidates, ledger, policy='first_price',
                    event_id='new-event', reservation_id='new-res')
    record = capture_replay_record(
        request=request, scored_candidates=candidates, provenance=PROVENANCE,
        initial_budgets_cents=budgets, prior_ledger_events=prior,
        policy='first_price', event_id='new-event', reservation_id='new-res',
        reserve_cpc=0.0, decision=result, final_ledger=ledger)
    replayed, reconstructed = replay_decision(record)
    assert replayed == result
    assert reconstructed.events == ledger.events
    assert reconstructed.snapshot('adv-1') == ledger.snapshot('adv-1')


def test_tampering_is_detected():
    record = _record()
    record['scored_candidates'][0]['predicted_ctr'] = .99
    with pytest.raises(ValueError, match='hash mismatch'):
        replay_decision(record)


def test_rehashed_but_inconsistent_expected_result_is_detected():
    record = _record()
    record['expected_decision']['status'] = 'no_fill'
    record['sha256'] = _digest({k: v for k, v in record.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='decision mismatch'):
        replay_decision(record)


def test_provenance_mismatch_is_detected():
    record = _record()
    other = ModelProvenance('c' * 64, 'b' * 64, 'raw', None, 1)
    with pytest.raises(ValueError, match='provenance mismatch'):
        replay_decision(record, expected_provenance=other)


def test_malformed_and_unsupported_versions():
    record = _record()
    with pytest.raises(ValueError, match='exactly'):
        replay_decision({**record, 'unknown': 1})
    broken = deepcopy(record)
    broken['schema_version'] = 2
    with pytest.raises(ValueError, match='schema version'):
        replay_decision(broken)
    broken = deepcopy(record)
    broken['scored_candidates'][0]['ad_id'] = ''
    broken['sha256'] = _digest({k: v for k, v in broken.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='nonempty'):
        replay_decision(broken)


def test_capture_rejects_inconsistent_original_decision():
    record = _record()
    budgets = {'adv-1': 500, 'adv-2': 500}
    ledger = BudgetLedger(budgets)
    request = AdRequest('request-1', 'shopper-1')
    candidates = (AdCandidate('ad-1', 'adv-1', .2, 2.),
                  AdCandidate('ad-2', 'adv-2', .1, 1.5))
    wrong = decide(request, candidates, ledger, policy='first_price',
                   event_id='event-1', reservation_id='reservation-1')
    with pytest.raises(ValueError, match='decision mismatch'):
        capture_replay_record(
            request=request, scored_candidates=candidates, provenance=PROVENANCE,
            initial_budgets_cents=budgets, prior_ledger_events=(),
            policy='score_second_price', event_id='event-1',
            reservation_id='reservation-1', reserve_cpc=0.0,
            decision=wrong, final_ledger=ledger)
