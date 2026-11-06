"""M07A-2 contract tests: inference, auction delegation and failure atomicity."""
import numpy as np
import pytest

from sponsored_ads.budget_ledger import BudgetLedger
from sponsored_ads.model_auction_service import (
    FeatureCandidate, ModelAuctionBundle, ModelProvenance,
    decide_with_model,
)
from sponsored_ads.models import AdRequest


PROVENANCE = ModelProvenance('a' * 64, 'b' * 64, 'raw', None, 1)


def bundle():
    return ModelAuctionBundle(object(), {'version': 1}, None, PROVENANCE)


def candidate(ad_id='a', advertiser='adv-a', bid=2.0, eligible=True):
    return FeatureCandidate(ad_id, advertiser, bid, {'f': 1}, eligible)


def run(items, ledger, monkeypatch, predictions, **kwargs):
    from sponsored_ads import model_auction_service as module
    monkeypatch.setattr(module, 'predict_calibrated', lambda *args: np.asarray(predictions))
    return decide_with_model(
        AdRequest('req-1', 'shopper-1'), items, ledger, bundle(),
        policy=kwargs.pop('policy', 'score_second_price'),
        event_id='e1', reservation_id='r1', **kwargs,
    )


def test_filled_reserves_without_spend(monkeypatch):
    ledger = BudgetLedger({'adv-a': 300})
    result = run([candidate()], ledger, monkeypatch, [.2])
    assert result.decision.status == 'filled'
    assert result.decision.outcome.ad_id == 'a'
    assert result.scored_candidate_count == 1
    assert result.provenance == PROVENANCE
    assert ledger.snapshot('adv-a').reserved_cents == 200
    assert ledger.snapshot('adv-a').spent_cents == 0


def test_budget_exclusion_reprices(monkeypatch):
    ledger = BudgetLedger({'adv-a': 199, 'adv-b': 300})
    result = run([candidate(), candidate('b', 'adv-b', 3.0)], ledger, monkeypatch, [.2, .1])
    assert result.decision.outcome.ad_id == 'b'
    assert result.decision.outcome.charged_cpc_if_clicked == 0.0
    assert result.decision.trace.excluded_ad_ids == ('a',)


def test_empty_candidates_no_fill_without_inference(monkeypatch):
    from sponsored_ads import model_auction_service as module
    monkeypatch.setattr(module, 'predict_calibrated', lambda *args: pytest.fail('unexpected inference'))
    ledger = BudgetLedger({'adv-a': 300})
    result = decide_with_model(AdRequest('r', 's'), [], ledger, bundle(),
                               policy='first_price', event_id='e', reservation_id='r')
    assert result.decision.no_fill_reason == 'no_candidates'
    assert result.scored_candidate_count == 0
    assert not ledger.events


@pytest.mark.parametrize('predictions', [[float('nan')], [-.1], [1.1], [.2, .3]])
def test_invalid_predictions_do_not_mutate(monkeypatch, predictions):
    ledger = BudgetLedger({'adv-a': 300})
    with pytest.raises(ValueError):
        run([candidate()], ledger, monkeypatch, predictions)
    assert not ledger.events


def test_missing_features_fail_before_reservation(monkeypatch):
    from sponsored_ads import model_auction_service as module
    def bad(*args):
        raise ValueError('Missing required feature: f')
    monkeypatch.setattr(module, 'predict_calibrated', bad)
    ledger = BudgetLedger({'adv-a': 300})
    with pytest.raises(ValueError, match='Missing required feature'):
        decide_with_model(AdRequest('r', 's'), [candidate()], ledger, bundle(),
                          policy='first_price', event_id='e', reservation_id='r')
    assert not ledger.events


def test_duplicate_ids_rejected_before_inference(monkeypatch):
    from sponsored_ads import model_auction_service as module
    monkeypatch.setattr(module, 'predict_calibrated', lambda *args: pytest.fail('unexpected inference'))
    ledger = BudgetLedger({'adv-a': 300})
    with pytest.raises(ValueError, match='Duplicate'):
        decide_with_model(AdRequest('r', 's'), [candidate(), candidate()], ledger, bundle(),
                          policy='first_price', event_id='e', reservation_id='r')
    assert not ledger.events


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.0])
def test_invalid_bid_rejected(bad):
    with pytest.raises(ValueError):
        candidate(bid=bad)


def test_features_must_be_mapping():
    with pytest.raises(TypeError):
        FeatureCandidate('a', 'adv-a', 1.0, [1, 2])


def test_no_budget_no_fill(monkeypatch):
    ledger = BudgetLedger({'adv-a': 0})
    result = run([candidate()], ledger, monkeypatch, [.2])
    assert result.decision.no_fill_reason == 'insufficient_budget'
    assert not ledger.events


def test_ineligible_candidate_does_not_reserve(monkeypatch):
    ledger = BudgetLedger({'adv-a': 300})
    result = run([candidate(eligible=False)], ledger, monkeypatch, [.2])
    assert result.decision.no_fill_reason == 'no_eligible_or_above_reserve'
    assert not ledger.events


def test_batch_order_preserved(monkeypatch):
    ledger = BudgetLedger({'adv-a': 300, 'adv-b': 300})
    result = run([candidate('b', 'adv-b', 1), candidate('a', 'adv-a', 2)], ledger,
                 monkeypatch, [.1, .3])
    assert result.decision.outcome.ad_id == 'a'
