"""M07B-1: scored diagnostics and one inference call per request."""
from dataclasses import asdict

import pytest

from sponsored_ads.budget_ledger import BudgetLedger
from sponsored_ads.model_auction_service import (
    FeatureCandidate, ModelAuctionBundle, ModelProvenance, decide_with_model,
)
from sponsored_ads.models import AdRequest


def _bundle():
    return ModelAuctionBundle(
        model=object(), contract={'features': ['f']}, calibration_artifact=None,
        provenance=ModelProvenance('a' * 64, 'b' * 64, 'raw', None, 1),
    )


def _candidates():
    return [
        FeatureCandidate('ad-1', 'adv-1', 2.0, {'f': '1'}),
        FeatureCandidate('ad-2', 'adv-2', 1.5, {'f': '2'}),
        FeatureCandidate('ad-3', 'adv-3', 1.0, {'f': '3'}, eligible=False),
    ]


def test_single_inference_and_scored_input_order(monkeypatch):
    import sponsored_ads.model_auction_service as service
    calls = []

    def infer(model, contract, rows, artifact):
        calls.append([dict(row) for row in rows])
        return [0.2, 0.1, 0.9]

    monkeypatch.setattr(service, 'predict_calibrated', infer)
    ledger = BudgetLedger({'adv-1': 500, 'adv-2': 500, 'adv-3': 500})
    result = decide_with_model(
        AdRequest('req', 'shopper'), iter(_candidates()), ledger, _bundle(),
        policy='score_second_price', event_id='reserve', reservation_id='res',
    )
    assert calls == [[{'f': '1'}, {'f': '2'}, {'f': '3'}]]
    assert result.scored_candidate_count == len(result.scored_candidates) == 3
    assert [c.ad_id for c in result.scored_candidates] == ['ad-1', 'ad-2', 'ad-3']
    assert [c.predicted_ctr for c in result.scored_candidates] == [0.2, 0.1, 0.9]
    assert result.scored_candidates[2].eligible is False
    assert result.decision.status == 'filled'
    assert result.decision.outcome.ad_id == 'ad-1'
    assert result.decision.outcome.predicted_ctr == result.scored_candidates[0].predicted_ctr
    assert ledger.snapshot('adv-1').reserved_cents == 200
    assert ledger.snapshot('adv-1').spent_cents == 0
    assert len(ledger.events) == 1
    assert asdict(result.provenance)['calibration_method'] == 'raw'


def test_empty_request_does_not_infer_or_mutate(monkeypatch):
    import sponsored_ads.model_auction_service as service

    def forbidden(*args):
        raise AssertionError('Empty batch must not call inference')

    monkeypatch.setattr(service, 'predict_calibrated', forbidden)
    ledger = BudgetLedger({'adv-1': 500})
    result = decide_with_model(
        AdRequest('empty', 'shopper'), [], ledger, _bundle(),
        policy='first_price', event_id='empty-event', reservation_id='empty-res',
    )
    assert result.decision.status == 'no_fill'
    assert result.scored_candidate_count == 0
    assert result.scored_candidates == ()
    assert len(ledger.events) == 0


def test_prediction_failure_keeps_ledger_unchanged(monkeypatch):
    import sponsored_ads.model_auction_service as service

    def bad_predict(*args):
        raise ValueError('invalid feature row')

    monkeypatch.setattr(service, 'predict_calibrated', bad_predict)
    ledger = BudgetLedger({'adv-1': 500, 'adv-2': 500, 'adv-3': 500})
    with pytest.raises(ValueError, match='invalid feature row'):
        decide_with_model(
            AdRequest('bad', 'shopper'), _candidates(), ledger, _bundle(),
            policy='score_second_price', event_id='bad-event', reservation_id='bad-res',
        )
    assert len(ledger.events) == 0
    assert all(ledger.snapshot(key).available_cents == 500 for key in ('adv-1', 'adv-2', 'adv-3'))


def test_smoke_reuses_service_scores(monkeypatch, tmp_path):
    import sponsored_ads.m07a3_smoke as smoke
    import sponsored_ads.model_auction_service as service

    features = tmp_path / 'features.csv'
    features.write_text('f\n1\n2\n3\n')
    model = tmp_path / 'model.txt'
    contract = tmp_path / 'contract.json'
    model.write_text('mock')
    contract.write_text('mock')
    monkeypatch.setattr(smoke, 'load_model_auction_bundle', lambda *a, **k: _bundle())
    calls = []

    def infer(*args):
        calls.append(1)
        return [0.2, 0.1, 0.05]

    monkeypatch.setattr(service, 'predict_calibrated', infer)
    output = tmp_path / 'smoke.json'
    report = smoke.run_smoke(
        model_path=model, contract_path=contract, features_path=features,
        output_path=output,
    )
    assert calls == [1]
    assert report['predicted_ctr'] == [0.2, 0.1, 0.05]
    assert report['decision']['status'] == 'filled'
    assert report['decision']['outcome']['predicted_ctr'] == report['predicted_ctr'][0]
    assert report['ledger_event_count'] == 1
    assert output.exists()
