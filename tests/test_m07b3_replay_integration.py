"""M07B-3: optional replay sidecar preserves existing report and inference count."""
import json

import pytest

from sponsored_ads.budget_ledger import BudgetLedger
from sponsored_ads.decision_replay import replay_decision
from sponsored_ads.m07a3_smoke import run_smoke
from sponsored_ads.model_auction_service import ModelAuctionBundle, ModelProvenance


@pytest.mark.parametrize('method', ['raw', 'sigmoid'])
def test_sidecar_preserves_report_and_replays(tmp_path, monkeypatch, method):
    import sponsored_ads.m07a3_smoke as smoke
    import sponsored_ads.model_auction_service as service

    features = tmp_path / 'features.csv'
    features.write_text('f\n1\n2\n3\n')
    model = tmp_path / 'model.txt'
    contract = tmp_path / 'contract.json'
    model.write_text('mock')
    contract.write_text('mock')
    provenance = ModelProvenance('a' * 64, 'b' * 64, method,
                                 'c' * 64 if method == 'sigmoid' else None, 1)
    bundle = ModelAuctionBundle(model=object(), contract={'features': ['f']},
                                calibration_artifact=None, provenance=provenance)
    monkeypatch.setattr(smoke, 'load_model_auction_bundle', lambda *a, **k: bundle)
    calls = []

    def predict(*args):
        calls.append(1)
        return [0.2, 0.1, 0.05]

    monkeypatch.setattr(service, 'predict_calibrated', predict)
    plain = tmp_path / 'plain.json'
    sidecar_report = tmp_path / 'sidecar_report.json'
    replay_file = tmp_path / 'replay.json'
    run_smoke(model_path=model, contract_path=contract, features_path=features,
              output_path=plain, method=method)
    assert len(calls) == 1
    run_smoke(model_path=model, contract_path=contract, features_path=features,
              output_path=sidecar_report, method=method,
              replay_output_path=replay_file)
    assert len(calls) == 2  # Exactly one inference call per smoke invocation.
    assert plain.read_bytes() == sidecar_report.read_bytes()
    record = json.loads(replay_file.read_text())
    decision, ledger = replay_decision(record, expected_provenance=provenance)
    report = json.loads(plain.read_text())
    assert decision.outcome.ad_id == report['decision']['outcome']['ad_id']
    assert record['scored_candidates'][0]['predicted_ctr'] == report['predicted_ctr'][0]
    assert len(ledger.events) == report['ledger_event_count'] == 1
    assert record['expected_ledger_snapshots'] == report['ledger_snapshots']
    assert record['prior_ledger_events'] == []


def test_replay_output_preflight_prevents_inference(tmp_path, monkeypatch):
    import sponsored_ads.m07a3_smoke as smoke
    features = tmp_path / 'features.csv'
    features.write_text('f\n1\n2\n3\n')
    model = tmp_path / 'model.txt'
    contract = tmp_path / 'contract.json'
    model.write_text('mock')
    contract.write_text('mock')

    def forbidden(*args, **kwargs):
        raise AssertionError('Model must not load if output paths conflict')

    monkeypatch.setattr(smoke, 'load_model_auction_bundle', forbidden)
    replay = tmp_path / 'existing.json'
    replay.write_text('keep')
    with pytest.raises(FileExistsError):
        run_smoke(model_path=model, contract_path=contract, features_path=features,
                  output_path=tmp_path / 'report.json', replay_output_path=replay)
    assert replay.read_text() == 'keep'
    with pytest.raises(FileExistsError):
        run_smoke(model_path=model, contract_path=contract, features_path=features,
                  output_path=tmp_path / 'report.json',
                  replay_output_path=tmp_path / 'report.json')


def test_cli_forwards_replay_output(monkeypatch, tmp_path, capsys):
    import sponsored_ads.m07a3_smoke as smoke

    observed = []

    def fake_run_smoke(**kwargs):
        observed.append(kwargs)
        return {'report_sha256': 'test-hash', 'decision': {'status': 'filled'}}

    monkeypatch.setattr(smoke, 'run_smoke', fake_run_smoke)
    replay_path = tmp_path / 'replay.json'
    smoke.main(['--output', str(tmp_path / 'report.json'),
                '--replay-output', str(replay_path)])
    assert len(observed) == 1
    assert observed[0]['replay_output_path'] == str(replay_path)
    assert json.loads(capsys.readouterr().out)['status'] == 'filled'
