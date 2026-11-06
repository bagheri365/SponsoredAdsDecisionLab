"""M07A-3 CSV sampling and deterministic report tests."""
import csv
import json
from pathlib import Path

import pytest

from sponsored_ads.m07a3_smoke import sample_features, run_smoke
from sponsored_ads.model_auction_service import ModelAuctionBundle, ModelProvenance


def test_sample_features(tmp_path):
    path = tmp_path / 'rows.csv'
    path.write_text('a,b,label\nx,1,0\ny,2,1\nz,3,0\n')
    assert sample_features(path, ['b', 'a'], 2) == [{'b': '1', 'a': 'x'}, {'b': '2', 'a': 'y'}]


@pytest.mark.parametrize('text,features,count', [
    ('a,b\nx,1\n', ['a'], 2),
    ('a,b\nx,1\n', ['missing'], 2),
    ('a,a\nx,1\n', ['a'], 2),
    ('a,b\nx\n', ['a', 'b'], 2),
    ('a,b\nx,1\n', ['a'], 1),
])
def test_invalid_samples(tmp_path, text, features, count):
    path = tmp_path / 'rows.csv'
    path.write_text(text)
    with pytest.raises(ValueError):
        sample_features(path, features, count)


def test_report_determinism(tmp_path, monkeypatch):
    import sponsored_ads.m07a3_smoke as smoke
    feature_path = tmp_path / 'features.csv'
    feature_path.write_text('f\n1\n2\n3\n')
    model = tmp_path / 'model.txt'
    contract = tmp_path / 'contract.json'
    model.write_text('mock')
    contract.write_text('mock')
    provenance = ModelProvenance('a' * 64, 'b' * 64, 'raw', None, 1)
    bundle = ModelAuctionBundle(model=object(), contract={'features': ['f']},
                             calibration_artifact=None, provenance=provenance)
    monkeypatch.setattr(smoke, 'load_model_auction_bundle', lambda *a, **k: bundle)
    from sponsored_ads import model_auction_service
    mock_predict = lambda *a: [0.2, 0.1, 0.05]
    monkeypatch.setattr(model_auction_service, 'predict_calibrated', mock_predict)
    monkeypatch.setattr(smoke, 'predict_calibrated', mock_predict)
    # Use actual auction/ledger, only mock inference to isolate the report contract.
    outputs = []
    for i in (1, 2):
        out = tmp_path / f'report{i}.json'
        report = run_smoke(model_path=model, contract_path=contract,
                           features_path=feature_path, output_path=out)
        outputs.append(out.read_bytes())
        assert report['decision']['status'] == 'filled'
        assert report['ledger_event_count'] == 1
        assert report['predicted_ctr'] == [0.2, 0.1, 0.05]
        assert report['scored_candidate_count'] == 3
        assert sum(x['reserved_cents'] for x in report['ledger_snapshots']) == 200
        assert sum(x['spent_cents'] for x in report['ledger_snapshots']) == 0
        assert json.loads(out.read_text())['report_sha256'] == report['report_sha256']
    assert outputs[0] == outputs[1]
    with pytest.raises(FileExistsError):
        run_smoke(model_path=model, contract_path=contract,
                  features_path=feature_path, output_path=tmp_path / 'report1.json')
