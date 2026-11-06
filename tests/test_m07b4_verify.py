"""M07B-4: offline artifact-verification regression tests."""
import copy
import hashlib
import json

import pytest

from sponsored_ads.m07b4_verify import BASELINES, _canonical, main, verify_artifacts


def _pair(tmp_path, monkeypatch, method='raw'):
    """Build a genuine synthetic auction pair; patch only baseline hashes."""
    import sponsored_ads.m07a3_smoke as smoke
    import sponsored_ads.model_auction_service as service
    from sponsored_ads.model_auction_service import ModelAuctionBundle, ModelProvenance

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
    monkeypatch.setattr(service, 'predict_calibrated', lambda *a: [0.2, 0.1, 0.05])
    report_path = tmp_path / 'report.json'
    replay_path = tmp_path / 'replay.json'
    smoke.run_smoke(model_path=model, contract_path=contract, features_path=features,
                    output_path=report_path, replay_output_path=replay_path, method=method)
    report = json.loads(report_path.read_text())
    monkeypatch.setitem(BASELINES[method], 'report_sha256', report['report_sha256'])
    import sponsored_ads.m07b4_verify as verifier
    monkeypatch.setattr(verifier, 'MODEL_SHA', provenance.model_sha256)
    monkeypatch.setattr(verifier, 'CONTRACT_SHA', provenance.contract_sha256)
    if method == 'sigmoid':
        monkeypatch.setattr(verifier, 'SIGMOID_SHA', provenance.calibration_artifact_sha256)
    return report_path, replay_path


def _rewrite(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


@pytest.mark.parametrize('method', ['raw', 'sigmoid'])
def test_valid_pair_and_cli(tmp_path, monkeypatch, capsys, method):
    report, replay = _pair(tmp_path, monkeypatch, method)
    result = verify_artifacts(report_path=report, replay_path=replay, scenario=method,
                              expected_report_file_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                              expected_replay_file_sha256=hashlib.sha256(replay.read_bytes()).hexdigest())
    assert result['status'] == 'PASS'
    assert result['winner'] == 'synthetic-ad-1'
    main(['--scenario', method, '--report', str(report), '--replay', str(replay)])
    assert json.loads(capsys.readouterr().out)['status'] == 'PASS'


def test_corrupt_report_checksum(tmp_path, monkeypatch):
    report, replay = _pair(tmp_path, monkeypatch)
    data = json.loads(report.read_text())
    data['predicted_ctr'][0] = 0.9
    _rewrite(report, data)
    with pytest.raises(ValueError, match='report checksum mismatch'):
        verify_artifacts(report_path=report, replay_path=replay, scenario='raw')


def test_corrupt_replay_checksum(tmp_path, monkeypatch):
    report, replay = _pair(tmp_path, monkeypatch)
    data = json.loads(replay.read_text())
    data['scored_candidates'][0]['predicted_ctr'] = 0.9
    _rewrite(replay, data)
    with pytest.raises(ValueError, match='replay record hash mismatch'):
        verify_artifacts(report_path=report, replay_path=replay, scenario='raw')


def test_rehashed_mismatched_pair(tmp_path, monkeypatch):
    report, replay = _pair(tmp_path, monkeypatch)
    data = json.loads(report.read_text())
    data['predicted_ctr'][0] = 0.9
    data['report_sha256'] = hashlib.sha256(_canonical({
        k: v for k, v in data.items() if k != 'report_sha256'
    })).hexdigest()
    monkeypatch.setitem(BASELINES['raw'], 'report_sha256', data['report_sha256'])
    _rewrite(report, data)
    with pytest.raises(ValueError, match='paired CTR predictions mismatch'):
        verify_artifacts(report_path=report, replay_path=replay, scenario='raw')


def test_wrong_complete_file_hash(tmp_path, monkeypatch):
    report, replay = _pair(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='report complete-file checksum mismatch'):
        verify_artifacts(report_path=report, replay_path=replay, scenario='raw',
                         expected_report_file_sha256='0' * 64)
    with pytest.raises(ValueError, match='replay complete-file checksum mismatch'):
        verify_artifacts(report_path=report, replay_path=replay, scenario='raw',
                         expected_replay_file_sha256='0' * 64)


def test_missing_file_and_cli_failure(tmp_path, monkeypatch, capsys):
    report, replay = _pair(tmp_path, monkeypatch)
    replay.unlink()
    with pytest.raises(SystemExit) as exc:
        main(['--scenario', 'raw', '--report', str(report), '--replay', str(replay)])
    assert exc.value.code == 1
    assert 'FAIL:' in capsys.readouterr().err


def test_provenance_mismatch(tmp_path, monkeypatch):
    report, replay = _pair(tmp_path, monkeypatch)
    data = json.loads(report.read_text())
    data['provenance']['model_sha256'] = 'f' * 64
    data['report_sha256'] = hashlib.sha256(_canonical({
        k: v for k, v in data.items() if k != 'report_sha256'
    })).hexdigest()
    monkeypatch.setitem(BASELINES['raw'], 'report_sha256', data['report_sha256'])
    _rewrite(report, data)
    with pytest.raises(ValueError, match='report provenance differs'):
        verify_artifacts(report_path=report, replay_path=replay, scenario='raw')
