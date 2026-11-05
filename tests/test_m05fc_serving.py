import json

import numpy as np
import pytest

from sponsored_ads import m05fc_serving as subject

HASH = 'a' * 64


def example(method):
    y = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0] * 6)
    p = np.linspace(.01, .6, len(y))
    return subject.fit_artifact(y, p, method=method, model_sha256=HASH,
                                contract_sha256=HASH, calibration_sha256=HASH), p


@pytest.mark.parametrize('method', ['raw', 'sigmoid', 'isotonic'])
def test_fit_and_apply(method):
    artifact, p = example(method)
    actual = subject.apply_calibration(p, artifact)
    assert actual.shape == p.shape
    assert np.isfinite(actual).all()
    assert ((actual >= 0) & (actual <= 1)).all()
    if method == 'raw':
        np.testing.assert_array_equal(actual, p)


def test_sigmoid_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sponsored_ads.ctr_calibration import _logit
    artifact, p = example('sigmoid')
    y = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0] * 6)
    model = LogisticRegression(random_state=2026, max_iter=1000).fit(_logit(p), y)
    np.testing.assert_allclose(subject.apply_calibration(p, artifact), model.predict_proba(_logit(p))[:, 1], atol=1e-14)


def test_isotonic_matches_sklearn():
    from sklearn.isotonic import IsotonicRegression
    artifact, p = example('isotonic')
    y = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0] * 6)
    model = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(p, y)
    np.testing.assert_allclose(subject.apply_calibration(np.array([0., .3, 1.]), artifact), model.predict([0., .3, 1.]))


def test_persistence_and_overwrite(tmp_path):
    artifact, p = example('sigmoid')
    path = tmp_path / 'artifact.json'
    digest = subject.save_artifact(artifact, path)
    assert digest == subject.sha256(path)
    assert json.loads(path.read_text()) == artifact
    with pytest.raises(FileExistsError):
        subject.save_artifact(artifact, path)


def test_policy_hash_checks(tmp_path, monkeypatch):
    artifact, _ = example('sigmoid')
    model, contract, spec = tmp_path / 'model', tmp_path / 'contract', tmp_path / 'calibrator.json'
    model.write_text('model')
    contract.write_text('contract')
    artifact['model_sha256'] = subject.sha256(model)
    artifact['contract_sha256'] = subject.sha256(contract)
    spec.write_text(json.dumps(artifact))
    monkeypatch.setattr(subject, 'load_lightgbm_bundle', lambda *_: ('model', 'contract'))
    assert subject.load_serving_bundle(model, contract, spec, method='sigmoid')[2] == artifact
    with pytest.raises(ValueError, match='policy'):
        subject.load_serving_bundle(model, contract, spec, method='isotonic')
    model.write_text('changed')
    with pytest.raises(ValueError, match='incompatible'):
        subject.load_serving_bundle(model, contract, spec, method='sigmoid')


def test_raw_default_and_explicit_calibrated_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, 'load_lightgbm_bundle', lambda *_: ('model', 'contract'))
    assert subject.load_serving_bundle('model', 'contract', None) == ('model', 'contract', None)
    with pytest.raises(ValueError, match='requires'):
        subject.load_serving_bundle('model', 'contract', None, method='sigmoid')


def test_bad_probabilities_and_corrupt_knots():
    artifact, _ = example('isotonic')
    with pytest.raises(ValueError):
        subject.apply_calibration([float('nan')], artifact)
    with pytest.raises(ValueError):
        subject.apply_calibration([-0.01], artifact)
    artifact['parameters']['x'] = [0.5, 0.5]
    artifact['parameters']['y'] = [0.2, 0.4]
    with pytest.raises(ValueError, match='ordered'):
        subject.apply_calibration([0.2], artifact)


def test_invalid_labels():
    with pytest.raises(ValueError, match='Binary labels'):
        subject.fit_artifact([0, 2, 1], [.1, .2, .3], method='raw', model_sha256=HASH,
                             contract_sha256=HASH, calibration_sha256=HASH)


def test_prediction_path(monkeypatch):
    artifact, _ = example('sigmoid')
    monkeypatch.setattr(subject, 'predict_lightgbm', lambda *_: np.array([.1, .2]))
    np.testing.assert_array_equal(subject.predict_calibrated('model', {}, [{}]), [.1, .2])
    np.testing.assert_allclose(subject.predict_calibrated('model', {}, [{}], artifact),
                               subject.apply_calibration([.1, .2], artifact))


def test_fit_runner_uses_only_calibration(tmp_path, monkeypatch):
    split, models = tmp_path / 'split', tmp_path / 'models'
    split.mkdir(); models.mkdir()
    calib = split / 'calibration_features.csv'; calib.write_text('calibration')
    evaluation = split / 'evaluation_features.csv'; evaluation.write_text('evaluation')
    model = models / 'lightgbm_intent.txt'; model.write_text('model')
    contract = models / 'lightgbm_intent.contract.json'; contract.write_text('contract')
    monkeypatch.setattr(subject, '_verify_inputs', lambda *_: (
        {'counts': {'calibration': {'rows': 4, 'clicks': 2}}}, {},
        {'calibration': calib, 'evaluation': evaluation}, model, contract, None, None))
    monkeypatch.setattr(subject, 'load_lightgbm_bundle', lambda *_: ('model', 'contract'))
    seen = []
    def score(path, *_args, **_kwargs):
        seen.append(path)
        return np.array([0, 1, 0, 1]), np.array([.1, .2, .3, .4])
    monkeypatch.setattr(subject, '_score_partition', score)
    result = subject.run_fit(split, models, tmp_path / 'out.json', method='sigmoid')
    assert seen == [calib]
    assert result['fit_rows'] == 4
    assert result['fit_clicks'] == 2
