"""Focused tests for M05F-A retraining guardrails."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sponsored_ads import m05fa_retrain as subject


def test_manifest_requires_test_exclusion():
    with pytest.raises(ValueError, match='exclude published test'):
        subject._validate_split_manifest({'published_test_used': True})


def test_manifest_requires_exact_partition_names():
    with pytest.raises(ValueError, match='three named'):
        subject._validate_split_manifest({'published_test_used': False, 'counts': {'early_stopping': {'rows': 2, 'clicks': 1}}})


@pytest.mark.parametrize('rows,clicks', [(0, 0), (3, 4), (3, -1)])
def test_manifest_rejects_invalid_counts(rows, clicks):
    counts = {name: {'rows': 10, 'clicks': 1} for name in ('early_stopping', 'calibration', 'evaluation')}
    counts['calibration'] = {'rows': rows, 'clicks': clicks}
    with pytest.raises(ValueError, match='Invalid partition counts'):
        subject._validate_split_manifest({'published_test_used': False, 'counts': counts})


def test_retrain_uses_only_train_and_early_stopping(tmp_path, monkeypatch):
    source, splits, output = (tmp_path / name for name in ('m03a', 'splits', 'output'))
    source.mkdir()
    splits.mkdir()
    (source / 'sample_manifest.json').write_text(json.dumps({'source_split': 'published train only'}))
    (splits / 'split_manifest.json').write_text(json.dumps({
        'published_test_used': False,
        'counts': {name: {'rows': 10, 'clicks': 1} for name in ('early_stopping', 'calibration', 'evaluation')},
    }))
    (source / 'train_features.csv').write_text('train')
    for name in ('early_stopping', 'calibration', 'evaluation'):
        (splits / f'{name}_features.csv').write_text(name)
    seen = []

    class FakeBooster:
        def save_model(self, path):
            Path(path).write_text('model')

    def fake_train(train, early, **kwargs):
        seen.extend((train, early))
        assert kwargs['return_contract'] is True
        return SimpleNamespace(booster_=FakeBooster()), {'best_iteration': 3}, {'features': ['price']}

    monkeypatch.setattr(subject, 'train_lightgbm', fake_train)
    monkeypatch.setattr(subject, 'save_contract', lambda contract, path: (Path(path).write_text('contract'), subject.sha256(path))[1])
    monkeypatch.setattr(subject, 'load_lightgbm_bundle', lambda model, contract: (SimpleNamespace(num_feature=lambda: 1), {'features': ['price']}))
    report = subject.retrain(source, splits, output)
    assert seen == [source / 'train_features.csv', splits / 'early_stopping_features.csv']
    assert report['calibration_partition_used_for_fitting'] is False
    assert report['evaluation_partition_used_for_fitting'] is False
    assert (output / 'retrain_report.json').exists()


def test_invalid_training_parameters(tmp_path):
    with pytest.raises(ValueError, match='Invalid seed'):
        subject.retrain(tmp_path, tmp_path, tmp_path, seed=-1)
    with pytest.raises(ValueError, match='Invalid seed'):
        subject.retrain(tmp_path, tmp_path, tmp_path, n_estimators=0)
