import csv
import json

import numpy as np
import pytest

from sponsored_ads import m05fb_experiment as subject
from sponsored_ads.ctr_features import FEATURE_COLUMNS


def _fixture(tmp_path, monkeypatch):
    splits, models = tmp_path / 'splits', tmp_path / 'models'
    splits.mkdir()
    models.mkdir()
    paths = {}
    for name in ('calibration', 'evaluation'):
        path = splits / f'{name}_features.csv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
            writer.writeheader()
            for i in range(40):
                row = {column: '1' for column in FEATURE_COLUMNS}
                row['clk'] = str(int(i % 4 == 0))
                row['price'] = str(i + 1)
                writer.writerow(row)
        paths[name] = path
    manifest = {'schema_version': 1, 'published_test_used': False,
                'counts': {name: {'rows': 40, 'clicks': 10} for name in paths},
                'files': {name: {'name': path.name, 'sha256': subject.sha256(path)} for name, path in paths.items()}}
    split_manifest = splits / 'split_manifest.json'
    split_manifest.write_text(json.dumps(manifest))
    model = models / 'lightgbm_intent.txt'
    contract = models / 'lightgbm_intent.contract.json'
    model.write_text('model')
    contract.write_text('contract')
    retrain = {'published_test_used': False, 'calibration_partition_used_for_fitting': False,
               'evaluation_partition_used_for_fitting': False,
               'input_sha256': {'split_manifest': subject.sha256(split_manifest)},
               'output_sha256': {'model': subject.sha256(model), 'contract': subject.sha256(contract)}}
    (models / 'retrain_report.json').write_text(json.dumps(retrain))
    monkeypatch.setattr(subject, 'load_lightgbm_bundle', lambda *args: (object(), {}))
    monkeypatch.setattr(subject, 'predict_lightgbm', lambda model, contract, rows:
                        np.asarray([min(0.95, 0.08 + int(row['price']) / 500) for row in rows]))
    return splits, models, paths


def test_real_partition_report(tmp_path, monkeypatch):
    splits, models, _ = _fixture(tmp_path, monkeypatch)
    report = subject.run(splits, models, tmp_path / 'report.json', batch_size=7)
    assert report['calibration_rows'] == report['evaluation_rows'] == 40
    assert report['calibration_clicks'] == report['evaluation_clicks'] == 10
    assert set(report['metrics']) == {'raw', 'sigmoid', 'isotonic'}
    assert report['published_test_used'] is False
    assert report['metrics']['raw']['rows'] == 40
    assert report['selection'].startswith('No method selected')


def test_rejects_modified_calibration_file(tmp_path, monkeypatch):
    splits, models, paths = _fixture(tmp_path, monkeypatch)
    paths['calibration'].write_text(paths['calibration'].read_text() + '\n')
    with pytest.raises(ValueError, match='calibration input hash'):
        subject.run(splits, models, tmp_path / 'report.json')


def test_rejects_modified_model(tmp_path, monkeypatch):
    splits, models, _ = _fixture(tmp_path, monkeypatch)
    (models / 'lightgbm_intent.txt').write_text('changed')
    with pytest.raises(ValueError, match='model hash'):
        subject.run(splits, models, tmp_path / 'report.json')


def test_rejects_retraining_holdout_violation(tmp_path, monkeypatch):
    splits, models, _ = _fixture(tmp_path, monkeypatch)
    path = models / 'retrain_report.json'
    report = json.loads(path.read_text())
    report['evaluation_partition_used_for_fitting'] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='holdouts'):
        subject.run(splits, models, tmp_path / 'report.json')


def test_rejects_split_manifest_mismatch(tmp_path, monkeypatch):
    splits, models, _ = _fixture(tmp_path, monkeypatch)
    path = splits / 'split_manifest.json'
    report = json.loads(path.read_text())
    report['published_test_used'] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='split provenance'):
        subject.run(splits, models, tmp_path / 'report.json')


def test_rejects_invalid_probabilities(tmp_path, monkeypatch):
    splits, models, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(subject, 'predict_lightgbm', lambda *args: [1.2] * len(args[2]))
    with pytest.raises(ValueError, match='finite probability'):
        subject.run(splits, models, tmp_path / 'report.json')


def test_rejects_changed_label_counts(tmp_path, monkeypatch):
    splits, models, paths = _fixture(tmp_path, monkeypatch)
    path = paths['evaluation']
    content = path.read_text().replace('0,', '1,', 1)
    path.write_text(content)
    manifest_path = splits / 'split_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['evaluation']['sha256'] = subject.sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    retrain_path = models / 'retrain_report.json'
    retrain = json.loads(retrain_path.read_text())
    retrain['input_sha256']['split_manifest'] = subject.sha256(manifest_path)
    retrain_path.write_text(json.dumps(retrain))
    with pytest.raises(ValueError, match='counts differ'):
        subject.run(splits, models, tmp_path / 'report.json')


def test_refuses_overwrite(tmp_path, monkeypatch):
    splits, models, _ = _fixture(tmp_path, monkeypatch)
    path = tmp_path / 'report.json'
    path.write_text('existing')
    with pytest.raises(FileExistsError, match='overwrite'):
        subject.run(splits, models, path)


@pytest.mark.parametrize('bins,batch', [(1, 4), (10, 0)])
def test_invalid_parameters(tmp_path, monkeypatch, bins, batch):
    splits, models, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        subject.run(splits, models, tmp_path / 'report.json', bins=bins, batch_size=batch)
