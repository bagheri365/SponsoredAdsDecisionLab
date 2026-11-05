import csv
import json

import numpy as np
import pytest

from sponsored_ads.ctr_features import FEATURE_COLUMNS
from sponsored_ads.ctr_baselines import (
    STATIC, FULL, _category_code, _hash_features, _matrix, _metrics, _read,
    train_logistic, train_lightgbm, run,
)


def row(i):
    result = {name: '1' for name in FEATURE_COLUMNS}
    result['clk'] = str(i % 2)
    result['userid'] = str(i)
    result['price'] = str(i % 5)
    result['history_length'] = str(i % 3)
    return result


def make_csv(path, n=40):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for i in range(n):
            writer.writerow(row(i))


def test_feature_sets():
    assert 'history_length' not in STATIC
    assert 'history_length' in FULL
    assert 'userid' in STATIC


def test_hashing_stable_and_numeric_scaled():
    assert _category_code('123') == _category_code('123')
    assert 0 <= _category_code('123') < 2**20
    values = _hash_features(row(5), STATIC)
    assert values['userid=5'] == 1
    assert values['price'] == pytest.approx(0)


def test_metrics_perfect():
    m = _metrics(np.array([0, 1]), np.array([0.01, 0.99]))
    assert m['roc_auc'] == 1
    assert m['pr_auc'] == 1
    assert m['log_loss'] < 0.02


def test_metrics_require_both_classes():
    with pytest.raises(ValueError):
        _metrics(np.array([0, 0]), np.array([0.1, 0.2]))


def test_schema_validation(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('clk,userid\n0,123\n')
    with pytest.raises(ValueError, match='schema'):
        list(_read(path))


def test_logistic_smoke(tmp_path):
    train, val = tmp_path / 'train.csv', tmp_path / 'val.csv'
    make_csv(train)
    make_csv(val, 20)
    _, result = train_logistic(train, val, batch_size=8)
    assert result['train_rows'] == 40
    assert result['metrics']['rows'] == 20


def test_lightgbm_smoke(tmp_path):
    pytest.importorskip('lightgbm')
    train, val = tmp_path / 'train.csv', tmp_path / 'val.csv'
    make_csv(train, 80)
    make_csv(val, 20)
    _, result = train_lightgbm(train, val, features=STATIC, n_estimators=5)
    assert result['metrics']['rows'] == 20


def test_run_rejects_non_train_provenance(tmp_path):
    inp = tmp_path / 'input'
    inp.mkdir()
    (inp / 'sample_manifest.json').write_text(json.dumps({'source_split': 'published test'}))
    with pytest.raises(ValueError, match='published-train-only'):
        run(inp, tmp_path / 'out')


def test_train_fitted_category_encoding(tmp_path):
    train, val = tmp_path / 'train.csv', tmp_path / 'val.csv'
    make_csv(train, 4)
    make_csv(val, 2)
    # The validation file includes a category not seen during fitting.
    with val.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        unseen = row(99)
        unseen['userid'] = 'new-user'
        writer.writerow(unseen)
    X_train, _, maps = _matrix(train, ('userid', 'price'))
    X_val, _, _ = _matrix(val, ('userid', 'price'), category_maps=maps)
    assert X_train[:, 0].tolist() == [0, 1, 2, 3]
    assert X_val[:2, 0].tolist() == [0, 1]
    assert X_val[2, 0] == -1
    assert 'new-user' not in maps['userid']
