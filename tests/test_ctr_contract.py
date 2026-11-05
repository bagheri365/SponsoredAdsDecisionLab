import csv
import json

import numpy as np
import pytest

from sponsored_ads.ctr_baselines import _matrix
from sponsored_ads.ctr_contract import (
    load_contract, make_contract, predict_lightgbm, save_contract, transform_rows,
)
from sponsored_ads.ctr_features import FEATURE_COLUMNS


def _row(**updates):
    row = {name: '0' for name in FEATURE_COLUMNS}
    row.update(userid='alice', adgroup_id='ad-1', price='12.5', history_length='2')
    row.update(updates)
    return row


def _csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_matrix_and_serving_parity(tmp_path):
    rows = [_row(), _row(userid='bob', adgroup_id='ad-2', price='0')]
    path = tmp_path / 'features.csv'
    _csv(path, rows)
    features = ('userid', 'adgroup_id', 'price', 'history_length')
    X, _, maps = _matrix(path, features)
    spec = make_contract(features, maps)
    np.testing.assert_array_equal(X, transform_rows(rows, spec))
    assert spec['category_maps']['userid'] == {'alice': 0, 'bob': 1}
    unseen = _row(userid='unseen', adgroup_id='ad-1')
    assert transform_rows([unseen], spec)[0].tolist() == [-1.0, 0.0, 12.5, 2.0]


def test_contract_round_trip(tmp_path):
    spec = make_contract(('userid', 'price'), {'userid': {'alice': 0}})
    path = tmp_path / 'contract.json'
    digest = save_contract(spec, path)
    assert len(digest) == 64
    assert load_contract(path) == spec
    assert json.loads(path.read_text())['version'] == 1


def test_invalid_and_missing_features():
    spec = make_contract(('userid', 'price'), {'userid': {'alice': 0}})
    with pytest.raises(ValueError, match='Missing required'):
        transform_rows([{'userid': 'alice'}], spec)
    for value in ('nan', '-1', 'inf', 'oops'):
        with pytest.raises(ValueError):
            transform_rows([{'userid': 'alice', 'price': value}], spec)
    with pytest.raises(ValueError, match='empty'):
        transform_rows([], spec)


def test_contract_rejects_corruption():
    spec = make_contract(('userid',), {'userid': {'alice': 0}})
    for change in ({'version': 2}, {'unknown_category': 0}, {'features': ['price']}):
        corrupted = dict(spec, **change)
        from sponsored_ads.ctr_contract import validate_contract
        with pytest.raises((ValueError, KeyError)):
            validate_contract(corrupted)
    with pytest.raises(ValueError, match='contiguous'):
        make_contract(('userid',), {'userid': {'alice': 2}})


def test_predict_uses_contract_encoding():
    class Model:
        def predict(self, matrix):
            assert matrix.tolist() == [[-1.0, 3.0]]
            return np.array([0.2])
    spec = make_contract(('userid', 'price'), {'userid': {'alice': 0}})
    assert predict_lightgbm(Model(), spec, [{'userid': 'new', 'price': '3'}]).tolist() == [0.2]
