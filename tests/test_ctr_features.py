"""Tests for deterministic CTR development dataset construction."""
import csv
from pathlib import Path

import pytest

from sponsored_ads.ctr_features import FEATURE_COLUMNS, build_sample, feature_row
from sponsored_ads.taobao import COLUMNS


def row(**overrides):
    values = {name: '1' for name in COLUMNS}
    values.update({'clk': '0', 'price': '0.5', 'cate_id': '2', 'brand': '3',
                   'btag': '4', 'cate_his': '2^9^2', 'brand_his': '3^5^6',
                   'btag_his': '4^4^7'})
    values.update(overrides)
    return values


def source(tmp_path: Path, n=100):
    path = tmp_path / 'train.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for i in range(n):
            writer.writerow(row(clk=str(i % 2)))
    return path


def test_feature_counts_and_last_element():
    features = feature_row(row())
    assert features['history_length'] == '3'
    assert features['category_history_count'] == '2'
    assert features['brand_history_count'] == '1'
    assert features['behavior_history_count'] == '2'
    assert features['last_category_matches'] == '1'
    assert features['last_brand_matches'] == '0'
    assert tuple(features) == FEATURE_COLUMNS


def test_empty_history():
    features = feature_row(row(cate_his='', brand_his='', btag_his=''))
    assert features['history_length'] == '0'
    assert features['last_category_matches'] == '0'


@pytest.mark.parametrize('price', ['nan', 'inf', '-1'])
def test_invalid_price(price):
    with pytest.raises(ValueError, match='price'):
        feature_row(row(price=price))


def test_determinism_and_disjoint_assignment(tmp_path):
    path = source(tmp_path)
    first = build_sample(path, tmp_path / 'a', sample_rate=1, seed=7)
    second = build_sample(path, tmp_path / 'b', sample_rate=1, seed=7)
    assert first['counts'] == second['counts']
    assert sum(s['rows'] for s in first['counts'].values()) == 100
    for name in ('train_features.csv', 'validation_features.csv'):
        assert (tmp_path / 'a' / name).read_bytes() == (tmp_path / 'b' / name).read_bytes()


def test_smoke_limit(tmp_path):
    result = build_sample(source(tmp_path), tmp_path / 'out', sample_rate=1, limit=10)
    assert result['source_rows_scanned'] == 10
    assert sum(s['rows'] for s in result['counts'].values()) == 10


def test_reject_test_split(tmp_path):
    with pytest.raises(ValueError, match='train.csv'):
        build_sample(tmp_path / 'test.csv', tmp_path / 'out')


@pytest.mark.parametrize('rate, fraction', [(0, .2), (1.1, .2), (.5, 0), (.5, 1)])
def test_invalid_sampling_parameters(tmp_path, rate, fraction):
    with pytest.raises(ValueError):
        build_sample(tmp_path / 'train.csv', tmp_path / 'out', sample_rate=rate, validation_fraction=fraction)
