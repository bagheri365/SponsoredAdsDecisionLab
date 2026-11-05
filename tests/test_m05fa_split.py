import csv
import json

import pytest

from sponsored_ads.ctr_features import FEATURE_COLUMNS
from sponsored_ads.m05fa_split import split_validation


def fixture(tmp_path, *, n=120):
    root = tmp_path / 'm03a'
    root.mkdir()
    for name in ('train', 'validation'):
        with (root / f'{name}_features.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FEATURE_COLUMNS)
            writer.writeheader()
            for i in range(n):
                writer.writerow({**{key: '0' for key in FEATURE_COLUMNS}, 'clk': str(i % 2)})
    (root / 'sample_manifest.json').write_text(json.dumps({
        'source_split': 'published train only', 'feature_columns': list(FEATURE_COLUMNS),
        'counts': {'validation': {'rows': n, 'clicks': n // 2}}
    }))
    return root


def test_disjoint_counts_and_provenance(tmp_path):
    root = fixture(tmp_path)
    report = split_validation(root, tmp_path / 'out')
    assert sum(x['rows'] for x in report['counts'].values()) == 120
    assert sum(x['clicks'] for x in report['counts'].values()) == 60
    assert all(x['rows'] > x['clicks'] > 0 for x in report['counts'].values())
    assert report['published_test_used'] is False
    assert report['temporal_split'] is False
    assert report['user_disjoint'] is False
    assert (tmp_path / 'out' / 'split_manifest.json').exists()


def test_reproducible_bytes(tmp_path):
    root = fixture(tmp_path)
    a = split_validation(root, tmp_path / 'a')
    b = split_validation(root, tmp_path / 'b')
    assert a == b


def test_seed_changes_assignment(tmp_path):
    root = fixture(tmp_path)
    a = split_validation(root, tmp_path / 'a', seed=2026)
    b = split_validation(root, tmp_path / 'b', seed=2027)
    assert a['files'] != b['files']


def test_reject_existing_outputs(tmp_path):
    root = fixture(tmp_path)
    split_validation(root, tmp_path / 'out')
    with pytest.raises(FileExistsError):
        split_validation(root, tmp_path / 'out')


@pytest.mark.parametrize('early,cal', [(0, .3), (.5, .5), (-.1, .3), (.8, .3)])
def test_bad_fractions(tmp_path, early, cal):
    root = fixture(tmp_path)
    with pytest.raises(ValueError):
        split_validation(root, tmp_path / 'out', early_stopping_fraction=early,
                         calibration_fraction=cal)


def test_reject_bad_provenance(tmp_path):
    root = fixture(tmp_path)
    path = root / 'sample_manifest.json'
    obj = json.loads(path.read_text())
    obj['source_split'] = 'published test'
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='provenance'):
        split_validation(root, tmp_path / 'out')


def test_reject_manifest_count_mismatch(tmp_path):
    root = fixture(tmp_path)
    path = root / 'sample_manifest.json'
    obj = json.loads(path.read_text())
    obj['counts']['validation']['rows'] += 1
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='counts'):
        split_validation(root, tmp_path / 'out')
    assert not (tmp_path / 'out' / 'split_manifest.json').exists()


def test_reject_same_directory(tmp_path):
    root = fixture(tmp_path)
    with pytest.raises(ValueError, match='differ'):
        split_validation(root, root)
