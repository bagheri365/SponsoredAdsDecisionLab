import csv
import json

import pytest

from sponsored_ads.taobao import (COLUMNS, DatasetValidationError, iter_rows,
                                  scan_file, write_manifest)


def make_csv(path, *, rows=2, bad_header=False, mismatch=False):
    header = list(COLUMNS)
    if bad_header:
        header[-1] = 'incorrect'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for i in range(rows):
            record = {col: '1' for col in COLUMNS}
            record['clk'] = str(i % 2)
            record['btag_his'] = '1^4'
            record['cate_his'] = '20^21'
            record['brand_his'] = '7' if mismatch else '7^8'
            if bad_header:
                record['incorrect'] = record.pop('btag')
            writer.writerow(record)


def test_stream_and_limit(tmp_path):
    path = tmp_path / 'train.csv'
    make_csv(path)
    assert len(list(iter_rows(path, limit=1))) == 1
    assert len(list(iter_rows(path, limit=0))) == 0
    assert len(list(iter_rows(path))) == 2


def test_scan(tmp_path):
    path = tmp_path / 'train.csv'
    make_csv(path)
    stats = scan_file(path)
    assert (stats.rows, stats.clicks, stats.max_history_length) == (2, 1, 2)
    assert len(stats.sha256) == 64


def test_header_rejected(tmp_path):
    path = tmp_path / 'train.csv'
    make_csv(path, bad_header=True)
    with pytest.raises(DatasetValidationError, match='unexpected header'):
        list(iter_rows(path))


def test_mismatched_history_rejected(tmp_path):
    path = tmp_path / 'train.csv'
    make_csv(path, mismatch=True)
    with pytest.raises(DatasetValidationError, match='history lengths differ'):
        list(iter_rows(path))


def test_invalid_label_rejected(tmp_path):
    path = tmp_path / 'train.csv'
    make_csv(path)
    text = path.read_text().replace('0,1^4', '2,1^4', 1)
    path.write_text(text)
    with pytest.raises(DatasetValidationError, match='clk'):
        list(iter_rows(path))


def test_manifest(tmp_path):
    for name in ('train.csv', 'test.csv'):
        make_csv(tmp_path / name)
    output = tmp_path / 'out' / 'manifest.json'
    manifest = write_manifest(tmp_path, output, limit=1)
    assert json.loads(output.read_text()) == manifest
    assert manifest['history_point_in_time_verified'] is False
    assert manifest['files']['train.csv']['rows'] == 1


def test_negative_limit_rejected(tmp_path):
    path = tmp_path / 'train.csv'
    make_csv(path)
    with pytest.raises(ValueError, match='limit'):
        list(iter_rows(path, limit=-1))
