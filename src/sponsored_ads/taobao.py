"""Streaming reader and validation for RecZoo TaobaoAd_x1 CSV exports.

The published split has no event timestamps; no chronological claims are made.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

COLUMNS = (
    'clk', 'btag_his', 'cate_his', 'brand_his', 'userid', 'cms_segid',
    'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level',
    'shopping_level', 'occupation', 'new_user_class_level', 'adgroup_id',
    'cate_id', 'campaign_id', 'customer', 'brand', 'price', 'pid', 'btag',
)
HISTORY_COLUMNS = ('btag_his', 'cate_his', 'brand_his')


@dataclass(frozen=True)
class DatasetStats:
    file: str
    bytes: int
    sha256: str
    rows: int
    clicks: int
    empty_histories: int
    max_history_length: int
    columns: list[str]


class DatasetValidationError(ValueError):
    """Invalid CSV schema or row content."""


def _tokens(value: str) -> list[str]:
    if value == '':
        return []
    parts = value.split('^')
    if any(not part or not part.isdigit() for part in parts):
        raise DatasetValidationError('history must be ^-separated nonnegative integer IDs')
    return parts


def validate_row(row: dict[str, str], line_number: int) -> int:
    """Validate one row and return the number of historical events."""
    if None in row or any(value is None for value in row.values()):
        raise DatasetValidationError(f'line {line_number}: malformed field count')
    if row['clk'] not in ('0', '1'):
        raise DatasetValidationError(f'line {line_number}: clk must be 0 or 1')
    for col in COLUMNS:
        if row[col] == '' and col not in HISTORY_COLUMNS:
            raise DatasetValidationError(f'line {line_number}: missing {col}')
    lengths = [len(_tokens(row[col])) for col in HISTORY_COLUMNS]
    if len(set(lengths)) != 1:
        raise DatasetValidationError(f'line {line_number}: history lengths differ: {lengths}')
    return lengths[0]


def iter_rows(path: str | Path, *, limit: int | None = None) -> Iterator[dict[str, str]]:
    """Stream validated rows, retaining ID values as strings.

    `limit` is intended for local smoke tests, not dataset-wide quality claims.
    """
    if limit is not None and limit < 0:
        raise ValueError('limit must be nonnegative')
    with Path(path).open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(COLUMNS):
            raise DatasetValidationError(f'unexpected header: {reader.fieldnames!r}')
        for line_number, row in enumerate(reader, start=2):
            if limit is not None and line_number - 2 >= limit:
                break
            validate_row(row, line_number)
            yield row


def scan_file(path: str | Path, *, limit: int | None = None) -> DatasetStats:
    """Stream rows and compute SHA-256 of the entire file (even with a row limit)."""
    path = Path(path)
    clicks = rows = empty_histories = max_history_length = 0
    for line_number, row in enumerate(iter_rows(path, limit=limit), start=2):
        history_length = validate_row(row, line_number)
        rows += 1
        clicks += int(row['clk'])
        empty_histories += history_length == 0
        max_history_length = max(max_history_length, history_length)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return DatasetStats(
        file=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest(),
        rows=rows, clicks=clicks, empty_histories=empty_histories,
        max_history_length=max_history_length, columns=list(COLUMNS),
    )


def write_manifest(dataset_dir: str | Path, output: str | Path, *, limit: int | None = None) -> dict:
    """Write a JSON manifest; never include raw shopper IDs or records."""
    dataset_dir = Path(dataset_dir)
    results = {name: asdict(scan_file(dataset_dir / name, limit=limit))
               for name in ('train.csv', 'test.csv')}
    manifest = {
        'dataset': 'reczoo/TaobaoAd_x1',
        'split_provenance': 'published train/test; temporal ordering unverified',
        'history_point_in_time_verified': False,
        'scanned_row_limit_per_file': limit,
        'files': results,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest
