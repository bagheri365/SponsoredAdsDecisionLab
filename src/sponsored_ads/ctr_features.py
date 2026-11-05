"""Deterministic, streaming feature extraction for TaobaoAd_x1.

Only the published training CSV is used. Historical arrays are treated as
provided, without asserting point-in-time validity or chronological order.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from .taobao import iter_rows

STATIC_CATEGORICAL = (
    'userid', 'cms_segid', 'cms_group_id', 'final_gender_code',
    'age_level', 'pvalue_level', 'shopping_level', 'occupation',
    'new_user_class_level', 'adgroup_id', 'cate_id', 'campaign_id',
    'customer', 'brand', 'pid', 'btag',
)
NUMERIC = (
    'price', 'history_length', 'category_history_count',
    'brand_history_count', 'behavior_history_count',
    'last_category_matches', 'last_brand_matches',
)
FEATURE_COLUMNS = ('clk', *STATIC_CATEGORICAL, *NUMERIC)


def feature_row(row: dict[str, str]) -> dict[str, str]:
    """Construct features using only fields supplied on this impression row.

    Sequence order is assumed to be source order; `last_*` refers to its final
    element, not a verified most-recent timestamped interaction.
    """
    category = row['cate_his'].split('^') if row['cate_his'] else []
    brand = row['brand_his'].split('^') if row['brand_his'] else []
    behavior = row['btag_his'].split('^') if row['btag_his'] else []
    price = float(row['price'])
    if not math.isfinite(price) or price < 0:
        raise ValueError('price must be finite and nonnegative')
    return {
        'clk': row['clk'],
        **{name: row[name] for name in STATIC_CATEGORICAL},
        'price': row['price'],
        'history_length': str(len(category)),
        'category_history_count': str(category.count(row['cate_id'])),
        'brand_history_count': str(brand.count(row['brand'])),
        'behavior_history_count': str(behavior.count(row['btag'])),
        'last_category_matches': str(int(bool(category) and category[-1] == row['cate_id'])),
        'last_brand_matches': str(int(bool(brand) and brand[-1] == row['brand'])),
    }


def _uniform(seed: int, row_number: int, salt: str) -> float:
    """Stable pseudo-random value independent of Python hash randomization."""
    payload = f'{seed}:{row_number}:{salt}'.encode('ascii')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'big') / 2**64


def build_sample(
    source: str | Path,
    output_dir: str | Path,
    *,
    sample_rate: float = 0.05,
    validation_fraction: float = 0.2,
    seed: int = 2026,
    limit: int | None = None,
) -> dict:
    """Stream published train rows into disjoint development train/validation CSVs.

    Uniform sampling precedes split assignment. No test.csv access. `limit`
    is a smoke-test control and never means a random sample of the full file.
    """
    if not 0 < sample_rate <= 1:
        raise ValueError('sample_rate must be in (0, 1]')
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be in (0, 1)')
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    source = Path(source)
    if source.name != 'train.csv':
        raise ValueError('source must be the published train.csv, never test.csv')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {'train': output_dir / 'train_features.csv', 'validation': output_dir / 'validation_features.csv'}
    counts = {'train': {'rows': 0, 'clicks': 0}, 'validation': {'rows': 0, 'clicks': 0}}
    scanned = 0
    with paths['train'].open('w', newline='', encoding='utf-8') as train_file, paths['validation'].open('w', newline='', encoding='utf-8') as val_file:
        writers = {'train': csv.DictWriter(train_file, fieldnames=FEATURE_COLUMNS),
                   'validation': csv.DictWriter(val_file, fieldnames=FEATURE_COLUMNS)}
        for writer in writers.values():
            writer.writeheader()
        for index, row in enumerate(iter_rows(source, limit=limit)):
            scanned += 1
            if _uniform(seed, index, 'sample') >= sample_rate:
                continue
            split = 'validation' if _uniform(seed, index, 'split') < validation_fraction else 'train'
            writers[split].writerow(feature_row(row))
            counts[split]['rows'] += 1
            counts[split]['clicks'] += int(row['clk'])
    manifest = {
        'source': str(source), 'source_split': 'published train only',
        'sampling': 'uniform deterministic row-index hash',
        'split': 'deterministic row-index hash; NOT chronological or user-disjoint',
        'seed': seed, 'sample_rate': sample_rate,
        'validation_fraction': validation_fraction, 'source_row_limit': limit,
        'source_rows_scanned': scanned, 'counts': counts,
        'feature_columns': list(FEATURE_COLUMNS),
        'history_point_in_time_verified': False,
        'warning': 'Source histories may contain leakage; test.csv remains untouched.',
    }
    (output_dir / 'sample_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest
