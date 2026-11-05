"""M05F-A: split the existing M03A validation partition by deterministic row hash.

No published test data is accessed. This is not a temporal or user-disjoint split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from .ctr_features import FEATURE_COLUMNS

PARTITIONS = ('early_stopping', 'calibration', 'evaluation')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _bucket(seed: int, index: int) -> float:
    payload = f'm05fa:{seed}:{index}'.encode('ascii')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'big') / 2**64


def split_validation(input_dir: str | Path, output_dir: str | Path, *, seed: int = 2026,
                     early_stopping_fraction: float = 0.34,
                     calibration_fraction: float = 0.33) -> dict:
    """Partition M03A validation rows; fail closed if provenance is unexpected."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not (0 < early_stopping_fraction < 1 and 0 < calibration_fraction < 1
            and early_stopping_fraction + calibration_fraction < 1):
        raise ValueError('fractions must be positive and leave an evaluation partition')
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    manifest_path = input_dir / 'sample_manifest.json'
    original = json.loads(manifest_path.read_text(encoding='utf-8'))
    if original.get('source_split') != 'published train only':
        raise ValueError('M03A published-train-only provenance required')
    if original.get('feature_columns') != list(FEATURE_COLUMNS):
        raise ValueError('M03A feature schema mismatch')
    source = input_dir / 'validation_features.csv'
    training = input_dir / 'train_features.csv'
    if not training.is_file() or not source.is_file():
        raise FileNotFoundError('M03A training and validation features required')
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError('output directory must differ from M03A input directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: output_dir / f'{name}_features.csv' for name in PARTITIONS}
    report_path = output_dir / 'split_manifest.json'
    if report_path.exists() or any(p.exists() for p in targets.values()):
        raise FileExistsError('M05F-A output already exists; use a fresh directory')

    counts = {name: {'rows': 0, 'clicks': 0} for name in PARTITIONS}
    handles = {}
    try:
        for name, path in targets.items():
            handles[name] = path.open('x', newline='', encoding='utf-8')
        writers = {name: csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
                   for name, handle in handles.items()}
        for writer in writers.values():
            writer.writeheader()
        with source.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FEATURE_COLUMNS):
                raise ValueError('Unexpected validation feature schema')
            for index, row in enumerate(reader):
                if row['clk'] not in ('0', '1'):
                    raise ValueError(f'Invalid click label at row {index}')
                value = _bucket(seed, index)
                name = ('early_stopping' if value < early_stopping_fraction else
                        'calibration' if value < early_stopping_fraction + calibration_fraction
                        else 'evaluation')
                writers[name].writerow(row)
                counts[name]['rows'] += 1
                counts[name]['clicks'] += int(row['clk'])
    except Exception:
        for handle in handles.values():
            handle.close()
        for path in targets.values():
            path.unlink(missing_ok=True)
        raise
    finally:
        for handle in handles.values():
            if not handle.closed:
                handle.close()

    expected = original.get('counts', {}).get('validation', {})
    if sum(v['rows'] for v in counts.values()) != expected.get('rows') or \
       sum(v['clicks'] for v in counts.values()) != expected.get('clicks'):
        for path in targets.values():
            path.unlink(missing_ok=True)
        raise ValueError('M03A manifest counts do not match validation data')
    if any(counts[name]['rows'] == 0 or counts[name]['clicks'] == 0 or
           counts[name]['clicks'] == counts[name]['rows'] for name in PARTITIONS):
        for path in targets.values():
            path.unlink(missing_ok=True)
        raise ValueError('Each partition requires both click classes')
    report = {
        'milestone': 'M05F-A', 'schema_version': 1, 'seed': seed,
        'method': 'deterministic row-index blake2b hash on existing M03A validation rows',
        'fractions': {'early_stopping': early_stopping_fraction,
                      'calibration': calibration_fraction,
                      'evaluation': 1 - early_stopping_fraction - calibration_fraction},
        'source_manifest_sha256': _sha256(manifest_path),
        'source_validation_sha256': _sha256(source),
        'source_training_sha256': _sha256(training),
        'counts': counts,
        'files': {name: {'name': targets[name].name, 'sha256': _sha256(targets[name])}
                  for name in PARTITIONS},
        'published_test_used': False, 'temporal_split': False,
        'user_disjoint': False, 'history_point_in_time_verified': False,
        'note': 'Early stopping, calibration, and final evaluation roles are disjoint by row only; '
                'source histories may leak future information.',
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default='artifacts/m03a')
    parser.add_argument('--output-dir', default='artifacts/m05fa/splits')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--early-stopping-fraction', type=float, default=0.34)
    parser.add_argument('--calibration-fraction', type=float, default=0.33)
    args = parser.parse_args()
    result = split_validation(args.input_dir, args.output_dir, seed=args.seed,
                              early_stopping_fraction=args.early_stopping_fraction,
                              calibration_fraction=args.calibration_fraction)
    print(json.dumps({'counts': result['counts'], 'output_dir': args.output_dir}, sort_keys=True))


if __name__ == '__main__':
    main()
