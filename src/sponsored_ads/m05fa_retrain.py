"""M05F-A: retrain CTR using a dedicated early-stopping split.

Calibration and final evaluation partitions are never read for model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .ctr_baselines import FULL, train_lightgbm
from .ctr_contract import load_lightgbm_bundle, save_contract


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_split_manifest(manifest: dict) -> None:
    if manifest.get('published_test_used') is not False:
        raise ValueError('Split manifest must explicitly exclude published test')
    counts = manifest.get('counts')
    if not isinstance(counts, dict) or set(counts) != {'early_stopping', 'calibration', 'evaluation'}:
        raise ValueError('Expected three named validation partitions')
    for name, entry in counts.items():
        if not isinstance(entry, dict) or not isinstance(entry.get('rows'), int) or not isinstance(entry.get('clicks'), int):
            raise ValueError(f'Invalid count metadata: {name}')
        if entry['rows'] <= 0 or not 0 <= entry['clicks'] <= entry['rows']:
            raise ValueError(f'Invalid partition counts: {name}')


def retrain(input_dir: str | Path, split_dir: str | Path, output_dir: str | Path, *, seed: int = 2026, n_estimators: int = 250) -> dict:
    input_dir, split_dir, output_dir = map(Path, (input_dir, split_dir, output_dir))
    if seed < 0 or n_estimators <= 0:
        raise ValueError('Invalid seed or estimator count')
    source_manifest = input_dir / 'sample_manifest.json'
    split_manifest_path = split_dir / 'split_manifest.json'
    source = json.loads(source_manifest.read_text(encoding='utf-8'))
    if source.get('source_split') != 'published train only':
        raise ValueError('Expected published-train-only source')
    split_manifest = json.loads(split_manifest_path.read_text(encoding='utf-8'))
    _validate_split_manifest(split_manifest)
    train_path = input_dir / 'train_features.csv'
    early_path = split_dir / 'early_stopping_features.csv'
    for path in (train_path, early_path, split_dir / 'calibration_features.csv', split_dir / 'evaluation_features.csv'):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, training_report, contract = train_lightgbm(
        train_path, early_path, features=FULL, seed=seed,
        n_estimators=n_estimators, return_contract=True,
    )
    model_path = output_dir / 'lightgbm_intent.txt'
    contract_path = output_dir / 'lightgbm_intent.contract.json'
    model.booster_.save_model(str(model_path))
    contract_sha = save_contract(contract, contract_path)
    loaded, loaded_contract = load_lightgbm_bundle(model_path, contract_path)
    if loaded.num_feature() != len(loaded_contract['features']):
        raise ValueError('Saved bundle failed verification')
    report = {
        'milestone': 'M05F-A', 'seed': seed, 'n_estimators': n_estimators,
        'published_test_used': False,
        'training_partition': 'M03A train_features.csv',
        'early_stopping_partition': 'M05F-A early_stopping_features.csv',
        'calibration_partition_used_for_fitting': False,
        'evaluation_partition_used_for_fitting': False,
        'split_is_temporal': False, 'split_is_user_disjoint': False,
        'history_point_in_time_verified': False,
        'input_sha256': {
            'sample_manifest': sha256(source_manifest),
            'split_manifest': sha256(split_manifest_path),
            'train_features': sha256(train_path),
            'early_stopping_features': sha256(early_path),
        },
        'output_sha256': {'model': sha256(model_path), 'contract': contract_sha},
        'training': training_report,
        'note': 'Early-stopping metrics are model-selection diagnostics, not independent calibration evaluation.',
    }
    (output_dir / 'retrain_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--split-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--n-estimators', type=int, default=250)
    args = parser.parse_args()
    result = retrain(args.input_dir, args.split_dir, args.output_dir, seed=args.seed, n_estimators=args.n_estimators)
    print(json.dumps({'output_dir': args.output_dir, 'best_iteration': result['training']['best_iteration'], 'contract_sha256': result['output_sha256']['contract']}, sort_keys=True))


if __name__ == '__main__':
    main()
