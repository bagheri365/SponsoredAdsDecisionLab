"""M05F-B: fit CTR calibrators on reserved rows, evaluate on untouched rows.

This experiment never selects a production calibrator or accesses published test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .ctr_calibration import _logit, diagnostics
from .ctr_contract import load_lightgbm_bundle, predict_lightgbm
from .ctr_features import FEATURE_COLUMNS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _score_partition(path, model, contract, *, batch_size=8192):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be positive')
    labels, probabilities, batch = [], [], []

    def flush():
        if not batch:
            return
        scores = np.asarray(predict_lightgbm(model, contract, batch), dtype=float)
        if scores.shape != (len(batch),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError('Model must return one finite probability in [0,1] per row')
        probabilities.extend(scores.tolist())
        batch.clear()

    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FEATURE_COLUMNS):
            raise ValueError('Unexpected feature schema')
        for row in reader:
            if row['clk'] not in ('0', '1'):
                raise ValueError('Invalid click label')
            labels.append(int(row['clk']))
            batch.append(row)
            if len(batch) >= batch_size:
                flush()
    flush()
    if not labels or len(set(labels)) != 2:
        raise ValueError('Both click classes required')
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def _verify_inputs(split_dir, model_dir):
    split_path = split_dir / 'split_manifest.json'
    retrain_path = model_dir / 'retrain_report.json'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    retrain = json.loads(retrain_path.read_text(encoding='utf-8'))
    if split.get('schema_version') != 1 or split.get('published_test_used') is not False:
        raise ValueError('Unverified split provenance')
    if retrain.get('published_test_used') is not False or retrain.get('calibration_partition_used_for_fitting') is not False or retrain.get('evaluation_partition_used_for_fitting') is not False:
        raise ValueError('Retraining provenance does not establish holdouts')
    if retrain.get('input_sha256', {}).get('split_manifest') != sha256(split_path):
        raise ValueError('Retraining/split manifest mismatch')
    model_path = model_dir / 'lightgbm_intent.txt'
    contract_path = model_dir / 'lightgbm_intent.contract.json'
    for name, path in (('model', model_path), ('contract', contract_path)):
        if retrain.get('output_sha256', {}).get(name) != sha256(path):
            raise ValueError(f'{name} hash mismatch')
    paths = {}
    for name in ('calibration', 'evaluation'):
        entry = split.get('files', {}).get(name)
        if not isinstance(entry, dict) or entry.get('name') != f'{name}_features.csv':
            raise ValueError(f'Invalid {name} file provenance')
        path = split_dir / entry['name']
        if sha256(path) != entry.get('sha256'):
            raise ValueError(f'{name} input hash mismatch')
        paths[name] = path
    if paths['calibration'].resolve() == paths['evaluation'].resolve():
        raise ValueError('Calibration and evaluation must be separate files')
    return split, retrain, paths, model_path, contract_path, split_path, retrain_path


def run(split_dir, model_dir, output, *, bins=10, batch_size=8192):
    split_dir, model_dir, output = Path(split_dir), Path(model_dir), Path(output)
    if type(bins) is not int or bins < 2:
        raise ValueError('bins must be >= 2')
    split, retrain, paths, model_path, contract_path, split_path, retrain_path = _verify_inputs(split_dir, model_dir)
    if output.resolve() in {p.resolve() for p in (*paths.values(), model_path, contract_path, split_path, retrain_path)}:
        raise ValueError('Output must not overwrite input artifacts')
    model, contract = load_lightgbm_bundle(model_path, contract_path)
    y_fit, p_fit = _score_partition(paths['calibration'], model, contract, batch_size=batch_size)
    y_eval, p_eval = _score_partition(paths['evaluation'], model, contract, batch_size=batch_size)
    for name, labels in (('calibration', y_fit), ('evaluation', y_eval)):
        expected = split['counts'][name]
        if len(labels) != expected['rows'] or int(labels.sum()) != expected['clicks']:
            raise ValueError(f'{name} counts differ from split manifest')
    sigmoid = LogisticRegression(random_state=2026, max_iter=1000)
    sigmoid.fit(_logit(p_fit), y_fit)
    isotonic = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    isotonic.fit(p_fit, y_fit)
    predictions = {'raw': p_eval, 'sigmoid': sigmoid.predict_proba(_logit(p_eval))[:, 1],
                   'isotonic': isotonic.predict(p_eval)}
    report = {
        'schema_version': 1, 'milestone': 'M05F-B',
        'experiment': 'real_observed_impression_ctr_calibration',
        'fit_partition': 'calibration', 'evaluation_partition': 'evaluation',
        'calibration_rows': int(len(y_fit)), 'calibration_clicks': int(y_fit.sum()),
        'evaluation_rows': int(len(y_eval)), 'evaluation_clicks': int(y_eval.sum()),
        'bins': bins,
        'metrics': {name: diagnostics(y_eval, p, bins=bins) for name, p in predictions.items()},
        'sha256': {'split_manifest': sha256(split_path), 'retrain_report': sha256(retrain_path),
                   'model': sha256(model_path), 'contract': sha256(contract_path),
                   'calibration_features': sha256(paths['calibration']),
                   'evaluation_features': sha256(paths['evaluation'])},
        'selection': 'No method selected or promoted; evaluation comparisons are exploratory',
        'limitations': [
            'Row-hash partitions are neither temporal nor user-disjoint',
            'History point-in-time validity is unverified',
            'Observed impressions only; no eligible inventory or counterfactual policy evaluation',
            'Comparing methods on this evaluation split makes subsequent selection on the same split optimistic',
            'No calibrated serving artifact is persisted by this experiment',
        ],
        'published_test_used': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing experiment report')
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', default='artifacts/m05fa/splits')
    parser.add_argument('--model-dir', default='artifacts/m05fa/model')
    parser.add_argument('--output', default='artifacts/m05fb/calibration_report.json')
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args(argv)
    report = run(args.split_dir, args.model_dir, args.output, bins=args.bins, batch_size=args.batch_size)
    print(json.dumps({'output': args.output, 'calibration_rows': report['calibration_rows'],
                      'evaluation_rows': report['evaluation_rows'],
                      'metrics': {k: {m: v[m] for m in ('log_loss', 'brier_score', 'roc_auc', 'ece')}
                                  for k, v in report['metrics'].items()}}, sort_keys=True))


if __name__ == '__main__':
    main()
