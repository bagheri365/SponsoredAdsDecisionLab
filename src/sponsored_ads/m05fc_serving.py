"""M05F-C: explicit, hash-bound CTR calibration for serving.

No evaluation labels are read. Raw is the default; sigmoid/isotonic require
an explicit policy choice. JSON stores only numeric calibrator parameters.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .ctr_calibration import _logit
from .ctr_contract import load_lightgbm_bundle, predict_lightgbm
from .m05fb_experiment import _score_partition, _verify_inputs, sha256

VERSION = 1
METHODS = ('raw', 'sigmoid', 'isotonic')


def _validate_probability(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected nonempty vector of finite probabilities in [0,1]')
    return p


def _validate_artifact(spec):
    if not isinstance(spec, dict) or set(spec) != {'schema_version', 'method', 'model_sha256',
                                                  'contract_sha256', 'calibration_sha256',
                                                  'fit_rows', 'fit_clicks', 'parameters'}:
        raise ValueError('Invalid calibration artifact schema')
    if type(spec['schema_version']) is not int or spec['schema_version'] != VERSION or spec['method'] not in METHODS:
        raise ValueError('Unsupported calibration method/version')
    for name in ('model_sha256', 'contract_sha256', 'calibration_sha256'):
        value = spec[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Invalid {name}')
    rows, clicks = spec['fit_rows'], spec['fit_clicks']
    if type(rows) is not int or type(clicks) is not int or rows < 2 or not 0 < clicks < rows:
        raise ValueError('Invalid calibration counts')
    params = spec['parameters']
    if not isinstance(params, dict):
        raise ValueError('Invalid calibration parameters')
    if spec['method'] == 'raw':
        if params:
            raise ValueError('Raw method has no parameters')
    elif spec['method'] == 'sigmoid':
        if set(params) != {'coefficient', 'intercept'}:
            raise ValueError('Invalid sigmoid parameters')
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in params.values()):
            raise ValueError('Nonfinite sigmoid parameters')
    else:
        if set(params) != {'x', 'y'} or not isinstance(params['x'], list) or not isinstance(params['y'], list):
            raise ValueError('Invalid isotonic parameters')
        x, y = params['x'], params['y']
        if len(x) < 2 or len(x) != len(y):
            raise ValueError('Invalid isotonic knots')
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in x + y):
            raise ValueError('Nonfinite isotonic knots')
        if any(not 0 <= v <= 1 for v in x + y):
            raise ValueError('Isotonic knots outside [0,1]')
        if any(a >= b for a, b in zip(x, x[1:])) or any(a > b for a, b in zip(y, y[1:])):
            raise ValueError('Isotonic knots must be ordered')
    return spec


def apply_calibration(probabilities, artifact):
    spec = _validate_artifact(artifact)
    p = _validate_probability(probabilities)
    if spec['method'] == 'raw':
        result = p.copy()
    elif spec['method'] == 'sigmoid':
        z = _logit(p).reshape(-1) * spec['parameters']['coefficient'] + spec['parameters']['intercept']
        result = np.exp(-np.logaddexp(0, -z))
    else:
        knots = spec['parameters']
        result = np.interp(p, knots['x'], knots['y'])
    return _validate_probability(result)


def fit_artifact(labels, probabilities, *, method, model_sha256, contract_sha256, calibration_sha256):
    if method not in METHODS:
        raise ValueError('Explicit method must be raw, sigmoid, or isotonic')
    p = _validate_probability(probabilities)
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(p) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Binary labels with both classes required')
    params = {}
    if method == 'sigmoid':
        model = LogisticRegression(random_state=2026, max_iter=1000)
        model.fit(_logit(p), y)
        params = {'coefficient': float(model.coef_[0, 0]), 'intercept': float(model.intercept_[0])}
    elif method == 'isotonic':
        model = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        model.fit(p, y)
        params = {'x': model.X_thresholds_.astype(float).tolist(),
                  'y': model.y_thresholds_.astype(float).tolist()}
    artifact = {'schema_version': VERSION, 'method': method,
                'model_sha256': model_sha256, 'contract_sha256': contract_sha256,
                'calibration_sha256': calibration_sha256, 'fit_rows': int(len(y)),
                'fit_clicks': int(np.sum(y)), 'parameters': params}
    return _validate_artifact(artifact)


def save_artifact(artifact, path):
    _validate_artifact(artifact)
    path = Path(path)
    if path.exists():
        raise FileExistsError('Refusing to overwrite calibration artifact')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(artifact, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return sha256(path)


def load_serving_bundle(model_path, contract_path, artifact_path, *, method='raw'):
    if method not in METHODS:
        raise ValueError('Unsupported serving policy')
    model, contract = load_lightgbm_bundle(model_path, contract_path)
    if method == 'raw' and artifact_path is None:
        return model, contract, None
    if artifact_path is None:
        raise ValueError('Calibrated policy requires an artifact')
    spec = _validate_artifact(json.loads(Path(artifact_path).read_text(encoding='utf-8')))
    if spec['method'] != method:
        raise ValueError('Calibration policy/artifact mismatch')
    if spec['model_sha256'] != sha256(model_path) or spec['contract_sha256'] != sha256(contract_path):
        raise ValueError('Calibration artifact incompatible with model/contract')
    return model, contract, spec


def predict_calibrated(model, contract, rows, artifact=None):
    raw = _validate_probability(predict_lightgbm(model, contract, rows))
    return raw if artifact is None else apply_calibration(raw, artifact)


def run_fit(split_dir, model_dir, output, *, method, batch_size=8192):
    if method not in METHODS:
        raise ValueError('Explicit calibration policy required')
    split_dir, model_dir, output = Path(split_dir), Path(model_dir), Path(output)
    split, _, paths, model_path, contract_path, _, _ = _verify_inputs(split_dir, model_dir)
    if output.exists() or output.resolve() in {p.resolve() for p in (*paths.values(), model_path, contract_path)}:
        raise FileExistsError('Output must not overwrite existing files')
    model, contract = load_lightgbm_bundle(model_path, contract_path)
    labels, probabilities = _score_partition(paths['calibration'], model, contract, batch_size=batch_size)
    expected = split['counts']['calibration']
    if len(labels) != expected['rows'] or int(labels.sum()) != expected['clicks']:
        raise ValueError('Calibration partition count mismatch')
    artifact = fit_artifact(labels, probabilities, method=method,
                            model_sha256=sha256(model_path), contract_sha256=sha256(contract_path),
                            calibration_sha256=sha256(paths['calibration']))
    digest = save_artifact(artifact, output)
    return {'output': str(output), 'artifact_sha256': digest, 'method': method,
            'fit_rows': artifact['fit_rows'], 'fit_clicks': artifact['fit_clicks']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', default='artifacts/m05fa/splits')
    parser.add_argument('--model-dir', default='artifacts/m05fa/model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args(argv)
    print(json.dumps(run_fit(args.split_dir, args.model_dir, args.output,
                             method=args.method, batch_size=args.batch_size), sort_keys=True))


if __name__ == '__main__':
    main()
