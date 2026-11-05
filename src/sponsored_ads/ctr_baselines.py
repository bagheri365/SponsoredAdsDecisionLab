"""M03B CTR benchmarks; reads only M03A prepared train/validation files.

Requires optional dependencies: pip install -e '.[ml,dev]'.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import inspect
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, brier_score_loss

from .ctr_features import STATIC_CATEGORICAL, NUMERIC, FEATURE_COLUMNS

INTENT = tuple(name for name in NUMERIC if name != 'price')
STATIC = (*STATIC_CATEGORICAL, 'price')
FULL = (*STATIC_CATEGORICAL, *NUMERIC)


def _read(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(FEATURE_COLUMNS):
            raise ValueError(f'Unexpected feature schema: {path}')
        for row in reader:
            if row['clk'] not in ('0', '1'):
                raise ValueError('clk must be 0 or 1')
            yield row


def _number(row, name):
    value = float(row[name])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return math.log1p(value)  # Stabilize price and history count scales for SGD.


def _hash_features(row, features):
    result = {f'{name}={row[name]}': 1.0 for name in STATIC_CATEGORICAL if name in features}
    result.update({name: _number(row, name) for name in NUMERIC if name in features})
    return result


def _metrics(y, probabilities):
    if len(y) == 0 or len(set(y.tolist())) != 2:
        raise ValueError('Evaluation needs both click classes')
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        'rows': int(len(y)), 'clicks': int(y.sum()), 'ctr': float(y.mean()),
        'log_loss': float(log_loss(y, p)),
        'roc_auc': float(roc_auc_score(y, p)),
        'pr_auc': float(average_precision_score(y, p)),
        'brier_score': float(brier_score_loss(y, p)),
        'mean_predicted_ctr': float(p.mean()),
        'calibration_bins': _calibration(y, p),
    }


def _calibration(y, p, bins=10):
    result = []
    for i in range(bins):
        mask = (p >= i / bins) & (p < (i + 1) / bins if i < bins - 1 else p <= 1)
        if mask.any():
            result.append({'lower': i / bins, 'upper': (i + 1) / bins,
                           'rows': int(mask.sum()), 'observed_ctr': float(y[mask].mean()),
                           'predicted_ctr': float(p[mask].mean())})
    return result


def train_logistic(train_path, validation_path, *, seed=2026, hash_dim=2**18, batch_size=8192):
    """Out-of-core logistic regression via SGD and a fixed hashing vocabulary."""
    hasher = FeatureHasher(n_features=hash_dim, input_type='dict', alternate_sign=False)
    model = SGDClassifier(loss='log_loss', alpha=1e-5, random_state=seed,
                          learning_rate='optimal', average=True)
    first = True
    batch = []
    n_train = 0
    def fit_batch(items, first_batch):
        X = hasher.transform(_hash_features(r, STATIC) for r in items)
        y = np.fromiter((int(r['clk']) for r in items), dtype=np.int8)
        if first_batch:
            model.partial_fit(X, y, classes=np.array([0, 1]))
        else:
            model.partial_fit(X, y)
    for row in _read(Path(train_path)):
        batch.append(row)
        if len(batch) >= batch_size:
            fit_batch(batch, first)
            n_train += len(batch)
            first = False
            batch = []
    if batch:
        fit_batch(batch, first)
        n_train += len(batch)
    if not n_train:
        raise ValueError('Empty training set')
    labels, scores = [], []
    batch = []
    def score_batch(items):
        X = hasher.transform(_hash_features(r, STATIC) for r in items)
        scores.extend(model.predict_proba(X)[:, 1].tolist())
        labels.extend(int(r['clk']) for r in items)
    for row in _read(Path(validation_path)):
        batch.append(row)
        if len(batch) >= batch_size:
            score_batch(batch)
            batch = []
    if batch:
        score_batch(batch)
    return model, {'algorithm': 'sgd_logistic_hashing', 'features': 'static',
                   'train_rows': n_train, 'hash_dim': hash_dim,
                   'metrics': _metrics(np.asarray(labels), np.asarray(scores))}


def _category_code(value):
    # Stable nonnegative categorical IDs; collisions possible and documented.
    return int.from_bytes(hashlib.blake2b(value.encode('utf-8'), digest_size=4).digest(), 'little') % (2**20)


def _matrix(path, features, *, category_maps=None):
    """Encode categorical IDs using train-fitted contiguous codes.

    Unknown validation IDs are encoded as -1 (LightGBM missing category).
    Maps are fitted only when category_maps is None; never fit on validation.
    """
    fitting = category_maps is None
    if fitting:
        category_maps = {name: {} for name in features if name in STATIC_CATEGORICAL}
    columns, labels = [], []
    for row in _read(Path(path)):
        values = []
        for name in features:
            if name in STATIC_CATEGORICAL:
                mapping = category_maps[name]
                value = row[name]
                if fitting and value not in mapping:
                    mapping[value] = len(mapping)
                values.append(mapping.get(value, -1))
            else:
                values.append(float(row[name]))
        columns.append(values)
        labels.append(int(row['clk']))
    if not columns:
        raise ValueError(f'Empty dataset: {path}')
    return (np.asarray(columns, dtype=np.float32),
            np.asarray(labels, dtype=np.int8), category_maps)


def train_lightgbm(train_path, validation_path, *, features=FULL, seed=2026, n_estimators=250):
    import lightgbm as lgb
    X_train, y_train, category_maps = _matrix(train_path, features)
    X_val, y_val, _ = _matrix(validation_path, features, category_maps=category_maps)
    categorical_indices = [i for i, name in enumerate(features) if name in STATIC_CATEGORICAL]
    model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05,
                               num_leaves=31, max_depth=-1, min_child_samples=100,
                               reg_lambda=1.0, random_state=seed, n_jobs=-1,
                               verbosity=-1)
    fit_kwargs = {'categorical_feature': categorical_indices,
                  'eval_metric': 'binary_logloss',
                  'callbacks': [lgb.early_stopping(25, verbose=False)]}
    # Support both older LightGBM releases and releases deprecating eval_set.
    if 'eval_X' in inspect.signature(model.fit).parameters:
        fit_kwargs.update(eval_X=X_val, eval_y=y_val)
    else:
        fit_kwargs['eval_set'] = [(X_val, y_val)]
    model.fit(X_train, y_train, **fit_kwargs)
    scores = model.predict_proba(X_val)[:, 1]
    return model, {'algorithm': 'lightgbm', 'features': 'static_plus_intent' if features == FULL else 'static',
                   'train_rows': int(len(y_train)), 'best_iteration': int(model.best_iteration_),
                   'category_encoding': 'train_fitted_contiguous; unknown=-1',
                   'metrics': _metrics(y_val, scores)}


def run(input_dir, output_dir, *, seed=2026, n_estimators=250):
    """Runs all three models; never opens the published Taobao test.csv."""
    import joblib
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = input_dir / 'train_features.csv'
    val = input_dir / 'validation_features.csv'
    manifest = input_dir / 'sample_manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError('M03A sample_manifest.json required')
    provenance = json.loads(manifest.read_text())
    if provenance.get('source_split') != 'published train only':
        raise ValueError('Expected M03A published-train-only provenance')
    results = {}
    lr, results['m0_logistic_static'] = train_logistic(train, val, seed=seed)
    joblib.dump(lr, output_dir / 'm0_logistic.joblib')
    for name, features in [('m1_lightgbm_static', STATIC), ('m2_lightgbm_intent', FULL)]:
        model, results[name] = train_lightgbm(train, val, features=features,
                                             seed=seed, n_estimators=n_estimators)
        model.booster_.save_model(str(output_dir / f'{name}.txt'))
        # Categorical encodings must be retained for future serving; this
        # experiment currently trains them inside train_lightgbm, so serving
        # remains unsupported until the preprocessing contract is persisted.
    report = {
        'milestone': 'M03B', 'seed': seed, 'n_estimators': n_estimators,
        'input_manifest': str(manifest), 'input_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'published_test_used': False, 'validation_is_temporal': False,
        'validation_is_user_disjoint': False, 'history_point_in_time_verified': False,
        'note': 'Offline observed-impression CTR only; not unbiased ad policy evaluation.',
        'models': results,
    }
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
