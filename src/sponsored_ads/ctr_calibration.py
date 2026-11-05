"""M05E: diagnostics for out-of-sample CTR probabilities on logged impressions.

Input scores MUST be generated without training on the calibration/evaluation
rows. The caller owns upstream provenance; this module cannot prove it.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

EPS = 1e-7


def _inputs(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) < 8:
        raise ValueError('Expected matching one-dimensional arrays with at least 8 rows')
    if any(type(v) not in (int, np.int64, np.int32, np.int8, np.int16, np.uint8, bool, np.bool_)
           or int(v) not in (0, 1) for v in y):
        raise ValueError('Labels must be binary integers')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and in [0,1]')
    return y.astype(int), p


def reliability_bins(labels, probabilities, *, bins=10):
    if type(bins) is not int or bins < 2:
        raise ValueError('bins must be an integer >= 2')
    y, p = _inputs(labels, probabilities)
    indices = np.minimum((p * bins).astype(int), bins - 1)
    return [dict(lower=i / bins, upper=(i + 1) / bins,
                 rows=int(mask.sum()), observed_ctr=float(y[mask].mean()),
                 predicted_ctr=float(p[mask].mean()))
            for i in range(bins) if (mask := indices == i).any()]


def diagnostics(labels, probabilities, *, bins=10):
    y, p = _inputs(labels, probabilities)
    clipped = np.clip(p, EPS, 1 - EPS)
    groups = reliability_bins(y, p, bins=bins)
    ece = sum(group['rows'] / len(y) * abs(group['observed_ctr'] - group['predicted_ctr'])
              for group in groups)
    return dict(rows=len(y), clicks=int(y.sum()), observed_ctr=float(y.mean()),
                predicted_ctr=float(p.mean()), log_loss=float(log_loss(y, clipped, labels=[0, 1])),
                brier_score=float(brier_score_loss(y, p)),
                roc_auc=(float(roc_auc_score(y, p)) if len(set(y)) == 2 else None),
                ece=float(ece), reliability_bins=groups)


def _logit(probabilities):
    p = np.clip(probabilities, EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def compare_calibrators(labels, probabilities, *, seed=2026, fit_fraction=0.5, bins=10):
    """Fit calibrators on one partition; evaluate on a disjoint partition.

    Both classes must occur at least twice to permit stratified splitting.
    This does not guarantee independence from the original CTR model fit.
    """
    y, p = _inputs(labels, probabilities)
    if type(seed) is not int or seed < 0 or not 0 < fit_fraction < 1:
        raise ValueError('Invalid seed or fit_fraction')
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2:
        raise ValueError('At least two examples of each class required')
    indices = np.arange(len(y))
    fit_idx, eval_idx = train_test_split(indices, train_size=fit_fraction,
                                         random_state=seed, stratify=y)
    if len(set(y[fit_idx])) < 2 or len(set(y[eval_idx])) < 2:
        raise ValueError('Both classes required in fit and evaluation partitions')
    sigmoid = LogisticRegression(random_state=seed, max_iter=1000)
    sigmoid.fit(_logit(p[fit_idx]), y[fit_idx])
    isotonic = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    isotonic.fit(p[fit_idx], y[fit_idx])
    predictions = {
        'raw': p[eval_idx],
        'sigmoid': sigmoid.predict_proba(_logit(p[eval_idx]))[:, 1],
        'isotonic': isotonic.predict(p[eval_idx]),
    }
    return dict(schema_version=1, seed=seed, fit_fraction=fit_fraction,
                fit_rows=len(fit_idx), evaluation_rows=len(eval_idx),
                fit_clicks=int(y[fit_idx].sum()), evaluation_clicks=int(y[eval_idx].sum()),
                split='stratified random split of supplied out-of-sample predictions',
                metrics={name: diagnostics(y[eval_idx], scores, bins=bins)
                         for name, scores in predictions.items()},
                limitations=[
                    'Upstream CTR model and score provenance are not verified by this module',
                    'Random split is not temporal or user-disjoint',
                    'Observed impressions only; no eligible inventory or counterfactual labels',
                    'Do not choose a calibrator using evaluation metrics and report the same split as final test',
                ])


def read_predictions(path):
    """Read a CSV with exactly `clk,probability` columns."""
    labels, scores = [], []
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['clk', 'probability']:
            raise ValueError('Expected CSV columns: clk,probability')
        for row in reader:
            if row['clk'] not in ('0', '1'):
                raise ValueError('clk must be 0 or 1')
            labels.append(int(row['clk']))
            scores.append(float(row['probability']))
    return _inputs(labels, scores)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='CSV: clk,probability (out-of-sample)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--fit-fraction', type=float, default=0.5)
    parser.add_argument('--bins', type=int, default=10)
    args = parser.parse_args(argv)
    y, p = read_predictions(args.input)
    report = compare_calibrators(y, p, seed=args.seed, fit_fraction=args.fit_fraction, bins=args.bins)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({name: {k: value for k, value in metrics.items()
                             if k in ('log_loss', 'brier_score', 'roc_auc', 'ece')}
                      for name, metrics in report['metrics'].items()}, sort_keys=True))


if __name__ == '__main__':
    main()
