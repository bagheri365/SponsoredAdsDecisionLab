import csv
import json

import numpy as np
import pytest

from sponsored_ads.ctr_calibration import (
    compare_calibrators, diagnostics, main, read_predictions, reliability_bins,
)


def sample():
    rng = np.random.default_rng(2026)
    p = rng.uniform(0.01, 0.7, size=200)
    y = rng.binomial(1, p)
    return y, p


def test_reproducible_and_disjoint_split():
    y, p = sample()
    first = compare_calibrators(y, p)
    second = compare_calibrators(y, p)
    assert {k: v for k, v in first.items() if k != 'metrics'} == {
        k: v for k, v in second.items() if k != 'metrics'
    }
    for method in first['metrics']:
        left, right = first['metrics'][method], second['metrics'][method]
        assert left.keys() == right.keys()
        for key in left:
            if key == 'reliability_bins':
                assert len(left[key]) == len(right[key])
                for bin_left, bin_right in zip(left[key], right[key]):
                    assert bin_left == pytest.approx(bin_right, rel=1e-8, abs=1e-10)
            else:
                assert left[key] == pytest.approx(right[key], rel=1e-8, abs=1e-10)
    assert first['fit_rows'] + first['evaluation_rows'] == len(y)
    assert first['fit_clicks'] + first['evaluation_clicks'] == int(y.sum())
    assert set(first['metrics']) == {'raw', 'sigmoid', 'isotonic'}


def test_metrics_valid():
    y, p = sample()
    result = diagnostics(y, p)
    assert 0 <= result['ece'] <= 1
    assert result['roc_auc'] is not None
    assert sum(x['rows'] for x in result['reliability_bins']) == len(y)


def test_reliability_edges():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0, 1, 0.5, 0.5, 0.2, 0.8, 0.1, 0.9])
    assert sum(x['rows'] for x in reliability_bins(y, p)) == 8


@pytest.mark.parametrize('p', [[-0.1] * 8, [1.1] * 8, [float('nan')] * 8])
def test_invalid_probability(p):
    with pytest.raises(ValueError, match='Probabilities'):
        diagnostics([0, 1] * 4, p)


def test_invalid_labels():
    with pytest.raises(ValueError, match='binary'):
        diagnostics([0, 2] * 4, [0.2] * 8)


def test_single_class_fails_split():
    with pytest.raises(ValueError, match='two examples'):
        compare_calibrators([0] * 10, [0.2] * 10)


def test_invalid_fit_fraction():
    y, p = sample()
    with pytest.raises(ValueError, match='fit_fraction'):
        compare_calibrators(y, p, fit_fraction=1)


def test_csv_roundtrip_and_cli(tmp_path, capsys):
    y, p = sample()
    input_path = tmp_path / 'predictions.csv'
    with input_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clk', 'probability'])
        writer.writerows(zip(y, p))
    labels, scores = read_predictions(input_path)
    assert np.array_equal(labels, y)
    assert np.allclose(scores, p)
    output = tmp_path / 'report.json'
    main(['--input', str(input_path), '--output', str(output)])
    assert json.loads(output.read_text())['schema_version'] == 1
    assert 'raw' in capsys.readouterr().out


def test_csv_schema_rejected(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('label,prediction\n0,0.5\n')
    with pytest.raises(ValueError, match='columns'):
        read_predictions(path)


def test_auc_absent_for_one_class_diagnostics():
    result = diagnostics([0] * 8, [0.2] * 8)
    assert result['roc_auc'] is None
