"""M05D synthetic runner tests; no trained-model claims."""
import json

import pytest

from sponsored_ads.preranking_experiment import (
    ExperimentConfig, build_synthetic_cases, main, run_experiment,
)


def test_deterministic_cases_and_labels():
    config = ExperimentConfig(seed=4, cases=3, candidates_per_case=5, missing_rate=0.2)
    a, _, _ = build_synthetic_cases(config)
    b, _, _ = build_synthetic_cases(config)
    assert a == b
    assert all(sum(case.relevance.values()) == 1 for case in a)
    assert all(set(case.relevance) == {c.adgroup_id for c in case.candidates} for case in a)


def test_quality_reproducible_but_timings_not_asserted():
    config = ExperimentConfig(seed=8, cases=4, candidates_per_case=5, top_k=2, repeats=2)
    first = run_experiment(config)
    second = run_experiment(config)
    assert first['quality_and_coverage'] == second['quality_and_coverage']
    assert first['case_details'] == second['case_details']
    assert first['experiment'] == 'M05D_synthetic_infrastructure_only'
    assert first['timing_ms']['inference_ms']['p95_nearest_rank'] >= 0


def test_all_missing_features_keep_positive_in_denominator():
    result = run_experiment(ExperimentConfig(cases=3, candidates_per_case=4,
                                              top_k=2, missing_rate=1, repeats=1))
    quality = result['quality_and_coverage']
    assert quality['candidates'] == 12 and quality['scorable'] == 0
    assert quality['coverage'] == 0 and quality['hit_at_k'] == 0
    assert quality['mrr_at_k'] == 0


def test_no_missing_features():
    result = run_experiment(ExperimentConfig(cases=2, candidates_per_case=4,
                                              top_k=4, missing_rate=0, repeats=1))
    quality = result['quality_and_coverage']
    assert quality['coverage'] == 1
    assert quality['hit_at_k'] == 1


@pytest.mark.parametrize('config', [
    ExperimentConfig(cases=0), ExperimentConfig(candidates_per_case=0),
    ExperimentConfig(top_k=0), ExperimentConfig(top_k=11),
    ExperimentConfig(repeats=0), ExperimentConfig(seed=-1),
    ExperimentConfig(missing_rate=-0.1), ExperimentConfig(missing_rate=1.1),
    ExperimentConfig(missing_rate=float('nan')),
])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        config.validate()


def test_cli_writes_report(tmp_path):
    path = tmp_path / 'nested' / 'report.json'
    assert main(['--cases', '2', '--candidates-per-case', '3', '--top-k', '2',
                 '--repeats', '1', '--output', str(path)]) == 0
    report = json.loads(path.read_text())
    assert report['config']['cases'] == 2
    assert report['quality_and_coverage']['candidates'] == 6
    assert len(report['case_details']) == 2
