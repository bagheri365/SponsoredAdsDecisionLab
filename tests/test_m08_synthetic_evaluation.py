"""M08-1 synthetic quality evaluation and accounting invariants."""
import json
from dataclasses import replace

import pytest

from sponsored_ads.m08_synthetic_evaluation import (
    EvaluationScenario, main, run_evaluation, scenarios,
)
from sponsored_ads.models import AdCandidate


def test_declared_scenarios_and_metrics():
    report = run_evaluation()
    assert report['experiment'] == 'M08-1'
    assert report['allocation_agreement'] is True
    assert len(report['policy_reports']) == 2
    for policy in report['policy_reports']:
        metrics = policy['metrics']
        assert metrics['scenario_count'] == 6
        assert metrics['filled'] == 3
        assert metrics['fill_rate'] == .5
        assert metrics['relevance_agreement_on_filled'] == pytest.approx(2 / 3)
        assert metrics['no_fill_by_reason'] == {
            'no_candidates': 1, 'no_eligible_or_above_reserve': 1,
            'insufficient_budget': 1}
        assert metrics['total_reserved_cents'] == 600
        assert metrics['budget_reserved_fraction'] == pytest.approx(600 / 2499)
        assert all(row['ledger_event_count'] == (row['status'] == 'filled')
                   for row in policy['outcomes'])


def test_price_differs_without_allocation_change():
    first, second = run_evaluation()['policy_reports']
    assert [r['winner_ad_id'] for r in first['outcomes']] == [r['winner_ad_id'] for r in second['outcomes']]
    assert first['outcomes'][0]['conditional_cpc_if_clicked'] == 2.0
    assert second['outcomes'][0]['conditional_cpc_if_clicked'] == pytest.approx(1.5)
    assert second['outcomes'][1]['conditional_cpc_if_clicked'] == 0.0
    assert first['outcomes'][0]['relevance_agreement'] is False
    assert first['outcomes'][1]['relevance_agreement'] is True


def test_repeatable_report_and_input_order_fingerprint():
    assert run_evaluation() == run_evaluation()
    original = scenarios()
    changed = (replace(original[0], candidates=tuple(reversed(original[0].candidates))),) + original[1:]
    assert run_evaluation(changed)['input_sha256'] != run_evaluation()['input_sha256']
    assert run_evaluation(changed)['policy_reports'] == run_evaluation()['policy_reports']


@pytest.mark.parametrize('items', [(), (scenarios()[0], scenarios()[0])])
def test_invalid_scenario_collection(items):
    with pytest.raises(ValueError):
        run_evaluation(items)


def test_invalid_relevance_labels():
    item = scenarios()[0]
    for labels in ({'a': .5}, {'a': float('nan'), 'b': .5}, {'a': True, 'b': .5}):
        with pytest.raises(ValueError, match='relevance'):
            run_evaluation((replace(item, relevance=labels),))


def test_unknown_advertiser_fails_closed():
    item = EvaluationScenario('unknown', (AdCandidate('a', 'missing', .2, 1),),
                              {'a': .2}, {})
    with pytest.raises(ValueError, match='budget'):
        run_evaluation((item,))


def test_zero_budget_and_zero_bid():
    item = EvaluationScenario('free', (AdCandidate('free', 'adv', .0, 0),),
                              {'free': 1.0}, {'adv': 0})
    report = run_evaluation((item,))
    for policy in report['policy_reports']:
        assert policy['metrics']['filled'] == 1
        assert policy['metrics']['total_reserved_cents'] == 0
        assert policy['metrics']['budget_reserved_fraction'] is None


def test_cli_canonical_and_no_overwrite(tmp_path):
    path = tmp_path / 'nested' / 'report.json'
    assert main(['--output', str(path)]) == 0
    assert json.loads(path.read_text()) == run_evaluation()
    original = path.read_bytes()
    with pytest.raises(SystemExit):
        main(['--output', str(path)])
    assert path.read_bytes() == original
