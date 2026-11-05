"""M06A-3 synthetic auction experiment invariants."""
import json

import pytest

from sponsored_ads.auction_experiment import Scenario, main, run_experiment, scenarios
from sponsored_ads.models import AdCandidate


def _rows(report, name):
    return [r for r in report['outcomes'] if r['scenario'] == name]


def test_deterministic_replay():
    assert run_experiment() == run_experiment()


def test_declared_inputs_and_policy_count():
    report = run_experiment()
    assert report['scenarios'] == len(scenarios()) == 8
    assert len(report['outcomes']) == 16
    assert report['allocation_agreement'] is True
    assert len(report['input_sha256']) == 64


def test_two_bidders_prices():
    first, second = _rows(run_experiment(), 'two_bidders')
    assert first['winner_ad_id'] == second['winner_ad_id'] == 'a'
    assert first['conditional_cpc'] == 2.0
    assert second['conditional_cpc'] == pytest.approx(1.5)
    assert first['expected_revenue_proxy_per_impression'] == pytest.approx(0.4)
    assert second['expected_revenue_proxy_per_impression'] == pytest.approx(0.3)


def test_reserve_changes_admission_not_policy_allocation():
    report = run_experiment()
    assert all(r['winner_ad_id'] == 'a' for r in _rows(report, 'reserve_zero'))
    assert all(r['winner_ad_id'] == 'b' for r in _rows(report, 'reserve_2_5'))
    assert _rows(report, 'reserve_2_5')[1]['conditional_cpc'] == 2.5


def test_tie_break():
    assert all(r['winner_ad_id'] == 'a' for r in _rows(run_experiment(), 'exact_tie'))


def test_sole_bidder_reserve_price():
    first, second = _rows(run_experiment(), 'sole_bidder')
    assert first['conditional_cpc'] == 1.5
    assert second['conditional_cpc'] == 0.5


def test_zero_ctr_proxy_is_zero():
    for row in _rows(run_experiment(), 'zero_ctr'):
        assert row['winner_ad_id'] == 'a'
        assert row['expected_revenue_proxy_per_impression'] == 0.0


def test_no_admitted_bidder():
    for row in _rows(run_experiment(), 'no_admitted_bidder'):
        assert row['winner_ad_id'] is None
        assert row['conditional_cpc'] is None
        assert row['expected_revenue_proxy_per_impression'] is None


def test_ineligible_high_score_excluded():
    assert all(r['winner_ad_id'] == 'b' for r in _rows(run_experiment(), 'ineligible_high_score'))


def test_permutation_invariance():
    ads = (AdCandidate('z', 'adv-z', 0.2, 1), AdCandidate('a', 'adv-a', 0.1, 3))
    one = run_experiment((Scenario('s', 0.0, ads),))
    two = run_experiment((Scenario('s', 0.0, tuple(reversed(ads))),))
    assert one['outcomes'] == two['outcomes']
    assert one['input_sha256'] != two['input_sha256']  # provenance records order


def test_invalid_scenario_names():
    s = Scenario('s', 0, ())
    with pytest.raises(ValueError, match='unique'):
        run_experiment((s, s))
    with pytest.raises(ValueError, match='nonempty'):
        run_experiment(())


def test_invalid_candidates_fail_closed():
    with pytest.raises(ValueError, match='unique'):
        run_experiment((Scenario('dup', 0, (
            AdCandidate('a', 'x', .1, 1), AdCandidate('a', 'y', .2, 2))),))


def test_cli_writes_replayable_json(tmp_path):
    path = tmp_path / 'nested' / 'report.json'
    assert main(['--output', str(path)]) == 0
    first = path.read_bytes()
    assert json.loads(first) == run_experiment()
    assert main(['--output', str(path)]) == 0
    assert path.read_bytes() == first


def test_first_price_not_below_second_price_in_declared_scenarios():
    report = run_experiment()
    for first, second in zip(report['outcomes'][::2], report['outcomes'][1::2]):
        if first['conditional_cpc'] is not None:
            assert first['conditional_cpc'] >= second['conditional_cpc']


def test_all_reported_values_within_explicit_bounds():
    report = run_experiment()
    for row in report['outcomes']:
        if row['conditional_cpc'] is not None:
            assert row['conditional_cpc'] >= row['reserve_cpc']
            assert row['expected_revenue_proxy_per_impression'] >= 0
