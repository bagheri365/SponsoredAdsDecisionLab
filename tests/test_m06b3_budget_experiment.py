"""M06B-3 synthetic accounting experiment regression tests."""
import json
from dataclasses import replace

import pytest

from sponsored_ads.budget_experiment import (
    AuctionStep, canonical_bytes, main, run_experiment, sha256, steps,
)
from sponsored_ads.models import AdCandidate


def test_repeatable_report():
    assert canonical_bytes(run_experiment()) == canonical_bytes(run_experiment())


def test_input_and_report_hashes():
    report = run_experiment()
    assert report['input_sha256'] == sha256(report['inputs'])
    checksum = report.pop('report_sha256')
    assert checksum == sha256(report)


def test_explicitly_synthetic():
    report = run_experiment()
    assert report['synthetic'] is True
    assert 'not observed revenue' in report['limitations']


@pytest.mark.parametrize('policy', ['first_price', 'score_second_price'])
def test_fill_and_no_fill(policy):
    p = next(x for x in run_experiment()['policies'] if x['policy'] == policy)
    assert (p['filled'], p['no_fill']) == (4, 2)
    assert p['event_count'] == 8
    assert p['replay_verified'] and p['accounting_verified']


@pytest.mark.parametrize('policy', ['first_price', 'score_second_price'])
def test_budget_exclusion_and_repricing(policy):
    p = next(x for x in run_experiment()['policies'] if x['policy'] == policy)
    row = p['auctions'][1]
    assert row['excluded_ad_ids'] == ['a']
    assert row['winner_ad_id'] == 'b'
    assert row['settled_cents'] == 0
    assert row['released_cents'] == 300
    assert row['conditional_cpc'] == (3.0 if policy == 'first_price' else 0.0)


def test_policy_settled_spend_differs_without_claiming_lift():
    first, second = run_experiment()['policies']
    assert first['total_settled_cents'] == 300
    assert second['total_settled_cents'] == 151


@pytest.mark.parametrize('policy', ['first_price', 'score_second_price'])
def test_no_fill_has_no_ledger_event(policy):
    p = next(x for x in run_experiment()['policies'] if x['policy'] == policy)
    for row in (p['auctions'][2], p['auctions'][4]):
        assert row['fill'] is False
        assert row['winner_ad_id'] is None
        assert row['reserved_cents'] == row['settled_cents'] == 0


def test_zero_budget_zero_bid_allowed():
    for p in run_experiment()['policies']:
        row = p['auctions'][3]
        assert row['winner_ad_id'] == 'free'
        assert row['reserved_cents'] == row['settled_cents'] == 0


def test_conservation_and_no_holds():
    for p in run_experiment()['policies']:
        for snap in p['snapshots'].values():
            assert snap['budget_cents'] == (snap['available_cents'] + snap['reserved_cents']
                                            + snap['spent_cents'])
            assert snap['reserved_cents'] == 0
            assert snap['available_cents'] >= 0


def test_custom_sequence_and_budgets():
    sequence = (AuctionStep('one', (AdCandidate('a', 'x', .5, 1.),), True),)
    report = run_experiment(sequence, initial_budgets={'x': 100})
    assert [p['total_settled_cents'] for p in report['policies']] == [100, 0]


@pytest.mark.parametrize('sequence', [(), [], 'bad'])
def test_invalid_sequence(sequence):
    with pytest.raises(ValueError):
        run_experiment(sequence)


def test_duplicate_names():
    s = steps()[0]
    with pytest.raises(ValueError, match='unique'):
        run_experiment((s, s))


def test_non_boolean_clicked():
    with pytest.raises(ValueError, match='invalid auction step'):
        run_experiment((replace(steps()[0], clicked=1),))


def test_invalid_budget():
    with pytest.raises(ValueError):
        run_experiment(initial_budgets={'adv-a': -1})


def test_invalid_candidate():
    with pytest.raises(ValueError):
        run_experiment((AuctionStep('unknown', (AdCandidate('a', 'missing', .1, 1.),), True),))


def test_cli(tmp_path, capsys):
    path = tmp_path / 'nested' / 'report.json'
    assert main(['--output', str(path)]) == 0
    report = json.loads(path.read_text())
    assert report == run_experiment()
    assert path.read_bytes() == canonical_bytes(report)
    assert 'report_sha256=' in capsys.readouterr().out
