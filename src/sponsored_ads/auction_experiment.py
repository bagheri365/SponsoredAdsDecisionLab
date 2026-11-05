"""M06A-3: deterministic, explicitly synthetic auction comparison.

No observed bids, clicks, settlement, revenue, or causal lift are inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .auction_pricing import price_single_slot
from .models import AdCandidate

POLICIES = ('first_price', 'score_second_price')
VERSION = 1


@dataclass(frozen=True)
class Scenario:
    name: str
    reserve_cpc: float
    candidates: tuple[AdCandidate, ...]


def scenarios() -> tuple[Scenario, ...]:
    """Declared synthetic inputs; neither learned nor sampled from TaobaoAd."""
    return (
        Scenario('two_bidders', 0.0, (
            AdCandidate('a', 'adv-a', 0.20, 2.0),
            AdCandidate('b', 'adv-b', 0.10, 3.0),
        )),
        Scenario('reserve_zero', 0.0, (
            AdCandidate('a', 'adv-a', 0.20, 2.0),
            AdCandidate('b', 'adv-b', 0.10, 3.0),
        )),
        Scenario('reserve_2_5', 2.5, (
            AdCandidate('a', 'adv-a', 0.20, 2.0),
            AdCandidate('b', 'adv-b', 0.10, 3.0),
        )),
        Scenario('exact_tie', 0.0, (
            AdCandidate('z', 'adv-z', 0.10, 2.0),
            AdCandidate('a', 'adv-a', 0.20, 1.0),
        )),
        Scenario('sole_bidder', 0.5, (
            AdCandidate('solo', 'adv-s', 0.25, 1.5),
        )),
        Scenario('zero_ctr', 0.0, (
            AdCandidate('a', 'adv-a', 0.0, 2.0),
            AdCandidate('b', 'adv-b', 0.0, 1.0),
        )),
        Scenario('no_admitted_bidder', 3.0, (
            AdCandidate('a', 'adv-a', 0.5, 2.0),
        )),
        Scenario('ineligible_high_score', 0.0, (
            AdCandidate('a', 'adv-a', 0.9, 9.0, False),
            AdCandidate('b', 'adv-b', 0.1, 1.0),
        )),
    )


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
            + '\n').encode('utf-8')


def run_experiment(items: tuple[Scenario, ...] | None = None) -> dict:
    """Compare identical allocations under two synthetic CPC pricing rules."""
    if items is None:
        items = scenarios()
    if not isinstance(items, tuple) or not items:
        raise ValueError('scenarios must be a nonempty tuple')
    names: set[str] = set()
    rows: list[dict] = []
    input_rows: list[dict] = []
    for item in items:
        if not isinstance(item, Scenario) or not isinstance(item.name, str) or not item.name or item.name in names:
            raise ValueError('scenario names must be unique nonempty strings')
        names.add(item.name)
        input_rows.append(asdict(item))
        outcomes = {policy: price_single_slot(item.candidates, policy=policy,
                                              reserve_cpc=item.reserve_cpc)
                    for policy in POLICIES}
        first, second = (outcomes[p] for p in POLICIES)
        if (first is None) != (second is None) or (first is not None and first.ad_id != second.ad_id):
            raise AssertionError('pricing policy changed allocation')
        for policy, outcome in outcomes.items():
            if outcome is None:
                rows.append(dict(scenario=item.name, policy=policy,
                                 reserve_cpc=item.reserve_cpc, winner_ad_id=None,
                                 winning_score=None, conditional_cpc=None,
                                 expected_revenue_proxy_per_impression=None))
                continue
            cpc = outcome.charged_cpc_if_clicked
            if not (0 <= cpc <= outcome.bid_cpc and cpc >= item.reserve_cpc):
                raise AssertionError('conditional CPC violates bid/reserve bounds')
            rows.append(dict(scenario=item.name, policy=policy,
                             reserve_cpc=item.reserve_cpc, winner_ad_id=outcome.ad_id,
                             winning_score=outcome.score, conditional_cpc=cpc,
                             expected_revenue_proxy_per_impression=outcome.predicted_ctr * cpc))
    comparisons = []
    for i, item in enumerate(items):
        a, b = rows[2*i:2*i+2]
        comparisons.append(dict(scenario=item.name,
                                same_winner=a['winner_ad_id'] == b['winner_ad_id'],
                                conditional_cpc_difference=None if a['conditional_cpc'] is None else
                                a['conditional_cpc'] - b['conditional_cpc']))
    payload = dict(schema_version=VERSION, experiment='M06A-3',
                   input_sha256=hashlib.sha256(_canonical_bytes(input_rows)).hexdigest(),
                   scenarios=len(items), policies=list(POLICIES),
                   allocation_agreement=all(c['same_winner'] for c in comparisons),
                   outcomes=rows, comparisons=comparisons,
                   caveat='All bids, CTRs, reserves, and eligibility are synthetic. '
                          'Expected revenue proxy is predicted CTR times conditional CPC, '
                          'not observed revenue, settlement, or causal lift.')
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='JSON output path; existing files are overwritten')
    args = parser.parse_args(argv)
    report = run_experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))
    print(json.dumps(dict(output=str(args.output), scenarios=report['scenarios'],
                          allocation_agreement=report['allocation_agreement'],
                          input_sha256=report['input_sha256']), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
