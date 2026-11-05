"""M06B-3: fixed synthetic auction sequence with ledger replay verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .budget_admission import admit_budget_aware
from .budget_ledger import BudgetLedger, finalize_auction_winner
from .models import AdCandidate

VERSION = 1
INITIAL_BUDGETS = {'adv-a': 250, 'adv-b': 350, 'adv-c': 0}
POLICIES = ('first_price', 'score_second_price')


@dataclass(frozen=True)
class AuctionStep:
    name: str
    candidates: tuple[AdCandidate, ...]
    clicked: bool
    reserve_cpc: float = 0.0


def steps() -> tuple[AuctionStep, ...]:
    """Fixed inputs and click labels; not sampled or observed traffic."""
    return (
        AuctionStep('initial_click', (AdCandidate('a', 'adv-a', .2, 2.),
                                      AdCandidate('b', 'adv-b', .1, 3.)), True),
        AuctionStep('budget_exclusion', (AdCandidate('a', 'adv-a', .2, 2.),
                                        AdCandidate('b', 'adv-b', .1, 3.)), False),
        AuctionStep('reserve_no_fill', (AdCandidate('c', 'adv-c', .9, 2.),), False, 1.),
        AuctionStep('zero_bid', (AdCandidate('free', 'adv-c', .1, 0.),), True),
        AuctionStep('below_reserve', (AdCandidate('low', 'adv-c', .2, .5),), False, 1.),
        AuctionStep('final_click', (AdCandidate('b', 'adv-b', .1, 1.),), True),
    )


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
            + '\n').encode('utf-8')


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def run_experiment(
    sequence: tuple[AuctionStep, ...] | None = None,
    *,
    initial_budgets: dict[str, int] | None = None,
) -> dict:
    if sequence is None:
        sequence = steps()
    if initial_budgets is None:
        initial_budgets = dict(INITIAL_BUDGETS)
    if not isinstance(sequence, tuple) or not sequence:
        raise ValueError('sequence must be a nonempty tuple')
    if any(not isinstance(s, AuctionStep) or not isinstance(s.name, str) or not s.name
           or type(s.clicked) is not bool for s in sequence):
        raise ValueError('invalid auction step')
    names = [s.name for s in sequence]
    if len(set(names)) != len(names):
        raise ValueError('step names must be unique')
    # Validate budgets before recording input fingerprint.
    BudgetLedger(initial_budgets)
    inputs = json.loads(json.dumps(
        {'version': VERSION, 'budgets_cents': initial_budgets,
         'steps': [asdict(s) for s in sequence], 'policies': POLICIES},
        allow_nan=False))
    policy_reports = []
    for policy in POLICIES:
        ledger = BudgetLedger(initial_budgets)
        rows = []
        for index, step in enumerate(sequence):
            prefix = f'{policy}:{index}'
            result = admit_budget_aware(
                ledger, step.candidates, policy=policy, reserve_cpc=step.reserve_cpc,
                event_id=f'{prefix}:reserve', reservation_id=f'{prefix}:reservation',
            )
            outcome = result.outcome
            if outcome is None:
                assert result.reservation is None
                rows.append({'step': step.name, 'clicked_input': step.clicked,
                             'winner_ad_id': None, 'advertiser_id': None,
                             'excluded_ad_ids': list(result.excluded_ad_ids),
                             'conditional_cpc': None, 'reserved_cents': 0,
                             'settled_cents': 0, 'released_cents': 0,
                             'fill': False})
                continue
            assert result.reservation is not None
            held = result.reservation.max_charge_cents
            final = finalize_auction_winner(
                ledger, outcome, event_id=f'{prefix}:finalize',
                reservation_id=f'{prefix}:reservation', clicked=step.clicked,
            )
            rows.append({'step': step.name, 'clicked_input': step.clicked,
                         'winner_ad_id': outcome.ad_id,
                         'advertiser_id': outcome.advertiser_id,
                         'excluded_ad_ids': list(result.excluded_ad_ids),
                         'conditional_cpc': outcome.charged_cpc_if_clicked,
                         'reserved_cents': held,
                         'settled_cents': final.settled_cents,
                         'released_cents': held - final.settled_cents,
                         'fill': True})
        snapshots = {key: asdict(ledger.snapshot(key)) for key in sorted(initial_budgets)}
        events = [asdict(event) for event in ledger.events]
        replay = BudgetLedger.replay(initial_budgets, ledger.events)
        replay_snapshots = {key: asdict(replay.snapshot(key)) for key in sorted(initial_budgets)}
        if snapshots != replay_snapshots or replay.events != ledger.events:
            raise AssertionError('ledger replay mismatch')
        for snapshot in snapshots.values():
            if (snapshot['budget_cents'] != snapshot['available_cents']
                    + snapshot['reserved_cents'] + snapshot['spent_cents']):
                raise AssertionError('budget conservation violated')
        if any(snapshot['reserved_cents'] for snapshot in snapshots.values()):
            raise AssertionError('experiment left outstanding reservations')
        policy_reports.append({'policy': policy, 'auctions': rows,
                               'filled': sum(row['fill'] for row in rows),
                               'no_fill': sum(not row['fill'] for row in rows),
                               'total_settled_cents': sum(s['spent_cents'] for s in snapshots.values()),
                               'snapshots': snapshots, 'events': events,
                               'event_count': len(events), 'replay_verified': True,
                               'accounting_verified': True})
    report = {'schema_version': VERSION, 'synthetic': True,
              'input_sha256': sha256(inputs), 'inputs': inputs, 'policies': policy_reports,
              'limitations': ('Synthetic bids, budgets, and click flags; not observed revenue, '
                              'causal lift, platform billing, pacing, or concurrent accounting.')}
    report['report_sha256'] = sha256(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report))
    print(f"input_sha256={report['input_sha256']}")
    print(f"report_sha256={report['report_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
