"""M08-1 deterministic synthetic decision-quality evaluation; no inference or billing."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path

from .auction_pricing import PricingPolicy
from .budget_ledger import BudgetLedger, cpc_to_cents
from .decision_service import decide
from .models import AdCandidate, AdRequest

VERSION = 1
POLICIES: tuple[PricingPolicy, ...] = ('first_price', 'score_second_price')


@dataclass(frozen=True)
class EvaluationScenario:
    name: str
    candidates: tuple[AdCandidate, ...]
    relevance: dict[str, float]
    budgets_cents: dict[str, int]
    reserve_cpc: float = 0.0


def scenarios() -> tuple[EvaluationScenario, ...]:
    """Fixed synthetic cases, including disagreement and all no-fill reasons."""
    return (
        EvaluationScenario('score_vs_relevance', (
            AdCandidate('a', 'adv-a', .20, 2.0),
            AdCandidate('b', 'adv-b', .10, 3.0)),
            {'a': .4, 'b': .9}, {'adv-a': 500, 'adv-b': 500}),
        EvaluationScenario('budget_exclusion', (
            AdCandidate('a', 'adv-a', .20, 2.0),
            AdCandidate('b', 'adv-b', .10, 3.0)),
            {'a': .9, 'b': .5}, {'adv-a': 199, 'adv-b': 500}),
        EvaluationScenario('tie_break', (
            AdCandidate('z', 'adv-z', .10, 2.0),
            AdCandidate('a', 'adv-a', .20, 1.0)),
            {'z': .5, 'a': .5}, {'adv-z': 200, 'adv-a': 100}),
        EvaluationScenario('no_candidates', (), {}, {}),
        EvaluationScenario('below_reserve', (
            AdCandidate('a', 'adv-a', .20, 1.0),),
            {'a': 1.0}, {'adv-a': 500}, 2.0),
        EvaluationScenario('insufficient_budget', (
            AdCandidate('a', 'adv-a', .20, 2.0),),
            {'a': 1.0}, {'adv-a': 0}),
    )


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _validate(items: tuple[EvaluationScenario, ...]) -> None:
    if not isinstance(items, tuple) or not items:
        raise ValueError('scenarios must be a nonempty tuple')
    names = set()
    for item in items:
        if not isinstance(item, EvaluationScenario) or not isinstance(item.name, str) or not item.name or item.name in names:
            raise ValueError('scenario names must be unique nonempty strings')
        names.add(item.name)
        if not isinstance(item.candidates, tuple) or not all(isinstance(c, AdCandidate) for c in item.candidates):
            raise ValueError('candidates must be a tuple of AdCandidate')
        ids = [c.ad_id for c in item.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate candidate ad_id')
        if not isinstance(item.relevance, dict) or set(item.relevance) != set(ids):
            raise ValueError('relevance labels must cover exactly the candidate IDs')
        for label in item.relevance.values():
            if isinstance(label, bool) or not isinstance(label, (int, float)) or not 0 <= label <= 1:
                raise ValueError('relevance must be finite in [0, 1]')
            # json canonicalization also rejects NaN/Infinity, but fail early.
            if not isfinite(label):
                raise ValueError('relevance must be finite')
        BudgetLedger(item.budgets_cents)
        if any(c.eligible and c.bid_cpc >= item.reserve_cpc and c.advertiser_id not in item.budgets_cents for c in item.candidates):
            raise ValueError('eligible candidate has no advertiser budget')
        if not isinstance(item.reserve_cpc, (int, float)) or isinstance(item.reserve_cpc, bool):
            raise ValueError('invalid reserve')
        if not isfinite(item.reserve_cpc) or item.reserve_cpc < 0:
            raise ValueError('invalid reserve')


def run_evaluation(items: tuple[EvaluationScenario, ...] | None = None) -> dict:
    """Compare policy decisions on independent fresh ledgers per scenario."""
    if items is None:
        items = scenarios()
    _validate(items)
    # Normalize tuple-valued candidate collections to JSON arrays so the
    # in-memory report matches the persisted canonical JSON representation.
    inputs = json.loads(json.dumps(
        {'schema_version': VERSION, 'scenarios': [asdict(item) for item in items],
         'policies': list(POLICIES)}, allow_nan=False))
    policy_reports = []
    for policy in POLICIES:
        rows = []
        for index, item in enumerate(items):
            ledger = BudgetLedger(item.budgets_cents)
            response = decide(
                AdRequest(f'm08-{index}', 'synthetic-shopper'), item.candidates, ledger,
                policy=policy, reserve_cpc=item.reserve_cpc,
                event_id=f'm08-{policy}-{index}-reserve',
                reservation_id=f'm08-{policy}-{index}-reservation')
            outcome = response.outcome
            # Evaluation oracle is the highest-labeled eligible/above-reserve,
            # budget-affordable ad. Tie-break by lexical ad_id, independently of score.
            affordable = [c for c in item.candidates if c.eligible and
                          c.bid_cpc >= item.reserve_cpc and
                          item.budgets_cents[c.advertiser_id] >= cpc_to_cents(c.bid_cpc)]
            oracle = min(affordable, key=lambda c: (-item.relevance[c.ad_id], c.ad_id)).ad_id if affordable else None
            snapshots = [asdict(ledger.snapshot(advertiser)) for advertiser in sorted(item.budgets_cents)]
            assert all(s['budget_cents'] == s['reserved_cents'] + s['spent_cents'] + s['available_cents'] for s in snapshots)
            assert all(s['spent_cents'] == 0 for s in snapshots)
            assert len(ledger.events) == (1 if outcome is not None else 0)
            rows.append({
                'scenario': item.name, 'status': response.status,
                'no_fill_reason': response.no_fill_reason,
                'winner_ad_id': outcome.ad_id if outcome else None,
                'synthetic_relevance_oracle_ad_id': oracle,
                'relevance_agreement': outcome.ad_id == oracle if outcome else None,
                'winning_score': outcome.score if outcome else None,
                'conditional_cpc_if_clicked': outcome.charged_cpc_if_clicked if outcome else None,
                'excluded_ad_ids': list(response.trace.excluded_ad_ids),
                'reserved_cents': sum(s['reserved_cents'] for s in snapshots),
                'ledger_event_count': len(ledger.events), 'snapshots': snapshots,
            })
        filled = [r for r in rows if r['status'] == 'filled']
        total_budget = sum(sum(s['budget_cents'] for s in r['snapshots']) for r in rows)
        total_reserved = sum(r['reserved_cents'] for r in rows)
        no_fill = {reason: sum(r['no_fill_reason'] == reason for r in rows)
                   for reason in ('no_candidates', 'no_eligible_or_above_reserve', 'insufficient_budget')}
        policy_reports.append({
            'policy': policy, 'outcomes': rows,
            'metrics': {'scenario_count': len(rows), 'filled': len(filled),
                        'fill_rate': len(filled) / len(rows),
                        'relevance_agreement_on_filled': (sum(r['relevance_agreement'] for r in filled) / len(filled)) if filled else None,
                        'mean_winning_score': (sum(r['winning_score'] for r in filled) / len(filled)) if filled else None,
                        'mean_conditional_cpc_if_clicked': (sum(r['conditional_cpc_if_clicked'] for r in filled) / len(filled)) if filled else None,
                        'total_reserved_cents': total_reserved,
                        'total_initial_budget_cents': total_budget,
                        'budget_reserved_fraction': total_reserved / total_budget if total_budget else None,
                        'no_fill_by_reason': no_fill},
        })
    left, right = policy_reports
    allocation_agreement = all(a['winner_ad_id'] == b['winner_ad_id']
                               for a, b in zip(left['outcomes'], right['outcomes']))
    if not allocation_agreement:
        raise AssertionError('pricing policy changed allocation under identical inputs')
    report = {'schema_version': VERSION, 'experiment': 'M08-1', 'synthetic': True,
              'input_sha256': _hash(inputs), 'inputs': inputs,
              'policy_reports': policy_reports, 'allocation_agreement': allocation_agreement,
              'limitations': ('Fixed synthetic scores, bids, relevance labels, and budgets. '
                              'Independent ledgers per scenario and policy; reservations only. '
                              'Conditional CPC is not charged revenue. No observed clicks, '
                              'causal lift, online quality, pacing, or concurrency.')}
    report['report_sha256'] = _hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; refusing to overwrite')
    report = run_evaluation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(_canonical(report))
    print(json.dumps({'output': str(args.output), 'input_sha256': report['input_sha256'],
                      'report_sha256': report['report_sha256'],
                      'allocation_agreement': report['allocation_agreement']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
