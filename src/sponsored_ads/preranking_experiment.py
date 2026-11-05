"""M05D deterministic *synthetic* demonstration of M05C evaluation.

This is an infrastructure smoke benchmark, NOT a trained CTR model, a
TaobaoAd experiment, or evidence of relevance / advertising lift.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .behavioral_retrieval import BehavioralIndex, ObservedAd, RetrievedAd
from .ctr_contract import make_contract
from .preranking_evaluation import EvaluationCase, evaluate_cases


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    cases: int = 20
    candidates_per_case: int = 10
    top_k: int = 3
    missing_rate: float = 0.1
    repeats: int = 3

    def validate(self) -> None:
        for name in ('seed', 'cases', 'candidates_per_case', 'top_k', 'repeats'):
            value = getattr(self, name)
            if type(value) is not int or (value < 0 if name == 'seed' else value <= 0):
                raise ValueError(f'{name} must be a {"nonnegative" if name == "seed" else "positive"} integer')
        if self.top_k > self.candidates_per_case:
            raise ValueError('top_k cannot exceed candidates_per_case')
        if not isinstance(self.missing_rate, (int, float)) or not math.isfinite(self.missing_rate) or not 0 <= self.missing_rate <= 1:
            raise ValueError('missing_rate must be finite and within [0, 1]')


class SyntheticPriceModel:
    """Fixed deterministic scoring function, NOT fitted/calibrated LightGBM."""

    def predict(self, matrix):
        return np.clip(np.asarray(matrix)[:, -1] / 100.0, 0.0, 1.0)


def build_synthetic_cases(config: ExperimentConfig):
    """Create explicit, fully labeled groups with a seeded random positive.

    The positive is selected independently of price / model score. Missing
    price fields are introduced independently to test coverage accounting.
    """
    config.validate()
    rng = random.Random(config.seed)
    ads = []
    cases = []
    ad_map = {}
    for case_idx in range(config.cases):
        candidates = []
        rows = {}
        ids = []
        for j in range(config.candidates_per_case):
            ad_id = f'synthetic_{case_idx:05d}_{j:04d}'
            ids.append(ad_id)
            ad_map[ad_id] = len(ad_map)
            ads.append(ObservedAd(ad_id, 'synthetic_category', 'synthetic_brand', j + 1))
            candidates.append(RetrievedAd(ad_id, float(j + 1), ('synthetic',)))
            price = rng.randint(1, 99)
            row = {'userid': f'synthetic_user_{case_idx:05d}',
                   'adgroup_id': ad_id, 'cate_id': 'synthetic_category',
                   'brand': 'synthetic_brand', 'price': str(price)}
            if rng.random() < config.missing_rate:
                del row['price']
            rows[ad_id] = row
        positive_id = rng.choice(ids)
        cases.append(EvaluationCase(tuple(candidates), rows,
                                    {ad_id: int(ad_id == positive_id) for ad_id in ids}))
    contract = make_contract(
        ('userid', 'adgroup_id', 'cate_id', 'brand', 'price'),
        {'userid': {f'synthetic_user_{i:05d}': i for i in range(config.cases)},
         'adgroup_id': ad_map,
         'cate_id': {'synthetic_category': 0},
         'brand': {'synthetic_brand': 0}},
    )
    return tuple(cases), BehavioralIndex(ads), contract


def _quantiles(values):
    ordered = sorted(values)
    return {'mean': statistics.mean(ordered),
            'p50': statistics.median(ordered),
            'p95_nearest_rank': ordered[math.ceil(0.95 * len(ordered)) - 1]}


def run_experiment(config: ExperimentConfig) -> dict:
    """Run the exact M05C evaluation API; retain raw per-case measurements."""
    cases, index, contract = build_synthetic_cases(config)
    model = SyntheticPriceModel()
    # Warm up code paths; excluded from recorded timings.
    evaluate_cases(cases[:1], index, model, contract, top_k=config.top_k)
    runs = []
    for _ in range(config.repeats):
        per_case, summary = evaluate_cases(cases, index, model, contract,
                                           top_k=config.top_k, missing_policy='skip')
        runs.append((per_case, summary))
    reference = runs[0][1]
    for _, summary in runs[1:]:
        for key in ('cases', 'candidates', 'scorable', 'unscorable',
                    'hit_at_k', 'mrr_at_k'):
            if getattr(summary, key) != getattr(reference, key):
                raise RuntimeError(f'nondeterministic quality/coverage metric: {key}')
    timings = {
        phase: _quantiles([getattr(case, phase) for per_case, _ in runs for case in per_case])
        for phase in ('assembly_ms', 'inference_ms', 'sorting_ms')
    }
    first_run = runs[0][0]
    return {
        'schema_version': 1,
        'experiment': 'M05D_synthetic_infrastructure_only',
        'limitations': [
            'Synthetic candidates, features, and random labels; no TaobaoAd observations',
            'Fixed price-based scoring stub; not a trained or calibrated CTR model',
            'No eligible inventory, counterfactuals, or online effectiveness evidence',
            'Timings exclude retrieval, artifact loading, I/O, and network latency',
        ],
        'config': asdict(config),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'quality_and_coverage': {
            'cases': reference.cases, 'candidates': reference.candidates,
            'scorable': reference.scorable, 'unscorable': reference.unscorable,
            'coverage': reference.scorable / reference.candidates,
            'hit_at_k': reference.hit_at_k, 'mrr_at_k': reference.mrr_at_k,
        },
        'timing_ms': timings,
        'case_details': [
            {'case_id': i, 'positive_ids': sorted(k for k, v in case.relevance.items() if v),
             'unscorable_ids': list(result.unscorable),
             'ranked_ids': [candidate.adgroup_id for candidate in result.ranked],
             'hit_at_k': result.hit_at_k,
             'reciprocal_rank_at_k': result.reciprocal_rank_at_k}
            for i, (case, result) in enumerate(zip(cases, first_run))
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--cases', type=int, default=20)
    parser.add_argument('--candidates-per-case', type=int, default=10)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--missing-rate', type=float, default=0.1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    config = ExperimentConfig(args.seed, args.cases, args.candidates_per_case,
                              args.top_k, args.missing_rate, args.repeats)
    result = run_experiment(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), **result['quality_and_coverage']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
