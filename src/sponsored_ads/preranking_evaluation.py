"""M05C labeled-candidate evaluation and phase-level profiling.

Candidate groups and relevance labels must be explicitly supplied. Observed
exposures are not counterfactual labels for unexposed ads.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .behavioral_retrieval import BehavioralIndex, RetrievedAd
from .candidate_preranking import assemble_candidate_rows, RankedCandidate
from .ctr_contract import predict_lightgbm


@dataclass(frozen=True)
class EvaluationCase:
    """One labeled candidate set; labels cover *every* retrieved candidate."""

    candidates: tuple[RetrievedAd, ...]
    supplied_rows: Mapping[str, Mapping[str, str]]
    relevance: Mapping[str, int]


@dataclass(frozen=True)
class CaseMetrics:
    candidate_count: int
    scorable_count: int
    unscorable: tuple[str, ...]
    positive_count: int
    hit_at_k: float
    reciprocal_rank_at_k: float
    assembly_ms: float
    inference_ms: float
    sorting_ms: float
    ranked: tuple[RankedCandidate, ...]


@dataclass(frozen=True)
class EvaluationSummary:
    cases: int
    candidates: int
    scorable: int
    unscorable: int
    cases_with_positive: int
    hit_at_k: float
    mrr_at_k: float
    mean_assembly_ms: float
    mean_inference_ms: float
    mean_sorting_ms: float


def _validate_labels(case: EvaluationCase) -> int:
    ids = [candidate.adgroup_id for candidate in case.candidates]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate candidate IDs')
    if set(case.relevance) != set(ids):
        raise ValueError('relevance labels must cover exactly the candidate IDs')
    if any(type(value) is not int or value not in (0, 1)
           for value in case.relevance.values()):
        raise ValueError('relevance labels must be binary integers')
    positives = sum(case.relevance.values())
    if positives == 0:
        raise ValueError('at least one labeled positive is required per case')
    return positives


def evaluate_case(
    case: EvaluationCase,
    index: BehavioralIndex,
    model,
    contract: dict,
    *,
    top_k: int,
    missing_policy: str = 'skip',
) -> CaseMetrics:
    """Evaluate labeled ranking; unscorable positives remain in the denominator.

    Timings are wall-clock observations, not deterministic assertions or online
    service latency. Model warmup and I/O are not included.
    """
    if type(top_k) is not int or top_k <= 0:
        raise ValueError('top_k must be a positive integer')
    positives = _validate_labels(case)
    t0 = time.perf_counter()
    selected, rows, unscorable = assemble_candidate_rows(
        case.candidates, index, case.supplied_rows, contract,
        missing_policy=missing_policy,
    )
    t1 = time.perf_counter()
    if selected:
        scores = np.asarray(predict_lightgbm(model, contract, rows), dtype=float)
        if scores.shape != (len(selected),) or not np.isfinite(scores).all():
            raise ValueError('CTR model returned invalid score shape or nonfinite scores')
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError('CTR scores must be probabilities in [0, 1]')
    else:
        scores = np.empty(0, dtype=float)
    t2 = time.perf_counter()
    ranked = tuple(sorted(
        (RankedCandidate(candidate.adgroup_id, float(score), candidate.score, candidate.sources)
         for candidate, score in zip(selected, scores)),
        key=lambda item: (-item.predicted_ctr, item.adgroup_id),
    )[:top_k])
    t3 = time.perf_counter()
    first_positive_rank = next(
        (rank for rank, candidate in enumerate(ranked, start=1)
         if case.relevance[candidate.adgroup_id] == 1), None,
    )
    return CaseMetrics(
        candidate_count=len(case.candidates), scorable_count=len(selected),
        unscorable=unscorable, positive_count=positives,
        hit_at_k=float(first_positive_rank is not None),
        reciprocal_rank_at_k=(1.0 / first_positive_rank if first_positive_rank else 0.0),
        assembly_ms=(t1-t0)*1000, inference_ms=(t2-t1)*1000,
        sorting_ms=(t3-t2)*1000, ranked=ranked,
    )


def summarize_cases(results: Sequence[CaseMetrics]) -> EvaluationSummary:
    """Macro-average Hit@K/MRR; candidate coverage is micro-averaged."""
    if not results:
        raise ValueError('at least one evaluated case is required')
    if any(r.candidate_count <= 0 or r.positive_count <= 0 or
           r.scorable_count < 0 or r.scorable_count > r.candidate_count or
           len(r.unscorable) != r.candidate_count-r.scorable_count
           for r in results):
        raise ValueError('invalid case counts')
    n = len(results)
    return EvaluationSummary(
        cases=n, candidates=sum(r.candidate_count for r in results),
        scorable=sum(r.scorable_count for r in results),
        unscorable=sum(len(r.unscorable) for r in results),
        cases_with_positive=n,
        hit_at_k=sum(r.hit_at_k for r in results)/n,
        mrr_at_k=sum(r.reciprocal_rank_at_k for r in results)/n,
        mean_assembly_ms=sum(r.assembly_ms for r in results)/n,
        mean_inference_ms=sum(r.inference_ms for r in results)/n,
        mean_sorting_ms=sum(r.sorting_ms for r in results)/n,
    )


def evaluate_cases(
    cases: Sequence[EvaluationCase], index: BehavioralIndex, model,
    contract: dict, *, top_k: int, missing_policy: str = 'skip',
) -> tuple[tuple[CaseMetrics, ...], EvaluationSummary]:
    """Evaluate explicit cases in input order, with no implicit negative sampling."""
    results = tuple(evaluate_case(case, index, model, contract, top_k=top_k,
                                  missing_policy=missing_policy) for case in cases)
    return results, summarize_cases(results)
