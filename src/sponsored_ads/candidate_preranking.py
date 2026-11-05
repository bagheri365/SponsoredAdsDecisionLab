"""M05B strict bridge from retrieved IDs to complete M03A feature rows.

The M04A catalog is NOT a source of missing campaign, price, or shopper data.
Callers must explicitly supply a complete feature row for every scored candidate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .behavioral_retrieval import BehavioralIndex, RetrievedAd
from .ctr_contract import predict_lightgbm, validate_contract


@dataclass(frozen=True)
class RankedCandidate:
    adgroup_id: str
    predicted_ctr: float
    retrieval_score: float
    sources: tuple[str, ...]


@dataclass(frozen=True)
class PrerankResult:
    ranked: tuple[RankedCandidate, ...]
    unscorable: tuple[str, ...]
    input_candidates: int


def assemble_candidate_rows(
    candidates: Sequence[RetrievedAd],
    index: BehavioralIndex,
    supplied_rows: Mapping[str, Mapping[str, str]],
    contract: dict,
    *,
    missing_policy: str = 'error',
) -> tuple[list[RetrievedAd], list[dict[str, str]], tuple[str, ...]]:
    """Validate candidate identity and known catalog fields; never invent features.

    `missing_policy='skip'` reports candidates without complete supplied rows.
    Contradictory catalog fields or duplicate candidates always fail closed.
    """
    spec = validate_contract(contract)
    if missing_policy not in ('error', 'skip'):
        raise ValueError('missing_policy must be error or skip')
    seen: set[str] = set()
    selected: list[RetrievedAd] = []
    rows: list[dict[str, str]] = []
    unscorable: list[str] = []
    for candidate in candidates:
        ad_id = candidate.adgroup_id
        if not isinstance(ad_id, str) or not ad_id or ad_id in seen:
            raise ValueError(f'invalid or duplicate candidate ID: {ad_id!r}')
        seen.add(ad_id)
        if not math.isfinite(candidate.score):
            raise ValueError(f'nonfinite retrieval score: {ad_id}')
        ad = index.ads.get(ad_id)
        if ad is None:
            raise ValueError(f'candidate absent from retrieval index: {ad_id}')
        supplied = supplied_rows.get(ad_id)
        if supplied is None:
            if missing_policy == 'error':
                raise ValueError(f'missing feature row for candidate: {ad_id}')
            unscorable.append(ad_id)
            continue
        if not isinstance(supplied, Mapping):
            raise ValueError(f'invalid feature row for candidate: {ad_id}')
        missing = [name for name in spec['features'] if name not in supplied]
        if missing:
            if missing_policy == 'error':
                raise ValueError(f'incomplete feature row for {ad_id}: {missing}')
            unscorable.append(ad_id)
            continue
        if supplied.get('adgroup_id') != ad_id:
            raise ValueError(f'candidate ID mismatch: {ad_id}')
        for field in ('cate_id', 'brand'):
            if field in supplied and supplied[field] != getattr(ad, field):
                raise ValueError(f'catalog {field} mismatch: {ad_id}')
        selected.append(candidate)
        rows.append(dict(supplied))
    return selected, rows, tuple(unscorable)


def prerank_candidates(
    candidates: Sequence[RetrievedAd],
    index: BehavioralIndex,
    supplied_rows: Mapping[str, Mapping[str, str]],
    model,
    contract: dict,
    *,
    top_k: int,
    missing_policy: str = 'error',
) -> PrerankResult:
    """Score complete candidate rows and sort by CTR, then lexical ad ID.

    The model must have been loaded with its matching M05A contract; prefer
    `load_lightgbm_bundle` for artifact loading. No online relevance claim.
    """
    if type(top_k) is not int or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer')
    selected, rows, unscorable = assemble_candidate_rows(
        candidates, index, supplied_rows, contract, missing_policy=missing_policy
    )
    if not selected:
        return PrerankResult((), unscorable, len(candidates))
    scores = np.asarray(predict_lightgbm(model, contract, rows), dtype=float)
    if scores.shape != (len(selected),) or not np.isfinite(scores).all():
        raise ValueError('CTR model returned invalid score shape or nonfinite scores')
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError('CTR scores must be probabilities in [0, 1]')
    ranked = sorted(
        (RankedCandidate(c.adgroup_id, float(score), c.score, c.sources)
         for c, score in zip(selected, scores)),
        key=lambda item: (-item.predicted_ctr, item.adgroup_id),
    )
    return PrerankResult(tuple(ranked[:top_k]), unscorable, len(candidates))
