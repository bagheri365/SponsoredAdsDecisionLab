"""Deterministic allocation only; no auction payment or budget charging."""

from collections.abc import Iterable

from .models import AdCandidate, AdDecision, AdRequest


def choose_ad(
    request: AdRequest, candidates: Iterable[AdCandidate]
) -> AdDecision | None:
    """Select the highest bid-weighted CTR among eligible candidates.

    This is a *simulated* allocation rule. Tie-breaking uses ad_id for
    reproducibility. It does not calculate CPC payments.
    """
    candidates = tuple(candidates)
    ids = [candidate.ad_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ad_id values must be unique")

    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        return None

    winner = min(
        eligible,
        key=lambda candidate: (
            -(candidate.bid_cpc * candidate.predicted_ctr),
            candidate.ad_id,
        ),
    )
    return AdDecision(
        request_id=request.request_id,
        ad_id=winner.ad_id,
        advertiser_id=winner.advertiser_id,
        score=winner.bid_cpc * winner.predicted_ctr,
    )
