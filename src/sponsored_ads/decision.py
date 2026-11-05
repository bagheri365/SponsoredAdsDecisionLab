"""Deterministic allocation only; no auction payment or budget charging."""

from collections.abc import Iterable

from .auction_ranking import rank_eligible
from .models import AdCandidate, AdDecision, AdRequest


def choose_ad(
    request: AdRequest, candidates: Iterable[AdCandidate]
) -> AdDecision | None:
    """Select the highest bid-weighted CTR among eligible candidates.

    This is a *simulated* allocation rule. Tie-breaking uses ad_id for
    reproducibility. It does not calculate CPC payments.
    """
    ranked = rank_eligible(candidates)
    if not ranked:
        return None
    winner = ranked[0]
    return AdDecision(
        request_id=request.request_id,
        ad_id=winner.ad_id,
        advertiser_id=winner.advertiser_id,
        score=winner.score,
    )
