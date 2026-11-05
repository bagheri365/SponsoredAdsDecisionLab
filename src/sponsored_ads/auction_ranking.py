"""M06A-1 deterministic allocation ranking; no payment or budget logic.

Inputs are fully specified auction candidates, not retrieval or model features.
The caller supplies advertiser identity, CPC bid, and chosen CTR policy.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

from .models import AdCandidate


@dataclass(frozen=True)
class AuctionRank:
    ad_id: str
    advertiser_id: str
    predicted_ctr: float
    bid_cpc: float
    score: float


def rank_eligible(candidates: Iterable[AdCandidate]) -> tuple[AuctionRank, ...]:
    """Rank eligible ads by descending bid*CTR, then ascending lexical ad_id.

    Duplicate IDs fail even when one candidate is ineligible. Scores are
    checked after multiplication to reject finite-input floating overflow.
    Zero bids and zero CTRs remain valid for M01 compatibility.
    """
    seen: set[str] = set()
    ranked: list[AuctionRank] = []
    for candidate in candidates:
        if not isinstance(candidate, AdCandidate):
            raise TypeError('candidates must be AdCandidate instances')
        if candidate.ad_id in seen:
            raise ValueError('candidate ad_id values must be unique')
        seen.add(candidate.ad_id)
        if not candidate.eligible:
            continue
        score = candidate.bid_cpc * candidate.predicted_ctr
        if not isfinite(score):
            raise ValueError('bid-weighted CTR score must be finite')
        ranked.append(AuctionRank(
            ad_id=candidate.ad_id,
            advertiser_id=candidate.advertiser_id,
            predicted_ctr=candidate.predicted_ctr,
            bid_cpc=candidate.bid_cpc,
            score=score,
        ))
    return tuple(sorted(ranked, key=lambda item: (-item.score, item.ad_id)))
