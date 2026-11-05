"""M06A-2 synthetic single-slot CPC pricing, without charging or settlement.

The score-based second-price rule is a declared simulation, not a claim about
TaobaoAd's actual auction. Payments are conditional on a click.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from .auction_ranking import rank_eligible
from .models import AdCandidate

PricingPolicy = Literal['first_price', 'score_second_price']


@dataclass(frozen=True)
class AuctionOutcome:
    ad_id: str
    advertiser_id: str
    score: float
    bid_cpc: float
    predicted_ctr: float
    charged_cpc_if_clicked: float
    policy: PricingPolicy
    reserve_cpc: float
    runner_up_score: float | None


def price_single_slot(
    candidates: Iterable[AdCandidate],
    *,
    policy: PricingPolicy,
    reserve_cpc: float = 0.0,
) -> AuctionOutcome | None:
    """Allocate and compute a simulated conditional CPC for a single slot.

    Reserve is a minimum *bid* and a minimum conditional CPC, not a minimum
    bid-weighted score. Ineligible and below-reserve ads cannot win or set price.
    A single bidder in score-second-price pays the reserve. A zero-CTR winner
    can occur only when all remaining scores are zero; its price is the reserve.
    No clicks, money transfers, or budgets are simulated here.
    """
    if policy not in ('first_price', 'score_second_price'):
        raise ValueError('unsupported pricing policy')
    if isinstance(reserve_cpc, bool) or not isinstance(reserve_cpc, (int, float)):
        raise ValueError('reserve_cpc must be a finite nonnegative number')
    if not isfinite(reserve_cpc) or reserve_cpc < 0:
        raise ValueError('reserve_cpc must be a finite nonnegative number')

    # Rank before filtering to preserve duplicate-ID and overflow validation,
    # including for candidates whose bids are below the reserve.
    ranked = rank_eligible(candidates)
    admitted = [item for item in ranked if item.bid_cpc >= reserve_cpc]
    if not admitted:
        return None
    winner = admitted[0]
    runner_score = admitted[1].score if len(admitted) > 1 else None

    if policy == 'first_price':
        payment = winner.bid_cpc
    elif winner.predicted_ctr == 0.0:
        # A zero-CTR winner implies every admitted score is zero.
        payment = float(reserve_cpc)
    else:
        threshold = 0.0 if runner_score is None else runner_score / winner.predicted_ctr
        if not isfinite(threshold):
            raise ValueError('nonfinite second-price threshold')
        # Winner's score >= runner's score mathematically; rounding may make
        # threshold marginally exceed the winning bid. Cap to bid explicitly.
        payment = min(winner.bid_cpc, max(float(reserve_cpc), threshold))

    if not isfinite(payment) or not 0 <= payment <= winner.bid_cpc:
        raise ValueError('invalid conditional CPC payment')
    return AuctionOutcome(
        ad_id=winner.ad_id,
        advertiser_id=winner.advertiser_id,
        score=winner.score,
        bid_cpc=winner.bid_cpc,
        predicted_ctr=winner.predicted_ctr,
        charged_cpc_if_clicked=payment,
        policy=policy,
        reserve_cpc=float(reserve_cpc),
        runner_up_score=runner_score,
    )
