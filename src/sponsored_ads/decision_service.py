"""M07A-1 synchronous, deterministic request-to-auction decision contract.

Inputs are already scored AdCandidate objects; no model inference or real billing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from collections.abc import Iterable

from .auction_pricing import AuctionOutcome, PricingPolicy
from .budget_admission import admit_budget_aware
from .budget_ledger import BudgetLedger
from .models import AdCandidate, AdRequest

NoFillReason = Literal['no_candidates', 'no_eligible_or_above_reserve', 'insufficient_budget']


@dataclass(frozen=True)
class DecisionTrace:
    input_count: int
    eligible_above_reserve_count: int
    excluded_ad_ids: tuple[str, ...]
    policy: PricingPolicy
    reserve_cpc: float
    reservation_event_id: str | None
    reservation_id: str | None


@dataclass(frozen=True)
class DecisionResponse:
    request_id: str
    status: Literal['filled', 'no_fill']
    no_fill_reason: NoFillReason | None
    outcome: AuctionOutcome | None
    trace: DecisionTrace


def decide(
    request: AdRequest,
    candidates: Iterable[AdCandidate],
    ledger: BudgetLedger,
    *,
    policy: PricingPolicy,
    event_id: str,
    reservation_id: str,
    reserve_cpc: float = 0.0,
) -> DecisionResponse:
    """Validate and allocate one request; reserve the winner's bid if filled.

    Invalid inputs raise; no-fill is not an error. No-fill has no ledger event.
    Event and reservation identifiers must be fresh, even for retry attempts.
    No impression/click event or settlement is emitted here.
    """
    if not isinstance(request, AdRequest):
        raise TypeError('request must be AdRequest')
    if not isinstance(ledger, BudgetLedger):
        raise TypeError('ledger must be BudgetLedger')
    items = tuple(candidates)
    # Delegate complete validation and the only possible mutation to M06B-2.
    result = admit_budget_aware(
        ledger, items, policy=policy, event_id=event_id,
        reservation_id=reservation_id, reserve_cpc=reserve_cpc,
    )
    qualifying = sum(1 for item in items if item.eligible and item.bid_cpc >= reserve_cpc)
    filled = result.outcome is not None
    reason: NoFillReason | None = None
    if not filled:
        if not items:
            reason = 'no_candidates'
        elif qualifying == 0:
            reason = 'no_eligible_or_above_reserve'
        else:
            reason = 'insufficient_budget'
    trace = DecisionTrace(
        input_count=len(items),
        eligible_above_reserve_count=qualifying,
        excluded_ad_ids=result.excluded_ad_ids,
        policy=policy,
        reserve_cpc=float(reserve_cpc),
        reservation_event_id=event_id if filled else None,
        reservation_id=reservation_id if filled else None,
    )
    return DecisionResponse(
        request_id=request.request_id,
        status='filled' if filled else 'no_fill',
        no_fill_reason=reason,
        outcome=result.outcome,
        trace=trace,
    )
