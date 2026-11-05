"""M06B-2 deterministic budget-aware single-slot auction admission.

Single-threaded synthetic simulation; not transactional or real billing.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .auction_pricing import AuctionOutcome, PricingPolicy, price_single_slot
from .auction_ranking import rank_eligible
from .budget_ledger import BudgetLedger, Reservation, cpc_to_cents, reserve_auction_winner
from .models import AdCandidate


@dataclass(frozen=True)
class BudgetAdmissionResult:
    outcome: AuctionOutcome | None
    reservation: Reservation | None
    excluded_ad_ids: tuple[str, ...]
    considered_candidates: int


def admit_budget_aware(
    ledger: BudgetLedger,
    candidates: Iterable[AdCandidate],
    *,
    policy: PricingPolicy,
    event_id: str,
    reservation_id: str,
    reserve_cpc: float = 0.0,
) -> BudgetAdmissionResult:
    """Filter unaffordable ads, reprice surviving auction, reserve winner bid.

    Admission requires enough *available* cents to hold the entire CPC bid,
    rounded upward. Below-reserve and ineligible ads are not budget-checked.
    Rejected candidates cannot set the second-price threshold. All admission
    validation is completed before the sole ledger mutation (reservation).
    A no-fill auction creates no ledger event.

    The caller must supply a fresh event and reservation ID for each attempt;
    orchestrator-level idempotency across changed inputs is not provided.
    """
    if not isinstance(ledger, BudgetLedger):
        raise TypeError('ledger must be BudgetLedger')
    if not isinstance(event_id, str) or not event_id:
        raise ValueError('event_id must be nonempty')
    if not isinstance(reservation_id, str) or not reservation_id:
        raise ValueError('reservation_id must be nonempty')
    # Reject reused identifiers even if no bidder is ultimately admitted.
    if any(event.event_id == event_id for event in ledger.events):
        raise ValueError('event_id already used')
    try:
        ledger.reservation(reservation_id)
    except KeyError:
        pass
    else:
        raise ValueError('reservation_id already used')

    all_candidates = tuple(candidates)
    # Price original auction first to validate policy/reserve and all candidates,
    # including duplicate IDs and overflow, without changing the ledger.
    price_single_slot(all_candidates, policy=policy, reserve_cpc=reserve_cpc)
    ranked = rank_eligible(all_candidates)
    by_id = {candidate.ad_id: candidate for candidate in all_candidates}
    excluded: list[str] = []
    affordable: list[AdCandidate] = []
    for item in ranked:
        if item.bid_cpc < reserve_cpc:
            continue
        required = cpc_to_cents(item.bid_cpc)
        if ledger.snapshot(item.advertiser_id).available_cents < required:
            excluded.append(item.ad_id)
        else:
            affordable.append(by_id[item.ad_id])
    if not affordable:
        return BudgetAdmissionResult(None, None, tuple(excluded), len(all_candidates))
    # Reprice after exclusion: unaffordable ads cannot set the clearing price.
    outcome = price_single_slot(affordable, policy=policy, reserve_cpc=reserve_cpc)
    assert outcome is not None
    reservation = reserve_auction_winner(
        ledger, outcome, event_id=event_id, reservation_id=reservation_id
    )
    return BudgetAdmissionResult(outcome, reservation, tuple(excluded), len(all_candidates))
