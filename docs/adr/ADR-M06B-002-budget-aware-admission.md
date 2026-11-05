# ADR-M06B-002: Budget-aware deterministic auction admission

Status: Accepted.

## Context

M06A ranks eligible CPC candidates and computes synthetic conditional CPC. M06B-1 holds a maximum CPC bid in integer cents before serving and settles only on an explicitly simulated click. A top-ranked winner may not have sufficient *available* funds to reserve its full bid.

## Decision

Introduce `admit_budget_aware(ledger, candidates, *, policy, event_id, reservation_id, reserve_cpc=0)` as a single-threaded, in-memory simulation. Validate the full auction first, then check each ranked, eligible, above-reserve candidate against the advertiser's currently available budget using `cpc_to_cents(bid_cpc)` (ceiling). Exclude unaffordable candidates, **recompute the auction price on the reduced candidate set**, and reserve the admitted winner's full maximum CPC bid. Do not let excluded candidates set the second-price threshold. Return a frozen result with outcome, reservation, excluded IDs in ranking order, and input count; return a no-fill result without recording an event when none are affordable.

Only an insufficient available balance excludes an otherwise eligible candidate. Unknown advertisers, invalid bids/scores, malformed policies/reserves, duplicate IDs, and reused event/reservation identifiers are errors, not silently skipped. Event IDs and reservation IDs must be fresh for each admission attempt; the underlying ledger has its own exact-retry semantics, but orchestration-level retries with changed auction inputs are **not** supported. No ledger mutation occurs until final reservation.

## Invariants and limitations

- Winner must be eligible, meet the CPC reserve, and have sufficient funds for its entire rounded-up CPC bid.
- All admitted candidates participate in repricing; excluded candidates do not affect the clearing threshold.
- The ledger never spends on allocation alone; a click/no-click outcome must be finalized separately.
- Deterministic for a fixed input set, policy, and ledger state, independent of candidate order.
- Zero bids and CTRs retain M01/M06A semantics; ceiling rounding is the M06B-1 simulation policy.
- A zero-bid reservation is allowed, even with zero available funds.
- This is not atomic across concurrent writers, does not model pacing or real billing, and does not reconstruct actual Taobao/Amazon auctions.
