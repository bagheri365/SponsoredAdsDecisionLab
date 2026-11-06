# ADR-M07A-001: Synchronous auction decision service contract

Status: Accepted.

## Context

M06A and M06B provide deterministic ranking, conditional CPC pricing, budget-aware admission and a reservation ledger, but not a single request-to-decision interface. Inputs to the auction are already scored `AdCandidate` objects; the trained CTR model is **not** yet connected to the auction path.

## Decision

Add `decide(AdRequest, Iterable[AdCandidate], BudgetLedger, *, policy, event_id, reservation_id, reserve_cpc=0)` returning an immutable `DecisionResponse` with `request_id`, `status`, `no_fill_reason`, `outcome` and `DecisionTrace`. The trace records input and qualifying counts, budget-excluded ad IDs, pricing policy, reserve, and reservation identifiers only when a fill occurs. No-fill reasons are `no_candidates`, `no_eligible_or_above_reserve`, and `insufficient_budget`.

Use M06B-2 for validation, admission, repricing, and reservation; do not duplicate accounting logic. Invalid inputs and unknown advertisers raise instead of returning no-fill. A successful fill creates a **reservation**, not settled spend. No-fill creates no ledger event. Callers must supply fresh event and reservation IDs; retries are not orchestrator-idempotent. The implementation is deterministic for fixed input and ledger state and is single-threaded.

## Limitations

The service does not perform retrieval, feature materialization, model inference, click generation, settlement, network I/O, persistence, concurrency control, pacing, or real billing. Trace fields are diagnostic, not a distributed trace or a production audit log. Later M07 increments will connect the versioned CTR model contract and add integration tests. No claims about Amazon Ads production internals.
