# ADR-M06B-001: Deterministic budget reservation and simulated CPC settlement

Status: Accepted.

## Context

M06A-2 returns a **conditional CPC if clicked**, not a realized charge. M06A-3 uses synthetic scenarios. No real auction payment or budget events are observed. A budget ledger must not equate allocation with spending.

## Decision

Introduce `BudgetLedger` as a single-threaded in-memory, event-sourced simulation. Budget, reservation, and spend use nonnegative **integer cents**. The ledger has `reserve`, `settle`, `release`, `snapshot`, and `replay` operations. Reservations are unique; event IDs are globally unique. Exact event retries are idempotent, conflicting retries fail closed, and finalized reservations cannot be finalized again. Failed operations do not append events or change balances.

Before serving an auction winner, reserve the **maximum CPC bid** in cents; if insufficient budget, reject reservation. This does **not** rerun the auction with the next bidder: budget-aware allocation and pacing are separate work. On an explicitly simulated click, settle the conditional CPC; on a simulated nonclick, release the hold. No implicit charge is made for an impression. A settlement must not exceed the reserved amount.

`cpc_to_cents` converts finite nonnegative numeric CPC values using `Decimal(str(value))`, rounded **up** to a cent. The same rule is applied to maximum bids and conditional CPCs. For example, the binary-float result `1.5000000000000002` becomes **151 cents**, not 150. This deliberately conservative convention may overstate simulated costs by a cent; it is not a real billing or currency policy. Integer cents are the authoritative ledger representation.

## Invariants

- `budget = available + reserved + settled spend` per advertiser.
- All balances are nonnegative; reserved and settled spend never exceed budget in aggregate.
- Event replay in recorded order reconstructs the same account snapshots.
- Only a click explicitly supplied by the simulation triggers settlement.
- A release returns all reserved funds; settlement releases the unused hold.
- Budget state is independent of CTR inference, ranking, and pricing-policy implementation.

## Scope exclusions

No concurrency, persistent transactions, event authentication, multi-currency support, refunds, daily resets, pacing, delayed click reconciliation, or real billing. `BudgetLedger` is not safe for concurrent writers. Replay assumes trusted event data and the original initial budgets. Simulated budget admission is not evidence of platform-specific billing behavior.
