# ADR-M06A-002: Explicit synthetic single-slot CPC pricing

Status: Accepted.

## Context

M06A-1 deterministically ranks eligible `AdCandidate` inputs by `bid_cpc * predicted_ctr` with lexical ad-ID ties. M01 `choose_ad` does not calculate payments. TaobaoAd does not provide the true auction bid/reserve/payment mechanism or counterfactual outcomes.

## Decision

Add `price_single_slot(candidates, *, policy, reserve_cpc=0)` without changing M01 or M06A-1. The caller must explicitly choose `first_price` or `score_second_price`. All inputs are simulated or explicitly supplied, never inferred from logged clicks or retrieval scores. The API returns an immutable `AuctionOutcome` or `None`; `charged_cpc_if_clicked` is **conditional CPC**, not a realized charge.

- Reserve is a finite, nonnegative CPC amount. It is a **bid floor** and **conditional CPC floor**, not an eCPM/score floor. Bids below the reserve are excluded from allocation and price setting. A zero bid remains valid when the reserve is zero.
- First-price: winning conditional CPC equals the winning CPC bid.
- Score-based second-price: conditional CPC equals `max(reserve_cpc, runner_up_score / winner_predicted_ctr)`, capped at the winning bid for floating-point rounding safety. A sole admitted bidder pays the reserve.
- If the winning CTR is zero, every admitted score must be zero. The winner is chosen by lexical ad ID and the second-price conditional CPC is the reserve. This is an explicit edge-case convention; no actual click is simulated.
- Exact score ties are resolved by lexical ad ID, and the second-price threshold may equal the winning bid.
- Invalid policies, invalid reserves, duplicate IDs, and nonfinite scores fail closed. The underlying ranking validates candidates before reserve filtering.

## Limitations and consequences

The score-based second-price rule is a **synthetic mechanism**. It is not asserted to be truthful bidding, a platform's generalized second-price mechanism, or a faithful reconstruction of Taobao auctions. In particular, CTRs are predictions, reserve is an imposed policy, and ad-ID tie-breaking is not an economic equilibrium rule. No budgets, pacing, multi-slot externalities, actual clicks, or settlement are modeled. Revenue, welfare, or lift claims require separate assumptions and evidence.

For score-positive winners, the threshold is based on runner-up score divided by winner CTR; it is not simply the runner-up CPC bid. A zero-CTR winner is possible only when every admitted score is zero, so the reserve-only payment is a documented convention rather than a division-by-zero fallback that implies a real auction charge.
