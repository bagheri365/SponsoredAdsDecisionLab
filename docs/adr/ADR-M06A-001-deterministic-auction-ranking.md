# ADR-M06A-001: Reuse M01 allocation semantics for deterministic auction ranking

Status: Accepted.

## Context

M01 `choose_ad` already chooses the highest `bid_cpc * predicted_ctr` among
eligible ads, resolving exact score ties by lexical `ad_id`. It allows zero
bids, rejects duplicate ad IDs even for ineligible candidates, and makes no
payment or budget decision. M05B `RankedCandidate` carries CTR and retrieval
metadata, but **does not carry advertiser identity or a bid**.

## Decision

Introduce `rank_eligible(Iterable[AdCandidate]) -> tuple[AuctionRank, ...]`.
It returns the complete eligible ordering with score and input fields. Make
`choose_ad` delegate to this ranking function and select the first result.
Preserve M01 validation and tie-breaking semantics, including zero bids.
Require an explicit upstream construction of `AdCandidate` with real or
scenario-supplied advertiser IDs and bids. No synthetic bids are inferred
from retrieval scores, catalog data, or model predictions.

`rank_eligible` does not accept a reserve-price argument. Reserve eligibility,
first-price/second-price rules, payments, and budget charging are separate
M06A increments requiring their own policy contracts and tests.

## Invariants and limitations

- All candidate IDs must be unique, including ineligible candidates.
- Only eligible candidates appear in the ranked output.
- Descending `bid_cpc * predicted_ctr`; exact ties by ascending `ad_id`.
- Permutation-independent results for a fixed candidate set.
- The returned ranking is immutable and contains no auction payment.
- Scores are modeled eCPM-like value per impression in CPC currency units,
  not an actual realized revenue, charged CPC, or unbiased welfare estimate.
- CTR values depend on an explicitly chosen upstream raw/calibrated policy.
- The TaobaoAd dataset does not establish a real auction inventory, bids,
  counterfactual outcomes, or payment rules. Auction examples are simulations.

## Consequences

The allocation rule has one implementation shared by M01 and M06A. The
ranking contract is suitable for later explicit reserve and pricing policies,
but no economic conclusions can be drawn from ranking alone.
