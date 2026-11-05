# ADR-006 — Strict, explicit candidate feature rows for pre-ranking

**Status:** Accepted for M05B.

## Context

M04B's `RetrievedAd` has an ID, heuristic score and source labels. Its index
stores adgroup ID, category, brand and impression count. M05A's LightGBM
contract requires a complete ordered feature row, including fields absent
from M04A (e.g. price, campaign, pid, user attributes and history counts).

## Decision

Keep retrieval and CTR feature preparation separate. Require the caller to
supply complete feature rows keyed by retrieved adgroup ID. Check IDs and known
catalog attributes; use M05A's contract for transformations. Default to a hard
error on missing features, with explicit skip-and-report as an opt-in. Preserve
retrieval source attribution; rank deterministically by predicted CTR and ID.

## Consequences

This enables contract-correct, synthetic end-to-end testing without pretending
the historical catalog contains request-time features. It is not yet a full
candidate feature assembler, a policy evaluator, or a serving-latency claim.
Old LightGBM boosters without saved train-fitted mappings remain unusable for
contract-correct inference.
