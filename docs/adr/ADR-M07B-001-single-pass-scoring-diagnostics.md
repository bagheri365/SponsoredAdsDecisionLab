# ADR-M07B-001: Single-pass scored-candidate diagnostics

Status: Accepted (locally verified).

## Context

M07A-3 calls CTR inference once for report diagnostics and again inside
`decide_with_model`. This is redundant and could diverge if inference is
nondeterministic. M07A-2 already owns the batch scoring boundary.

## Decision

Append `scored_candidates: tuple[AdCandidate, ...] = ()` to the frozen
`ModelDecisionResponse` dataclass, preserving existing positional construction
and the existing `decide_with_model` signature. Populate the tuple from the
single inference batch before passing it to the existing auction service.
Retain input order and all candidates, including those marked ineligible.
The smoke runner obtains `predicted_ctr` from this response and never invokes
its own CTR inference. No change to ranking, pricing, budget accounting,
calibration validation, or report schema is intended.

## Verification gates

- Focused and full tests pass, including an inference call-count regression.
- Empty requests do not invoke inference or create ledger events.
- Inference failures do not mutate the ledger.
- Two new raw persisted-model smoke reports have identical complete-file hashes.
- One new sigmoid smoke report completes successfully.
- Compare the new raw/sigmoid embedded report hashes and decision fields to
  the M07A-3 baseline before accepting this ADR.

## Limitations and follow-up

The response exposes scored `AdCandidate` objects (including ineligible
candidates); callers must treat diagnostic scores as potentially sensitive.
There is no persistent replay input envelope or deterministic state replay in
this increment; those belong to M07B-2. The smoke still uses synthetic bids,
advertisers, and budgets and provides no evidence of online lift.
