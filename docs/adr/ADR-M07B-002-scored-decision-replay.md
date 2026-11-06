# ADR-M07B-002: Versioned scored-decision replay

Status: Proposed.

## Context

M07B-1 exposes scored candidates in input order, while M06B provides an
in-memory budget ledger and event replay. Debugging an auction requires more
than ledger events: the candidate CTRs, order, policy, reserve, and pre-decision
budget state must be available.

## Decision

Add an independent version-1 scored-decision replay record. Capture explicitly
supplied request, ordered scored candidates, model provenance, initial budgets,
pre-existing ledger events, policy and reservation identifiers, expected
DecisionResponse, resulting events, and account snapshots. Serialize with
canonical JSON and SHA-256. Validate by reconstructing the ledger and calling
`decide`, without loading a model or performing inference.

The capture API checks the supplied original result against an independent
reconstruction before returning the record. Replay rejects unknown top-level
fields, unsupported versions, hash mismatches, provenance mismatches when an
expected provenance is supplied, and differences in decision/events/snapshots.

## Limits and tradeoffs

- A SHA-256 digest is an integrity checksum, **not** authentication; an actor
  can rewrite both content and digest. External trusted provenance is optional.
- Replay consumes recorded CTRs and **does not** prove model re-execution
  equivalence, feature reconstruction, latency improvement, or production lift.
- This is a single-threaded, synthetic ledger simulation, not billing or pacing.
- Callers must capture the complete pre-decision event history before deciding;
  recording only the post-decision ledger is insufficient.
- The record may contain shopper identifiers and scored features-derived data;
  avoid publishing real user data. Tests use synthetic identifiers only.

## Verification plan

Focused tests for filled/no-fill outcomes, pre-existing reservations, JSON
round-trip, tampering, rehashed expectation mismatch, provenance mismatch,
malformed records, and incorrect capture. Run the full project test suite and
record observed results before accepting this ADR.
