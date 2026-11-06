# ADR-M07B-003: Optional persisted-model replay sidecar

Status: Accepted for local synthetic serving integration (pending commit).

## Context

M07B-1 returns ordered scored candidates from the same inference call that
serves the auction. M07B-2 validates scored-decision replay without inference.
The M07A-3 report schema and hashes are regression baselines.

## Decision

Add an optional `replay_output_path` to `run_smoke` and `--replay-output` to
its CLI. Capture pre-decision ledger events before `decide_with_model`, then
pass the original request, already-scored candidates, provenance, budgets,
auction parameters, decision, and final ledger to `capture_replay_record`.
Validate a JSON round-trip before writing the sidecar. Do not alter the
existing report payload, hashing, or the number of inference calls.

Refuse existing, colliding, or input-overwriting output paths before model
loading. Keep sidecar generation opt-in to preserve existing consumers.

## Limitations

- Replay validates the scored auction; it does not re-run inference.
- Synthetic bids, advertisers, and budgets are not observed auctions.
- The SHA-256 checksum is not a signature or proof of provenance.
- The runner's two output files are not written atomically as a pair.
- Do not publish replay records containing real shopper identifiers.

## Verification plan

Test optional sidecar and unchanged report bytes for raw and sigmoid paths,
assert one inference per invocation, replay decision and ledger equivalence,
output path preflight, full suite, and real persisted-model raw/sigmoid runs.
Record actual observed hashes only after local execution.
