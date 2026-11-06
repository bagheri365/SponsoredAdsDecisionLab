# ADR-M07B-004: Offline verification of paired serving artifacts

Status: Accepted for local synthetic-artifact verification; pending commit.

## Context

M07B-3 produces a legacy M07A-3 serving report and an opt-in versioned
scored-decision replay sidecar. Manual checks are easy to omit or misapply.

## Decision

Add `python -m sponsored_ads.m07b4_verify` as a read-only verifier. It checks
report checksum and accepted embedded baseline, replay integrity and
reconstructed decision, known model provenance, and cross-file decision,
scored CTRs, bids, auction configuration, snapshots, and event counts.
It optionally checks complete-file SHA-256 values supplied by the caller.

The verifier does not load a model or run inference. Keep the existing report
and replay schemas unchanged. Failure produces a nonzero exit status.

## Baseline semantics

Both embedded report hashes are accepted M07A-3 baselines. The raw complete-file
report hash was independently matched against M07A-3. The sigmoid complete-file
hash is only an M07B-3 recorded reference until independently compared against
the original M07A-3 sigmoid file. The verifier exposes optional complete-file
hash arguments rather than silently treating either reference as authoritative.

## Limits

SHA-256 is not an authentication mechanism. Recomputed replay proves scored
auction consistency, not model inference reproducibility or feature lineage.
The verifier checks a particular synthetic scenario and known provenance,
not arbitrary serving reports. It does not verify the source model file bytes.
