# ADR-M07A-002: Complete-feature CTR inference to auction decision

Status: Accepted.

## Context

M07A-1 accepts externally scored `AdCandidate` inputs. M05F-C already provides contract-checked LightGBM loading and explicit raw/sigmoid/isotonic calibration. The observed catalog does not contain all M03A serving features.

## Decision

Add `load_model_auction_bundle` and `decide_with_model`. The loader delegates to `load_serving_bundle` for model feature-name/order checks and calibrator/model/contract hash compatibility. It records model, contract, and optional calibration-artifact SHA-256 plus calibration method and feature-contract version. Raw is the default; calibrated methods require a matching artifact. The service accepts `FeatureCandidate` objects with complete M03A feature mappings, advertiser and ad identifiers, bid CPC, and eligibility. It batches predictions using `predict_calibrated`, converts them to existing `AdCandidate` objects, then delegates auction admission, pricing, and reservation to M07A-1 `decide`.

Reject invalid candidate metadata and duplicate ad IDs before inference. Missing/invalid features, invalid probabilities, and prediction-count mismatches fail before ledger mutation. Empty input bypasses inference and produces the existing `no_candidates` no-fill response. No-fill does not reserve budget. A fill reserves the maximum bid but does not settle a click. Caller-supplied event and reservation IDs must be fresh; the integration does not add retry idempotency.

## Limitations

The loader verifies compatibility at load time; it does not guarantee immutable artifacts after loading. A manually constructed bundle is a test seam, not a verified production artifact. Provenance is diagnostic and does not attest that input features are fresh or derived from online retrieval. This increment does not implement feature materialization, retrieval, network serving, concurrency, event persistence, real billing, or observed auction lift. Inference may run on candidates that are subsequently found ineligible or unaffordable; this is a correctness-first design, not a latency optimization.
