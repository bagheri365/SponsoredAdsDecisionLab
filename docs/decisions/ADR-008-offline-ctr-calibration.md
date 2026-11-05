# ADR-008 — Separate calibration fitting from evaluation

**Status:** Accepted for M05E

## Decision

Fit sigmoid and isotonic calibrators only on a designated calibration partition of out-of-sample CTR predictions. Compare raw and calibrated predictions on a disjoint evaluation partition. Do not automatically promote the winner or serialize it from this exploratory evaluation. Require upstream provenance before reporting empirical model calibration.

## Rationale

Auction value calculations depend on probability scale, not just AUC. Reusing calibrator-fit rows for evaluation biases calibration metrics. Reusing a LightGBM early-stopping partition as the final test also risks optimistic model selection. A row-random partition cannot establish chronological or user-disjoint generalization.

## Alternatives and limitations

A temporal/user-disjoint holdout is preferable when trustworthy timestamps and identifiers permit it. The available Taobao split has no verified chronology or point-in-time history integrity. ECE depends on binning and should be accompanied by Brier/log loss and reliability counts. Isotonic can overfit small calibration sets. These results apply to logged impression probabilities, not counterfactual auction outcomes.
