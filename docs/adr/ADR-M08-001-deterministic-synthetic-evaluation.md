# ADR-M08-001 — Deterministic synthetic decision-quality evaluation

Status: Proposed; pending local verification.

## Context

M06A-3 compares pricing without budget admission. M06B-3 exercises synthetic
click finalization and ledger replay. M07B verifies persisted-model decision
reproducibility. M08-1 needs an independent, fixed-input evaluation of decision
quality and reservation behavior, without conflating synthetic relevance with
observed click labels.

## Decision

Add `sponsored_ads.m08_synthetic_evaluation` as a standalone, inference-free
experiment. Reuse `decision_service.decide`, `BudgetLedger`, and existing pricing
policies. For each scenario and policy, create a fresh ledger, so cross-policy
and cross-scenario budgets do not contaminate comparisons. Do not settle or
release reservations. Preserve the winner's bid-based reservation semantics.

Declare synthetic relevance scores separately from predicted CTR and bids.
The relevance oracle is the highest-label *eligible, above-reserve,
budget-affordable* candidate, with lexical ad ID tie-breaking. This is an
assumed synthetic reference, not ground truth. Compare oracle agreement only
on filled decisions. Empty filled denominators produce `null`.

Report fill rate, no-fill reasons, relevance agreement, winning score,
conditional CPC if clicked, and reserved/initial budget ratio. All budget
ratios aggregate across independent scenario ledgers; they do not represent a
shared spending pool. Input and report fingerprints use sorted-key compact
UTF-8 JSON plus a newline. The CLI refuses to overwrite an existing report.

## Consequences and limits

Pricing policies should allocate the same winner under identical synthetic
inputs, although conditional CPC can differ. Relevance agreement is an
assumption-sensitive diagnostic, not a ranking-learning metric, causal effect,
or production ad quality. No real click, revenue, pacing, concurrent ledger,
or billing claims are made. Later milestones may add shared-budget sequences,
statistical uncertainty, and evaluation with appropriately sourced labels.
