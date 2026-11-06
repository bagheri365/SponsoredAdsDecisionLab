# SponsoredAdsDecisionLab

A reproducible, interview-oriented sponsored-products decision lab connecting
observed-data retrieval/CTR experimentation to a **simulated** single-slot ad
auction, budget admission, decision replay, and offline evaluation.

**Scope:** Taobao advertising data is used for documented offline retrieval and
CTR experiments. Auction bids, advertiser budgets, pricing outcomes, reservations,
and M08 evaluation scenarios are synthetic. This is not a production ad server,
online A/B test, or claim of measured advertising lift.

## Quick start (Python 3.10+)

```bash
python -m pip install -e '.[dev,ml]'
pytest -q
python -m sponsored_ads
python -m sponsored_ads.m08_synthetic_evaluation --output artifacts/m08/portfolio_demo.json
```

The final command writes a deterministic synthetic evaluation report and **refuses
to overwrite** an existing file. Use a fresh output filename for subsequent runs.
The basic smoke command and M08 evaluation do not require the Taobao CSV. The
ML extra is needed for the optional persisted-model demonstration; see the demo
guide for its artifact prerequisites.

## Architecture and design

Read [the architecture overview](docs/architecture.md) for component boundaries,
data contracts, and design tradeoffs. For an interview walkthrough, follow the
[approximately five-minute demo](docs/portfolio-demo.md).

| Stage | Implementation | Evidence |
| --- | --- | --- |
| Retrieval and CTR | Taobao-derived feature/retrieval experiments, CTR baselines and calibration | [Dataset card](docs/data/taobao-ad-x1.md), [CTR results](docs/experiments/M03B-ctr-baseline-results.md) |
| Ranking and pricing | Eligible single-slot ranking by bid × predicted CTR; first-price and score-second-price policies | [Auction ADRs](docs/adr/ADR-M06A-001-deterministic-auction-ranking.md) |
| Budget and decision | Budget-aware admission, reservations, decision traces | [Budget ADR](docs/adr/ADR-M06B-001-budget-ledger.md) |
| Model integration and replay | Persisted-model smoke, scored-decision replay and offline verification | [Replay verification](docs/milestones/M07B-4.md) |
| Synthetic evaluation | Six fixed scenarios, two pricing policies, canonical JSON report | [M08-1 results](docs/experiments/M08-1-results.md) |

## Latest verified local checkpoint

At M08-1 (`e952613`), the project owner reported **388 passing tests** with
one known scikit-learn/LightGBM feature-name warning. The fixed M08-1 synthetic
run filled 3/6 scenarios, agreed with its synthetic relevance oracle in 2/3
filled scenarios, and showed identical allocation across pricing policies.
These fixture metrics are **not** estimates of online fill rate or relevance.
See [the experiment record](docs/experiments/M08-1-results.md) for exact inputs,
metrics, checksums, and limitations. Fresh-clone verification for the final
release remains pending.

## Data and reproducibility

The Taobao data is not bundled as a prerequisite for the synthetic quick start.
Data-dependent CTR experiments require the local dataset and generated artifacts;
consult the [dataset card](docs/data/taobao-ad-x1.md) and
[M03A methodology](docs/milestones/M03A.md). The published test split was not
used for development in the documented CTR workflow. Source timestamps and
point-in-time correctness are unverified.

Generated outputs under `artifacts/` are local run evidence, not committed
production datasets. Report hashes describe a particular deterministic run;
reproduce them only with the corresponding inputs and code revision.

## Limitations and next steps

This is an offline, single-slot teaching/research system: it does not implement
production serving SLOs, concurrency-safe distributed reservations, pacing,
real auction integrations, measured clicks or conversions from its synthetic
auction, or causal lift estimation. The six M08 scenarios are deliberately small
and do not represent a traffic distribution. See [architecture tradeoffs](docs/architecture.md)
for discussion topics and possible future extensions.

## Historical milestones

The [milestone documents](docs/milestones/) and [experiment records](docs/experiments/)
retain the development history, including early interface-only smoke tests and
later retrieval/CTR, auction, budget, model-serving, and replay milestones.
