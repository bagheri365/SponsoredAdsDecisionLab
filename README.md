# SponsoredAdsDecisionLab

An incremental sponsored-product advertising system. Real-data retrieval and CTR
modeling will use Taobao advertising data in later milestones; bids, budgets,
auction outcomes, and counterfactual responses are **simulated**.

## Milestone 1: deterministic vertical slice

The first implementation provides typed request/ad/decision contracts and a
single-slot, bid-weighted CTR allocation rule. It deliberately uses supplied
CTR scores and synthetic bids, **not** a trained model or real auction data.

### Setup (Python 3.10+)

```bash
python -m pip install -e ".[dev]"
pytest
python -m sponsored_ads
```

### Decision contract

- One eligible ad is selected by `bid_cpc * predicted_ctr`.
- Invalid probabilities, negative bids, and duplicate ad IDs are rejected.
- No eligible ads returns `None`.
- The returned `score` is a ranking score, **not** revenue or a charge.
- Pricing, budgets, pacing, logging, and trained retrieval/ranking are future milestones.

This module is a deterministic interface smoke test, not a production auction.

## M03A: reproducible CTR development data

See [M03A methodology](docs/milestones/M03A.md), [dataset card](docs/data/taobao-ad-x1.md), and [development split ADR](docs/decisions/ADR-001-development-split.md).

```bash
python -m sponsored_ads.ctr_features_cli --limit 10000 --sample-rate 0.2
python -m sponsored_ads.ctr_features_cli --sample-rate 0.05
```

These commands read **only** `data/TaobaoAd_x1/train.csv`. Outputs under `artifacts/m03a/` are ignored by git. Published `test.csv` is reserved for later final evaluation. Source timestamps and point-in-time correctness are unverified.

## Milestone 3B: CTR model benchmarks

See [`docs/milestones/M03B.md`](docs/milestones/M03B.md) for experimental design, reproducibility, and limitations. After M03A feature preparation, install `python -m pip install -e '.[dev,ml]'` and run `python -m sponsored_ads.ctr_train_cli`. Model outputs are written under ignored `artifacts/m03b/`. The published test split is not used for development.
