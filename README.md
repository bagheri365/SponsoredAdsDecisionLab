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
