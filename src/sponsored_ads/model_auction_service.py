"""M07A-2: validated CTR inference into the existing synchronous auction service.

Requires complete M03A feature rows; does not perform feature materialization.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from pathlib import Path
import math

from .auction_pricing import PricingPolicy
from .budget_ledger import BudgetLedger
from .decision_service import DecisionResponse, decide
from .m05fb_experiment import sha256
from .m05fc_serving import METHODS, load_serving_bundle, predict_calibrated
from .models import AdCandidate, AdRequest


@dataclass(frozen=True)
class FeatureCandidate:
    ad_id: str
    advertiser_id: str
    bid_cpc: float
    features: Mapping[str, object]
    eligible: bool = True

    def __post_init__(self):
        # Reuse the exact M01 metadata and bid validation, without inventing a CTR.
        AdCandidate(self.ad_id, self.advertiser_id, 0.0, self.bid_cpc, self.eligible)
        if not isinstance(self.features, Mapping):
            raise TypeError('features must be a mapping')


@dataclass(frozen=True)
class ModelProvenance:
    model_sha256: str
    contract_sha256: str
    calibration_method: str
    calibration_artifact_sha256: str | None
    feature_contract_version: int


@dataclass(frozen=True)
class ModelAuctionBundle:
    model: object
    contract: dict
    calibration_artifact: dict | None
    provenance: ModelProvenance


@dataclass(frozen=True)
class ModelDecisionResponse:
    decision: DecisionResponse
    provenance: ModelProvenance
    scored_candidate_count: int


def load_model_auction_bundle(model_path, contract_path, artifact_path=None, *, method='raw'):
    """Load through M05F-C, preserving its model/contract/calibrator checks."""
    if method not in METHODS:
        raise ValueError('Unsupported calibration method')
    model, contract, artifact = load_serving_bundle(
        model_path, contract_path, artifact_path, method=method
    )
    return ModelAuctionBundle(
        model=model, contract=contract, calibration_artifact=artifact,
        provenance=ModelProvenance(
            model_sha256=sha256(model_path),
            contract_sha256=sha256(contract_path),
            calibration_method=method,
            calibration_artifact_sha256=sha256(artifact_path) if artifact_path is not None else None,
            feature_contract_version=contract['version'],
        ),
    )


def decide_with_model(
    request: AdRequest,
    candidates: Iterable[FeatureCandidate],
    ledger: BudgetLedger,
    bundle: ModelAuctionBundle,
    *,
    policy: PricingPolicy,
    event_id: str,
    reservation_id: str,
    reserve_cpc: float = 0.0,
) -> ModelDecisionResponse:
    """Batch-score complete feature rows and delegate all auction mutations to M07A-1.

    A validation/inference failure happens before the auction call and cannot
    create a budget event. No-fill also creates no event.
    """
    if not isinstance(request, AdRequest):
        raise TypeError('request must be AdRequest')
    if not isinstance(ledger, BudgetLedger):
        raise TypeError('ledger must be BudgetLedger')
    if not isinstance(bundle, ModelAuctionBundle):
        raise TypeError('bundle must be ModelAuctionBundle')
    items = tuple(candidates)
    seen = set()
    for item in items:
        if not isinstance(item, FeatureCandidate):
            raise TypeError('candidates must contain FeatureCandidate objects')
        if item.ad_id in seen:
            raise ValueError('Duplicate ad_id')
        seen.add(item.ad_id)
    # M05A rejects empty batches; an empty auction is instead a valid no-fill.
    if items:
        predictions = predict_calibrated(
            bundle.model, bundle.contract, [item.features for item in items],
            bundle.calibration_artifact,
        )
        if len(predictions) != len(items):
            raise ValueError('Prediction count does not match candidate count')
        scored = tuple(
            AdCandidate(item.ad_id, item.advertiser_id, float(probability),
                        item.bid_cpc, item.eligible)
            for item, probability in zip(items, predictions)
        )
    else:
        scored = ()
    decision = decide(request, scored, ledger, policy=policy,
                      event_id=event_id, reservation_id=reservation_id,
                      reserve_cpc=reserve_cpc)
    return ModelDecisionResponse(decision, bundle.provenance, len(scored))
