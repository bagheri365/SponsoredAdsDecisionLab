"""Small, immutable contracts for the first decision-engine vertical slice."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AdRequest:
    request_id: str
    shopper_id: str

    def __post_init__(self) -> None:
        if not self.request_id or not self.shopper_id:
            raise ValueError("request_id and shopper_id must be nonempty")


@dataclass(frozen=True)
class AdCandidate:
    ad_id: str
    advertiser_id: str
    predicted_ctr: float
    bid_cpc: float
    eligible: bool = True

    def __post_init__(self) -> None:
        if not self.ad_id or not self.advertiser_id:
            raise ValueError("ad_id and advertiser_id must be nonempty")
        if not isfinite(self.predicted_ctr) or not 0 <= self.predicted_ctr <= 1:
            raise ValueError("predicted_ctr must be finite and between 0 and 1")
        if not isfinite(self.bid_cpc) or self.bid_cpc < 0:
            raise ValueError("bid_cpc must be finite and nonnegative")


@dataclass(frozen=True)
class AdDecision:
    request_id: str
    ad_id: str
    advertiser_id: str
    score: float
