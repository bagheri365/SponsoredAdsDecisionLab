"""Run a tiny deterministic example without downloading data."""

from .decision import choose_ad
from .models import AdCandidate, AdRequest


def main() -> None:
    request = AdRequest(request_id="demo-001", shopper_id="shopper-001")
    candidates = [
        AdCandidate("ad-a", "advertiser-a", predicted_ctr=0.10, bid_cpc=1.5),
        AdCandidate("ad-b", "advertiser-b", predicted_ctr=0.08, bid_cpc=2.0),
    ]
    print(choose_ad(request, candidates))


if __name__ == "__main__":
    main()
