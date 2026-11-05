"""M04B adapter for the M04A observed-ad JSON catalog.

This is a retrospective simulator index, not a point-in-time eligible inventory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sponsored_ads.behavioral_retrieval import BehavioralIndex, ObservedAd

REQUIRED = frozenset({"adgroup_id", "customer", "cate_id", "brand", "impressions", "clicks"})


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def catalog_record_to_ad(record: dict[str, Any]) -> ObservedAd:
    """Validate a catalog record and adapt it to the behavioral index contract."""
    if not isinstance(record, dict) or not REQUIRED.issubset(record):
        raise ValueError("catalog record is missing required fields")
    for field in ("impressions", "clicks"):
        value = record[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    if record["clicks"] > record["impressions"]:
        raise ValueError("clicks cannot exceed impressions")
    _identifier(record["customer"], "customer")
    return ObservedAd(
        adgroup_id=_identifier(record["adgroup_id"], "adgroup_id"),
        cate_id=_identifier(record["cate_id"], "cate_id"),
        brand=_identifier(record["brand"], "brand"),
        impressions=record["impressions"],
    )


def load_observed_catalog(path: str | Path) -> BehavioralIndex:
    """Load M04A JSON and build the in-memory behavioral inverted index.

    Peak memory includes the parsed JSON and the index. Benchmark on the target
    machine before considering larger catalogs or production serving.
    """
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("M04A catalog must be a JSON list")
    return BehavioralIndex(catalog_record_to_ad(record) for record in records)
