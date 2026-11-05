"""Reconstructed Taobao ad catalog and deterministic popularity retrieval.

Counts reflect historical *shown impressions*, not complete ad eligibility.
No online/past-only or counterfactual retrieval claims are supported.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .taobao import iter_rows


@dataclass(frozen=True)
class CatalogAd:
    adgroup_id: str
    customer: str
    cate_id: str
    brand: str
    impressions: int
    clicks: int

    def __post_init__(self) -> None:
        if not all((self.adgroup_id, self.customer, self.cate_id, self.brand)):
            raise ValueError('catalog identifiers must be nonempty')
        if self.impressions <= 0 or not 0 <= self.clicks <= self.impressions:
            raise ValueError('invalid impression/click counts')


@dataclass(frozen=True)
class RetrievalQuery:
    category: str | None = None
    brand: str | None = None
    excluded_ad_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RetrievedAd:
    adgroup_id: str
    customer: str
    cate_id: str
    brand: str
    popularity: int


def build_catalog(source: str | Path, *, limit: int | None = None) -> tuple[list[CatalogAd], dict]:
    """Aggregate impression popularity from the published training file only.

    Reject conflicting metadata instead of silently picking one version.
    In production metadata would come from a separately versioned ad catalog.
    """
    source = Path(source)
    if source.name != 'train.csv':
        raise ValueError('catalog source must be published train.csv')
    counts: dict[str, list] = {}
    rows = clicks = 0
    for row in iter_rows(source, limit=limit):
        rows += 1
        click = int(row['clk'])
        clicks += click
        ad_id = row['adgroup_id']
        metadata = (row['customer'], row['cate_id'], row['brand'])
        if ad_id not in counts:
            counts[ad_id] = [metadata, 0, 0]
        record = counts[ad_id]
        if record[0] != metadata:
            raise ValueError(f'conflicting metadata for adgroup_id {ad_id}')
        record[1] += 1
        record[2] += click
    ads = [CatalogAd(ad_id, *meta, impressions, ad_clicks)
           for ad_id, (meta, impressions, ad_clicks) in counts.items()]
    ads.sort(key=lambda ad: ad.adgroup_id)
    manifest = {
        'source': str(source), 'source_split': 'published train only',
        'rows_scanned': rows, 'clicks_scanned': clicks, 'catalog_ads': len(ads),
        'row_limit': limit, 'score': 'historical impression count',
        'catalog_scope': 'ads observed in sampled/published training impressions only',
        'point_in_time_verified': False,
        'eligibility': 'NOT VERIFIED; query exclusions are explicit simulator controls',
        'warning': 'No test data used. Counts cannot be treated as online point-in-time features.',
    }
    return ads, manifest


def retrieve_popular(
    catalog: list[CatalogAd], query: RetrievalQuery, *, k: int = 20,
) -> list[RetrievedAd]:
    """Return at most k unique catalog ads, deterministically ordered.

    Category/brand are explicit filters, not inferred shopper intent.
    Caller-supplied exclusions are not a substitute for real eligibility.
    """
    if k < 0:
        raise ValueError('k must be nonnegative')
    if len({ad.adgroup_id for ad in catalog}) != len(catalog):
        raise ValueError('duplicate catalog ad IDs')
    eligible = (ad for ad in catalog
                if ad.adgroup_id not in query.excluded_ad_ids
                and (query.category is None or ad.cate_id == query.category)
                and (query.brand is None or ad.brand == query.brand))
    ranked = sorted(eligible, key=lambda ad: (-ad.impressions, ad.adgroup_id))
    return [RetrievedAd(ad.adgroup_id, ad.customer, ad.cate_id, ad.brand, ad.impressions)
            for ad in ranked[:k]]


def save_catalog(ads: list[CatalogAd], manifest: dict, output: str | Path) -> None:
    """Write a JSON snapshot with no user-level records."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'catalog.json').write_text(
        json.dumps([asdict(ad) for ad in ads], indent=2) + '\n', encoding='utf-8')
    (output / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def load_catalog(path: str | Path) -> list[CatalogAd]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('catalog must be a JSON array')
    ads = [CatalogAd(**entry) for entry in data]
    if len({ad.adgroup_id for ad in ads}) != len(ads):
        raise ValueError('duplicate catalog ad IDs')
    return ads
