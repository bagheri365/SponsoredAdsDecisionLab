"""Build and smoke-test a reconstructed Taobao popularity catalog."""
from __future__ import annotations

import argparse
from pathlib import Path

from .retrieval import RetrievalQuery, build_catalog, retrieve_popular, save_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/TaobaoAd_x1/train.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/m04a'))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    ads, manifest = build_catalog(args.source, limit=args.limit)
    save_catalog(ads, manifest, args.output_dir)
    print(f"rows={manifest['rows_scanned']} ads={manifest['catalog_ads']} clicks={manifest['clicks_scanned']}")
    for ad in retrieve_popular(ads, RetrievalQuery(), k=args.top_k):
        print(f'adgroup_id={ad.adgroup_id} impressions={ad.popularity}')
    print(f'Catalog: {args.output_dir / "catalog.json"}')


if __name__ == '__main__':
    main()
