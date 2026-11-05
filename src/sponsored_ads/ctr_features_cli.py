"""CLI for creating the Milestone 3A development sample."""
from __future__ import annotations

import argparse
from pathlib import Path

from .ctr_features import build_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/TaobaoAd_x1/train.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/m03a'))
    parser.add_argument('--sample-rate', type=float, default=0.05)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--limit', type=int, default=None, help='first N rows, smoke tests only')
    args = parser.parse_args()
    manifest = build_sample(args.source, args.output_dir, sample_rate=args.sample_rate,
                            validation_fraction=args.validation_fraction, seed=args.seed,
                            limit=args.limit)
    for split, stats in manifest['counts'].items():
        print(f"{split}: rows={stats['rows']}, clicks={stats['clicks']}")
    print(f"Manifest: {args.output_dir / 'sample_manifest.json'}")


if __name__ == '__main__':
    main()
