"""CLI for streaming TaobaoAd_x1 dataset audits."""
import argparse
from .taobao import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit RecZoo TaobaoAd_x1 CSV files')
    parser.add_argument('--data-dir', default='data/TaobaoAd_x1')
    parser.add_argument('--output', default='artifacts/taobao_manifest.json')
    parser.add_argument('--limit', type=int, default=None,
                        help='Optional row limit per split; hashes still cover full files')
    args = parser.parse_args()
    manifest = write_manifest(args.data_dir, args.output, limit=args.limit)
    for name, stats in manifest['files'].items():
        print(f"{name}: rows={stats['rows']}, clicks={stats['clicks']}, sha256={stats['sha256'][:12]}...")
    print(f'Manifest: {args.output}')


if __name__ == '__main__':
    main()
