"""Run M03B offline CTR benchmarks."""
import argparse
from .ctr_baselines import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default='artifacts/m03a')
    parser.add_argument('--output-dir', default='artifacts/m03b')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--n-estimators', type=int, default=250)
    args = parser.parse_args()
    report = run(args.input_dir, args.output_dir, seed=args.seed,
                 n_estimators=args.n_estimators)
    for name, result in report['models'].items():
        m = result['metrics']
        print(f"{name}: logloss={m['log_loss']:.5f}, AUC={m['roc_auc']:.5f}, PR-AUC={m['pr_auc']:.5f}")
    print(f"Metrics: {args.output_dir}/metrics.json")


if __name__ == '__main__':
    main()
