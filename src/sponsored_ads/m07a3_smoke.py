"""M07A-3: deterministic persisted-model/synthetic-auction integration smoke."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .budget_ledger import BudgetLedger
from .decision_replay import capture_replay_record, replay_decision
from .model_auction_service import FeatureCandidate, decide_with_model, load_model_auction_bundle
from .models import AdRequest


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def sample_features(path, features, count):
    """Use the first N CSV rows, preserving the contract's feature names."""
    if type(count) is not int or count < 2:
        raise ValueError('count must be an integer >= 2')
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('Missing or duplicate CSV headers')
        missing = set(features) - set(reader.fieldnames)
        if missing:
            raise ValueError(f'Missing CSV feature columns: {sorted(missing)}')
        rows = []
        for row in reader:
            if len(rows) == count:
                break
            if None in row or any(row[name] is None for name in features):
                raise ValueError('Malformed CSV row')
            rows.append({name: row[name] for name in features})
    if len(rows) != count:
        raise ValueError(f'Expected {count} rows, found {len(rows)}')
    return rows


def run_smoke(*, model_path, contract_path, features_path, output_path,
              calibration_path=None, method='raw', count=3, policy='score_second_price',
              replay_output_path=None):
    """Produce a canonical report. No click settlement or real auction claims."""
    output = Path(output_path)
    inputs = [Path(model_path), Path(contract_path), Path(features_path)]
    if calibration_path is not None:
        inputs.append(Path(calibration_path))
    replay_output = Path(replay_output_path) if replay_output_path is not None else None
    protected = {p.resolve() for p in inputs}
    if output.exists() or output.resolve() in protected:
        raise FileExistsError('Output exists or would overwrite an input')
    if replay_output is not None and (replay_output.exists() or
            replay_output.resolve() in protected or
            replay_output.resolve() == output.resolve()):
        raise FileExistsError('Replay output exists or would overwrite an input/report')
    bundle = load_model_auction_bundle(model_path, contract_path, calibration_path, method=method)
    rows = sample_features(features_path, bundle.contract['features'], count)
    # Synthetic values: no inference of logged advertiser, bid, budget or eligibility.
    bids = (2.00, 1.50, 1.00)
    candidates = [FeatureCandidate(f'synthetic-ad-{i+1}', f'synthetic-adv-{i+1}',
                                   bids[i % len(bids)], row)
                  for i, row in enumerate(rows)]
    budgets = {candidate.advertiser_id: 500 for candidate in candidates}
    ledger = BudgetLedger(budgets)
    request = AdRequest('m07a3-smoke', 'synthetic-shopper')
    prior_events = ledger.events  # Capture before the serving decision mutates the ledger.
    result = decide_with_model(request, candidates,
                               ledger, bundle, policy=policy, event_id='m07a3-reserve',
                               reservation_id='m07a3-reservation')
    if replay_output is not None:
        record = capture_replay_record(
            request=request, scored_candidates=result.scored_candidates,
            provenance=result.provenance, initial_budgets_cents=budgets,
            prior_ledger_events=prior_events, policy=policy,
            event_id='m07a3-reserve', reservation_id='m07a3-reservation',
            reserve_cpc=0.0, decision=result.decision, final_ledger=ledger)
        replay_decision(json.loads(json.dumps(record)),
                        expected_provenance=result.provenance)
    # M07B-1: reuse the exact batch scored by the auction service.
    predictions = [candidate.predicted_ctr for candidate in result.scored_candidates]
    if len(predictions) != count:
        raise ValueError('Unexpected scored candidate count')
    report = {
        'schema_version': 1, 'experiment': 'M07A-3',
        'interpretation': 'Real persisted-model inference; synthetic bids, advertisers and budgets; no observed auction lift',
        'inputs': {'features_sha256': hashlib.sha256(Path(features_path).read_bytes()).hexdigest(),
                   'sample_count': count, 'sample_policy': 'first_n_csv_rows'},
        'provenance': asdict(result.provenance),
        'auction': {'policy': policy, 'reserve_cpc': 0.0, 'synthetic_bids_cpc':
                    [candidate.bid_cpc for candidate in candidates]},
        'predicted_ctr': [float(p) for p in predictions],
        'decision': asdict(result.decision),
        'scored_candidate_count': result.scored_candidate_count,
        'ledger_snapshots': [asdict(ledger.snapshot(key)) for key in sorted(budgets)],
        'ledger_event_count': len(ledger.events),
    }
    report['report_sha256'] = _digest(report)
    if replay_output is not None:
        replay_output.parent.mkdir(parents=True, exist_ok=True)
        with replay_output.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='artifacts/m05fa/model/lightgbm_intent.txt')
    parser.add_argument('--contract', default='artifacts/m05fa/model/lightgbm_intent.contract.json')
    parser.add_argument('--features', default='artifacts/m05fa/splits/evaluation_features.csv')
    parser.add_argument('--calibrator', default=None)
    parser.add_argument('--method', choices=('raw', 'sigmoid', 'isotonic'), default='raw')
    parser.add_argument('--policy', choices=('first_price', 'score_second_price'), default='score_second_price')
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--output', required=True)
    parser.add_argument('--replay-output', default=None,
                        help='Optional versioned scored-decision replay sidecar JSON')
    args = parser.parse_args(argv)
    report = run_smoke(model_path=args.model, contract_path=args.contract,
                       features_path=args.features, output_path=args.output,
                       calibration_path=args.calibrator, method=args.method,
                       count=args.count, policy=args.policy,
                       replay_output_path=args.replay_output)
    print(json.dumps({'output': args.output, 'report_sha256': report['report_sha256'],
                      'status': report['decision']['status']}, sort_keys=True))


if __name__ == '__main__':
    main()
