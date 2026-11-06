"""M07B-4: independently verify a persisted serving report and replay sidecar.

No model loading, feature reconstruction, inference, or ledger mutation on disk.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .decision_replay import replay_decision
from .model_auction_service import ModelProvenance


BASELINES = {
    'raw': {
        'report_sha256': 'd23bc6a4d5857dd0ab320f4d3afa74d4a63a044206e4ae06cf6d7659ccb7a2f0',
        'report_file_sha256': 'f31db9166e4f13c19180f1bc84761590cb42ccaa6bab58794e5324bb7c006d53',
    },
    'sigmoid': {
        'report_sha256': '0a52e656ae3da19d2ea0911f7c4e27400a9ee34e591e8a7b2910d92ebf323576',
        # M07B-3 reference only: not independently compared with M07A-3 full file.
        'report_file_sha256': '9e72f3cf9f35319c1ca9b41af25071d45e782cb90053bd7329ab552bd69c2704',
    },
}
MODEL_SHA = '47d9611078d1f79fd4a25b1a00c45ba7cb920762d22e56e08182112ca2aee02c'
CONTRACT_SHA = '89497d6396079b9cf1d6d8353180a6757857e6df3be11235e7f45f3886c89daa'
SIGMOID_SHA = 'b2d6c5ff6ed077d090bc3461005fa4af12bda84d29259d029b271c6083f1a2b9'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _load(path):
    content = Path(path).read_bytes()
    value = json.loads(content)
    _require(isinstance(value, dict), f'{path}: expected JSON object')
    return value, hashlib.sha256(content).hexdigest()


def verify_artifacts(*, report_path, replay_path, scenario,
                     expected_report_file_sha256=None,
                     expected_replay_file_sha256=None):
    """Validate baseline, paired artifacts, and no-inference auction replay.

    Raises ValueError for verification failures. Missing files raise OSError.
    """
    if scenario not in BASELINES:
        raise ValueError('scenario must be raw or sigmoid')
    baseline = BASELINES[scenario]
    report, report_file_hash = _load(report_path)
    record, replay_file_hash = _load(replay_path)
    _require(report.get('schema_version') == 1 and report.get('experiment') == 'M07A-3',
             'unsupported serving report')
    _require(report.get('report_sha256') == hashlib.sha256(_canonical({
        key: value for key, value in report.items() if key != 'report_sha256'
    })).hexdigest(), 'report checksum mismatch')
    _require(report['report_sha256'] == baseline['report_sha256'],
             'report differs from accepted embedded baseline')
    if expected_report_file_sha256 is not None:
        _require(report_file_hash == expected_report_file_sha256,
                 'report complete-file checksum mismatch')
    if expected_replay_file_sha256 is not None:
        _require(replay_file_hash == expected_replay_file_sha256,
                 'replay complete-file checksum mismatch')
    expected_provenance = ModelProvenance(
        MODEL_SHA, CONTRACT_SHA, scenario,
        SIGMOID_SHA if scenario == 'sigmoid' else None, 1,
    )
    _require(report.get('provenance') == asdict(expected_provenance),
             'report provenance differs from accepted baseline')
    decision, ledger = replay_decision(record, expected_provenance=expected_provenance)
    _require(record['model_provenance'] == report['provenance'], 'paired provenance mismatch')
    _require(_canonical(asdict(decision)) == _canonical(report.get('decision')),
             'paired decision mismatch')
    _require(record['expected_ledger_snapshots'] == report.get('ledger_snapshots'),
             'paired ledger snapshots mismatch')
    _require(len(ledger.events) == report.get('ledger_event_count'),
             'paired ledger event count mismatch')
    candidates = record['scored_candidates']
    _require(len(candidates) == report.get('scored_candidate_count'),
             'paired scored candidate count mismatch')
    _require([c['predicted_ctr'] for c in candidates] == report.get('predicted_ctr'),
             'paired CTR predictions mismatch')
    _require([c['bid_cpc'] for c in candidates] == report.get('auction', {}).get('synthetic_bids_cpc'),
             'paired synthetic bids mismatch')
    _require(record['auction']['policy'] == report.get('auction', {}).get('policy') and
             record['auction']['reserve_cpc'] == report.get('auction', {}).get('reserve_cpc'),
             'paired auction configuration mismatch')
    _require(len(candidates) == report.get('inputs', {}).get('sample_count'),
             'paired sample count mismatch')
    return {
        'status': 'PASS', 'scenario': scenario,
        'report_sha256': report['report_sha256'],
        'report_file_sha256': report_file_hash,
        'replay_sha256': record['sha256'],
        'replay_file_sha256': replay_file_hash,
        'winner': decision.outcome.ad_id if decision.outcome is not None else None,
        'ledger_event_count': len(ledger.events),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=tuple(BASELINES), required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--expected-report-file-sha256')
    parser.add_argument('--expected-replay-file-sha256')
    args = parser.parse_args(argv)
    try:
        result = verify_artifacts(
            report_path=args.report, replay_path=args.replay, scenario=args.scenario,
            expected_report_file_sha256=args.expected_report_file_sha256,
            expected_replay_file_sha256=args.expected_replay_file_sha256,
        )
    except (ValueError, OSError, TypeError, OverflowError) as exc:
        parser.exit(1, f'FAIL: {exc}\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
