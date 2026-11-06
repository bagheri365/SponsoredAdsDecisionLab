"""M07B-2 scored-decision replay. No model loading, inference, or billing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from collections.abc import Mapping

from .budget_ledger import BudgetLedger, LedgerEvent
from .decision_service import DecisionResponse, decide
from .model_auction_service import ModelProvenance
from .models import AdCandidate, AdRequest

SCHEMA_VERSION = 1


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False, ensure_ascii=False).encode('utf-8')


def _digest(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _fields(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f'{name} must have exactly {sorted(expected)}')
    return value


def _list(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a list')
    return value


def _decode(value: object, cls: type, name: str):
    from dataclasses import fields
    keys = {field.name for field in fields(cls)}
    return cls(**_fields(value, keys, name))


def _snapshots(ledger: BudgetLedger, budgets: dict[str, int]) -> list[dict]:
    return [asdict(ledger.snapshot(key)) for key in sorted(budgets)]


def _event_dicts(events: tuple[LedgerEvent, ...]) -> list[dict]:
    return [asdict(event) for event in events]


def _payload(*, request: AdRequest, candidates: tuple[AdCandidate, ...],
             provenance: ModelProvenance, budgets_cents: dict[str, int],
             prior_events: tuple[LedgerEvent, ...], policy: str, event_id: str,
             reservation_id: str, reserve_cpc: float,
             decision: DecisionResponse, ledger: BudgetLedger) -> dict:
    return {
        'schema_version': SCHEMA_VERSION,
        'replay_kind': 'scored_decision_no_inference',
        'request': asdict(request),
        'scored_candidates': [asdict(item) for item in candidates],
        'model_provenance': asdict(provenance),
        'initial_budgets_cents': dict(budgets_cents),
        'prior_ledger_events': _event_dicts(prior_events),
        'auction': {'policy': policy, 'event_id': event_id,
                    'reservation_id': reservation_id, 'reserve_cpc': reserve_cpc},
        'expected_decision': asdict(decision),
        'expected_ledger_events': _event_dicts(ledger.events),
        'expected_ledger_snapshots': _snapshots(ledger, budgets_cents),
    }


def capture_replay_record(*, request: AdRequest,
                          scored_candidates: tuple[AdCandidate, ...],
                          provenance: ModelProvenance,
                          initial_budgets_cents: dict[str, int],
                          prior_ledger_events: tuple[LedgerEvent, ...],
                          policy: str, event_id: str, reservation_id: str,
                          reserve_cpc: float, decision: DecisionResponse,
                          final_ledger: BudgetLedger) -> dict:
    """Capture a verified post-decision result, including the pre-decision state.

    The caller must save prior events *before* deciding; no inference is done.
    """
    if not isinstance(request, AdRequest) or not isinstance(provenance, ModelProvenance):
        raise TypeError('request/provenance must use their declared contracts')
    if not isinstance(scored_candidates, tuple) or not all(
            isinstance(c, AdCandidate) for c in scored_candidates):
        raise TypeError('scored_candidates must be a tuple of AdCandidate')
    if not isinstance(prior_ledger_events, tuple) or not all(
            isinstance(e, LedgerEvent) for e in prior_ledger_events):
        raise TypeError('prior_ledger_events must be a tuple of LedgerEvent')
    if not isinstance(decision, DecisionResponse) or not isinstance(final_ledger, BudgetLedger):
        raise TypeError('decision/final_ledger must use their declared contracts')
    # Copy and validate budgets by replaying the complete initial event history.
    budgets = dict(initial_budgets_cents)
    BudgetLedger.replay(budgets, prior_ledger_events)
    record = _payload(request=request, candidates=scored_candidates,
                      provenance=provenance, budgets_cents=budgets,
                      prior_events=prior_ledger_events, policy=policy,
                      event_id=event_id, reservation_id=reservation_id,
                      reserve_cpc=reserve_cpc, decision=decision, ledger=final_ledger)
    record['sha256'] = _digest(record)
    # Reject a captured result if the supplied pre-state cannot reproduce it.
    replay_decision(record, expected_provenance=provenance)
    return record


def replay_decision(record: Mapping, *, expected_provenance: ModelProvenance | None = None
                    ) -> tuple[DecisionResponse, BudgetLedger]:
    """Validate a versioned record and deterministically re-run scored auction.

    A hash detects accidental changes, not malicious rewriting (no signature).
    A mismatch raises ValueError; the reconstructed ledger is not returned.
    """
    if not isinstance(record, Mapping):
        raise TypeError('record must be a mapping')
    data = dict(record)
    required = {'schema_version', 'replay_kind', 'request', 'scored_candidates',
                'model_provenance', 'initial_budgets_cents', 'prior_ledger_events',
                'auction', 'expected_decision', 'expected_ledger_events',
                'expected_ledger_snapshots', 'sha256'}
    _fields(data, required, 'record')
    if type(data['schema_version']) is not int or data['schema_version'] != SCHEMA_VERSION:
        raise ValueError('unsupported replay schema version')
    if data['replay_kind'] != 'scored_decision_no_inference':
        raise ValueError('unsupported replay kind')
    if not isinstance(data['sha256'], str) or data['sha256'] != _digest(
            {key: value for key, value in data.items() if key != 'sha256'}):
        raise ValueError('replay record hash mismatch')
    try:
        request = _decode(data['request'], AdRequest, 'request')
        candidates = tuple(_decode(item, AdCandidate, 'scored candidate') for item in
                           _list(data['scored_candidates'], 'scored_candidates'))
        provenance = _decode(data['model_provenance'], ModelProvenance, 'model_provenance')
        if expected_provenance is not None and provenance != expected_provenance:
            raise ValueError('model provenance mismatch')
        budgets = data['initial_budgets_cents']
        if not isinstance(budgets, dict):
            raise ValueError('initial_budgets_cents must be an object')
        prior = tuple(_decode(item, LedgerEvent, 'prior event') for item in
                      _list(data['prior_ledger_events'], 'prior_ledger_events'))
        auction = _fields(data['auction'],
                          {'policy', 'event_id', 'reservation_id', 'reserve_cpc'}, 'auction')
        ledger = BudgetLedger.replay(budgets, prior)
        result = decide(request, candidates, ledger, **auction)
        # Normalize tuples to JSON arrays before comparing persisted records.
        if _canonical(asdict(result)) != _canonical(data['expected_decision']):
            raise ValueError('replayed decision mismatch')
        if _event_dicts(ledger.events) != data['expected_ledger_events']:
            raise ValueError('replayed ledger events mismatch')
        if _snapshots(ledger, budgets) != data['expected_ledger_snapshots']:
            raise ValueError('replayed ledger snapshots mismatch')
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError('malformed replay record') from exc
    return result, ledger
