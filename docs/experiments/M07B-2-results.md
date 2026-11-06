# M07B-2: Deterministic scored-decision replay — local results

Status: Locally verified; pending review and commit.

## Test evidence

- `pytest -q tests/test_m07b2_decision_replay.py`: **10 passed** (2.74 s).
- `pytest -q`: **367 passed, 1 warning** (2.93 s).
- Warning: pre-existing scikit-learn/LightGBM feature-name warning in `tests/test_ctr_baselines.py::test_lightgbm_smoke`; unrelated to replay.
- `git diff --check`: no reported whitespace errors. Four M07B-2 files remained untracked at verification time.

## Synthetic replay experiment

The local experiment created `artifacts/m07b2/replay_record.json` and replayed the saved JSON record twice, producing `replay_run1.json` and `replay_run2.json`.

- Result: **Deterministic replay: PASS**; both replay summaries were byte-identical.
- Winner: `ad-1`.
- Ledger events: **1** (reservation).
- Embedded canonical record SHA-256: `6c655b59c87c5fddd7d7362e2144d76ab4372cfbea75ad90d6b7ffbc5176c7ea`.
- Complete-file SHA-256 of `replay_record.json`: `a51551a1e37b57f13168493640fd85fff14ccb612d48bc2c97005d60efbc41dc`.
- Complete-file SHA-256 of **both** `replay_run1.json` and `replay_run2.json`: `ae8e1b238a14ae36b6b5318b5e731f35fa80cd9ca09aa20f6510fcfb442d6a44`.

The experiment used synthetic candidates, bids, budgets, and **placeholder** model-provenance digests (`a` and `b` repeated 64 times). It demonstrates scored-decision replay **without model inference**, not verification of a persisted model or authenticity of its artifacts. The record checksum detects accidental modifications but is not a signature.

## Verification correction

An initial direct comparison of Python dataclass dictionaries to JSON records caused a tuple/list representation mismatch. The final locally tested implementation compares canonical JSON encodings of both decision values:

```python
if _canonical(asdict(result)) != _canonical(data['expected_decision']):
    raise ValueError('replayed decision mismatch')
```

This correction is included in the locally verified 367-pass result. No commit or push has been reported for M07B-2.
