# M05D — Synthetic experiment run log

**Status:** Executed on the target machine; measured results recorded below. The experiment is synthetic infrastructure validation, not an assessment of real-world CTR ranking quality.

## Reproduction

Repository baseline before M05D: `6af8fa4` (`main`). M05D changes were applied but had not yet been committed when the experiment was run. The eventual M05D commit hash should be recorded after committing.

```bash
python -m sponsored_ads.preranking_experiment \
  --seed 2026 \
  --cases 20 \
  --candidates-per-case 10 \
  --top-k 3 \
  --missing-rate 0.1 \
  --repeats 3 \
  --output artifacts/m05d/synthetic_report.json
```

### Configuration and environment

| Parameter | Value |
|---|---|
| Seed | 2026 |
| Evaluation cases | 20 |
| Candidates per case | 10 |
| Top-K | 3 |
| Configured missing-feature rate | 0.1 |
| Repeats | 3 |
| Python | 3.10.15 |
| NumPy | 2.2.6 |
| Platform | macOS (user's local Mac; OS version not recorded) |
| Report schema version | 1 |
| Report experiment identifier | `M05D_synthetic_infrastructure_only` |

## Measured quality and feature coverage

| Metric | Result |
|---|---:|
| Cases | 20 |
| Candidates | 200 |
| Scorable candidates | 181 |
| Unscorable candidates | 19 |
| Feature coverage | 90.5% (181/200) |
| Hit@3 | 0.4000 (8/20 cases) |
| MRR@3 | 0.2917 |

Feature coverage is calculated before top-K truncation. Unscorable positive candidates remain in the evaluation denominator. The observed 9.5% unscorable rate is the realized result of the synthetic experiment, not a measured missing-feature rate in TaobaoAd or a production service.

## Measured phase latency

All values are **milliseconds per evaluated case**, as reported by the experiment's phase timing summaries over three repeats. P95 uses the nearest-rank convention.

| Phase | Mean (ms) | P50 (ms) | P95 nearest rank (ms) |
|---|---:|---:|---:|
| Feature assembly | 0.0846 | 0.0835 | 0.0887 |
| Stub inference | 0.1089 | 0.1076 | 0.1193 |
| Sorting | 0.0158 | 0.0159 | 0.0194 |

These are local wall-clock measurements for small synthetic candidate groups, not end-to-end or production-serving latency. They exclude candidate retrieval, model/artifact loading, I/O, and network overhead. Inference uses a fixed price-based scoring stub, **not** a trained LightGBM model.

## Validation

```text
pytest -q
141 passed, 1 warning in 3.84s
```

The warning came from the existing LightGBM sklearn feature-name check in `tests/test_ctr_baselines.py::test_lightgbm_smoke`; it did not fail the test suite. `git apply --check` succeeded, and `git diff --check` reported no whitespace errors. Four intended M05D files were untracked prior to staging.

## Interpretation and limitations

The experiment uses synthetic candidates, synthetic feature values, and randomly assigned positive labels. The scoring function is a fixed price-based stub. **Hit@3 and MRR@3 validate evaluation mechanics only**; they do not establish trained-model quality, TaobaoAd relevance, online lift, or causal effectiveness.

The dataset does not supply verified eligible candidate sets or counterfactual labels for unexposed ads. Consequently, these synthetic ranking metrics must not be presented as observed-data effectiveness results.

The generated detailed report, `artifacts/m05d/synthetic_report.json`, remains a local artifact and is not committed to Git. The tables above transcribe its measured aggregate outputs, rounded for readability.
