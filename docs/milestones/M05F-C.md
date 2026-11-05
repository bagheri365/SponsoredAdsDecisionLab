# M05F-C — Explicit CTR calibration serving bundle

**Scope:** Persist a calibration transform fitted exclusively on the M05F-A calibration partition. Keep **raw** CTR as the default serving policy. Sigmoid and isotonic are opt-in policies, not automatically promoted based on the M05F-B evaluation metrics.

## Fit an opt-in calibrator

Prerequisites: the verified M05F-A split/model/contract artifacts. The runner checks the M05F-A manifest and retraining hashes through the M05F-B provenance verifier. It reads **only calibration labels**, never evaluation labels or published test data. Artifacts are JSON (not executable pickle), with a strict schema and SHA-256 binding to model, contract, and calibration input. It refuses overwrites.

```bash
pytest -q
python -m sponsored_ads.m05fc_serving \
  --split-dir artifacts/m05fa/splits \
  --model-dir artifacts/m05fa/model \
  --method sigmoid \
  --output artifacts/m05fc/sigmoid.calibrator.json
```

For a separately authorized isotonic policy, replace `sigmoid` with `isotonic` and use a different output file. This command **does not choose** a winner or run final evaluation. The explicit policy choice is a deployment/configuration decision.

## Serving API

```python
from sponsored_ads.m05fc_serving import load_serving_bundle, predict_calibrated
model, contract, calibrator = load_serving_bundle(
    'artifacts/m05fa/model/lightgbm_intent.txt',
    'artifacts/m05fa/model/lightgbm_intent.contract.json',
    'artifacts/m05fc/sigmoid.calibrator.json', method='sigmoid')
probabilities = predict_calibrated(model, contract, complete_feature_rows, calibrator)
```

Without a calibrator, `load_serving_bundle(model_path, contract_path, None)` defaults to raw probabilities. A calibrated policy requires a matching artifact. All candidate feature requirements from M05A/M05B remain unchanged: no missing features are fabricated.

## Limitations

M05F-B compared methods on the evaluation partition; selecting the method on that evidence and then reporting the same partition as an unbiased final test is invalid. This milestone implements explicit opt-in, not automatic promotion. The original development split is row-hash based, not temporal or user-disjoint, and histories lack verified point-in-time validity. These are logged-impression probabilities, not auction policy effectiveness estimates. Model and contract hashes guard compatibility but do not establish training-data authenticity beyond the verified provenance chain.
