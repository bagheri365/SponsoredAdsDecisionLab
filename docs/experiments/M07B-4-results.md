# M07B-4: Offline artifact verification — local results

Status: Locally verified; pending staged review and commit.
Baseline: `5f662da` (M07B-3).

## Tests and checks

- `pytest -q tests/test_m07b4_verify.py`: **8 passed** (3.09 s).
- `pytest -q`: **379 passed, 1 warning** (2.89 s).
- Warning: existing scikit-learn/LightGBM feature-name warning in `tests/test_ctr_baselines.py::test_lightgbm_smoke`.
- `git diff --check`: no whitespace errors reported.
- Implementation adds a read-only verifier and tests; no changes to serving, model inference, auction, ledger, or report schemas.

## Persisted-artifact verification

Both commands in `docs/milestones/M07B-4.md` returned `status: PASS`, winner `synthetic-ad-1`, and one ledger event. Raw command supplied the known M07A-3 complete-file report hash; sigmoid command did not supply a complete-file baseline hash.

| SHA-256 evidence | Raw | Sigmoid |
|---|---|---|
| Embedded report | `d23bc6a4d5857dd0ab320f4d3afa74d4a63a044206e4ae06cf6d7659ccb7a2f0` | `0a52e656ae3da19d2ea0911f7c4e27400a9ee34e591e8a7b2910d92ebf323576` |
| Complete report file | `f31db9166e4f13c19180f1bc84761590cb42ccaa6bab58794e5324bb7c006d53` | `9e72f3cf9f35319c1ca9b41af25071d45e782cb90053bd7329ab552bd69c2704` |
| Embedded replay record | `f4a7e1d382e9e60897d6f0aa4e751e62447a210b15335429c1c9665f8a97e461` | `971658b51d1ea76646fbd78ca0c5f71e42fc13fb3acd4e295f251de7b283f23a` |
| Complete replay file | `5713d3938b5303619368f0c924df90b772477823d24e07d2c86f30d315392a39` | `42cb6a83dfd4e6d576c227ec2f720b17bf26cbb12d83a75122588b381e31ebc1` |

The embedded report hashes match accepted M07A-3 values. The raw complete-file report hash matches the independently checked M07A-3 baseline. The sigmoid complete-file hash agrees with the recorded M07B-3 value, but has **not** been independently compared with the original M07A-3 sigmoid file.

## Interpretation and limitations

The verifier checks integrity, provenance, scored-decision replay, and report/sidecar consistency offline, without loading the model or running inference. Positive and negative regression tests passed. This does not authenticate artifacts cryptographically, prove feature or inference reproducibility, validate real billing, or establish online lift. The serving report and replay sidecar are not written atomically.
