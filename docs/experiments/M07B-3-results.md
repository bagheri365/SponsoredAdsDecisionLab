# M07B-3: Persisted-model replay integration — local results

Status: Locally verified; pending staged review and commit.

Baseline: `f6093bd`.

## Test and regression evidence

- Initial integration: 3 focused tests passed; full suite 370 passed, 1 warning.
- A real raw run generated its original report but no replay sidecar because the CLI accepted `--replay-output` without forwarding it to `run_smoke`.
- Corrective patch forwarded `replay_output_path=args.replay_output` and added a CLI regression test.
- After correction: `pytest -q tests/test_m07b3_replay_integration.py`: **4 passed** (3.02 s).
- `pytest -q`: **371 passed, 1 warning** (2.93 s).
- The warning is the existing scikit-learn/LightGBM feature-name warning in `tests/test_ctr_baselines.py::test_lightgbm_smoke`.
- `git diff --check`: no whitespace errors reported.

## Real persisted-model experiments

- Model and feature inputs: existing M07A-3 persisted-model serving defaults and first three evaluation feature rows.
- Calibrator: `artifacts/m05fc/sigmoid.calibrator.json`, verified complete-file SHA-256 `b2d6c5ff6ed077d090bc3461005fa4af12bda84d29259d029b271c6083f1a2b9`.
- Synthetic bids, advertisers, and budgets; `score_second_price` auction policy.
- The corrected raw run wrote `artifacts/m07b3/raw_report_v2.json` and `artifacts/m07b3/raw_replay_v2.json`.
- The sigmoid run wrote `artifacts/m07b3/sigmoid_report.json` and `artifacts/m07b3/sigmoid_replay.json`.
- Both CLI runs reported `filled`. Independent JSON-loaded `replay_decision` checks passed for both scenarios, including decisions, ledger event counts, snapshots, and model provenance against the serving reports.
- Both winners: `synthetic-ad-1`; both replayed ledgers: 1 event.

| SHA-256 evidence | Raw | Sigmoid |
|---|---|---|
| Embedded report (matches M07A-3) | `d23bc6a4d5857dd0ab320f4d3afa74d4a63a044206e4ae06cf6d7659ccb7a2f0` | `0a52e656ae3da19d2ea0911f7c4e27400a9ee34e591e8a7b2910d92ebf323576` |
| Complete report file | `f31db9166e4f13c19180f1bc84761590cb42ccaa6bab58794e5324bb7c006d53` | `9e72f3cf9f35319c1ca9b41af25071d45e782cb90053bd7329ab552bd69c2704` |
| Embedded replay record | `f4a7e1d382e9e60897d6f0aa4e751e62447a210b15335429c1c9665f8a97e461` | `971658b51d1ea76646fbd78ca0c5f71e42fc13fb3acd4e295f251de7b283f23a` |
| Complete replay file | `5713d3938b5303619368f0c924df90b772477823d24e07d2c86f30d315392a39` | `42cb6a83dfd4e6d576c227ec2f720b17bf26cbb12d83a75122588b381e31ebc1` |

The raw complete-file report hash also matches the previously recorded M07A-3 baseline. The sigmoid complete-file hash is recorded here but has **not** yet been independently compared with the original M07A-3 sigmoid report file; its embedded report hash matches the baseline.

## Interpretation and limitations

M07B-3 demonstrates opt-in, JSON-persisted replay of the scored auction from real persisted-model predictions, with no extra inference in the integration tests. Replay itself does not reload the model or reconstruct its features. Synthetic auction inputs do not establish online performance, production billing correctness, or revenue lift. The sidecar and report are not written atomically as a pair.
