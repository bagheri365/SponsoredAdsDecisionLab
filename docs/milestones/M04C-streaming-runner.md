# M04C — Streaming evaluation runner (stage 2)

**Status: implementation and fixture tests; real-data experiment pending.**

Run from the repository root:

```bash
python -m sponsored_ads.retrieval_evaluation_runner \
  --test-csv data/TaobaoAd_x1/test.csv \
  --catalog artifacts/m04a/catalog.json \
  --sample-size 100 --seed 2026 --pool-size 100 \
  --output artifacts/m04c/evaluation.json
python -m json.tool artifacts/m04c/evaluation.json
```

The runner streams the entire published test CSV to count clicked impressions and select the lowest 100 stable BLAKE2b hash keys of `(seed, CSV row number)` among valid clicked rows. It retains only sampled rows in memory, rather than the full test CSV. It builds the train-derived M04A index once and evaluates popularity, category/brand affinity, and RRF fusion at K=10/50/100 with a common pool of 100. Each sampled clicked impression is weighted equally, including repeated clicks on the same ad. Invalid clicked histories are counted and excluded **before sampling**. A CSV with an invalid `clk` label fails rather than silently skipping rows.

Outputs contain sample counts, catalog coverage, exposure-conditioned clicked-ad recovery, catalog-conditional recovery, fused-hit source labels, per-query latency, and runtime. The sample is deterministic for a fixed file order and seed; changing row order changes membership. Do not tune hyperparameters against the published test set. The sample is not a substitute for a full-split experiment or confidence intervals.

**Important limitations:** The M04A catalog is train-derived but not request-time eligible. Published train/test temporal and user separation are unverified. Historical category/brand lists have no verified point-in-time validity or ordering, so the measured retrieval results may be contaminated by leakage. Test clicks are logged exposures, not unbiased relevance labels. The report does not measure online CTR, lift, or counterfactual policy value. The runner does not yet perform automated train/test user overlap audits, content hashing, segment breakdowns, or native process RSS tracking. Those are outstanding gates before the milestone is considered complete.
