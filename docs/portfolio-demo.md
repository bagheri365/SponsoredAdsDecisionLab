# Five-minute portfolio demo

Run from the repository root with Python 3.10+. The demo uses **synthetic**
inputs and does not need the Taobao CSV or a persisted LightGBM artifact.

## 1. Install and test (approximately two minutes)

```bash
python -m pip install -e '.[dev,ml]'
pytest -q
```

The M08-1 checkpoint reported 388 passes and one known LightGBM feature-name
warning. A fresh-clone release run must be checked independently; do not assume
its result before executing it.

## 2. Show the decision interface (approximately one minute)

```bash
python -m sponsored_ads
```

Discuss single-slot allocation using bid × predicted CTR and the distinction
between ranking score and conditional CPC. This command is an interface smoke
test, not an end-to-end production-serving benchmark.

## 3. Run the synthetic evaluation (approximately one minute)

```bash
python -m sponsored_ads.m08_synthetic_evaluation \
  --output artifacts/m08/portfolio_demo.json
```

The CLI refuses to overwrite an existing file. Choose another output name if
`portfolio_demo.json` already exists. Inspect the emitted `input_sha256`,
`report_sha256`, and `allocation_agreement` fields.

## 4. Explain the results and boundaries (approximately one minute)

```bash
python - <<'PYCODE'
import json
from pathlib import Path
p = Path('artifacts/m08/portfolio_demo.json')
r = json.loads(p.read_text())
print('allocation_agreement:', r['allocation_agreement'])
for item in r['policy_reports']:
    print(item['policy'], item['metrics'])
PYCODE
```

In the documented M08-1 run, both policies filled 3/6 scenarios and agreed with
the synthetic relevance oracle on 2/3 fills. First-price mean conditional CPC
was 2.0, versus 0.8333 for score-second-price. No observed clicks or charges
were generated; budget amounts are reservations. The fixture comparison does
not establish a production winner or causal lift.

For deeper discussion, open [architecture and tradeoffs](architecture.md),
[M08-1 results](experiments/M08-1-results.md), and the
[M07B-4 verifier milestone](milestones/M07B-4.md). Persisted-model smoke and
replay require their documented local model/artifact prerequisites and are not
part of this zero-data demo.
