# RESULTS

No final official benchmark results have been produced yet. The current results are deterministic mock-model smoke tests for validating the experiment pipeline. Real-model tooling is present, but no provider-backed traces have been committed.

## Current Status

Implemented Day-1 and Week-2 pipeline components:

- built-in GSM8K-style, MATH-style, and HumanEval-style seed samples,
- fixed budgets `0`, `256`, `1024`, `4096`,
- mock and OpenAI-compatible model adapters,
- GSM8K numeric exact-match evaluator,
- exact-match evaluator for non-GSM8K seed tasks,
- SQLite trace store,
- Day-1 grid runner,
- frozen train/dev/test grid runner,
- benchmark JSONL export and input loader,
- real OpenAI-compatible endpoint configuration preflight,
- FastAPI endpoints,
- Streamlit trace demo,
- baseline summary and Pareto plotting scripts,
- sklearn difficulty estimator training,
- sklearn budget predictor training,
- optional router loading from joblib model paths.

## Frozen Seed Splits

The local seed suite is for deterministic pipeline validation only. It is not a replacement for official GSM8K, MATH-500, or HumanEval benchmark reporting.

| task | train | dev | test |
| --- | ---: | ---: | ---: |
| gsm8k | 12 | 4 | 4 |
| math | 6 | 2 | 2 |
| humaneval | 4 | 2 | 2 |

Generated trace tables:

| table | traces | task coverage |
| --- | ---: | --- |
| `results/tables/train_grid.csv` | 132 | gsm8k, math, humaneval |
| `results/tables/dev_grid.csv` | 48 | gsm8k, math, humaneval |
| `results/tables/test_grid.csv` | 48 | gsm8k, math, humaneval |

The trace counts reflect 2 mock models and 3 budget levels per sample.

## Generated Artifacts

- Day-1 trace database: `results/traces/day1.sqlite`
- frozen train/dev/test trace databases: `results/traces/train_grid.sqlite`, `results/traces/dev_grid.sqlite`, `results/traces/test_grid.sqlite`
- Day-1 grid table: `results/tables/day1_grid.csv`
- frozen split tables: `results/tables/train_grid.csv`, `results/tables/dev_grid.csv`, `results/tables/test_grid.csv`
- Day-1 baseline summary: `results/tables/baseline_summary.csv`
- dev baseline summary: `results/tables/dev_baseline_summary.csv`
- Day-1 Pareto figure: `results/figures/pareto.png`
- dev Pareto figure: `results/figures/dev_pareto.png`
- train-split difficulty model: `results/models/difficulty.joblib`
- train-split budget model: `results/models/budget.joblib`

## Reproduction Commands

```bash
python -m thinkrouter.experiments.run_day1_grid --limit 20 --db results/traces/day1.sqlite --out results/tables/day1_grid.csv
python -m thinkrouter.experiments.eval_baselines results/tables/day1_grid.csv --out results/tables/baseline_summary.csv
python -m thinkrouter.experiments.make_plots results/tables/day1_grid.csv --out results/figures/pareto.png

python -m thinkrouter.experiments.prepare_data --source seed --task all --split all --out data/splits/seed.jsonl --summary
python -m thinkrouter.experiments.run_grid --input data/splits/seed.jsonl --task all --split train --budgets 0,256,1024 --db results/traces/train_grid.sqlite --out results/tables/train_grid.csv
python -m thinkrouter.experiments.run_grid --input data/splits/seed.jsonl --task all --split dev --budgets 0,256,1024 --db results/traces/dev_grid.sqlite --out results/tables/dev_grid.csv
python -m thinkrouter.experiments.run_grid --input data/splits/seed.jsonl --task all --split test --budgets 0,256,1024 --db results/traces/test_grid.sqlite --out results/tables/test_grid.csv

python -m thinkrouter.experiments.train_difficulty results/tables/train_grid.csv --out results/models/difficulty.joblib
python -m thinkrouter.experiments.train_budget results/tables/train_grid.csv --out results/models/budget.joblib
python -m thinkrouter.experiments.eval_baselines results/tables/dev_grid.csv --out results/tables/dev_baseline_summary.csv
python -m thinkrouter.experiments.make_plots results/tables/dev_grid.csv --out results/figures/dev_pareto.png
```

The JSONL interface was smoke-tested by exporting 38 seed samples to `data/splits/seed.jsonl` and running a small `run_grid --input` job. `data/splits/` and SQLite traces are intentionally ignored by git.

A real endpoint can be checked with `python -m thinkrouter.experiments.smoke_real_model --model <model-id>`. Passing `--run` performs an actual provider call and can incur cost.

Because the committed results use deterministic mock adapters, all current accuracies are expected to be perfect. The value of this stage is validating the train/dev/test and JSONL experiment plumbing, not measuring model quality.

## Final Reporting Targets

The final report should include:

- accuracy,
- average cost per query,
- p95 latency,
- cost reduction vs Always Strong + Max Budget,
- accuracy drop vs Always Strong + Max Budget,
- cost-accuracy Pareto plot,
- per-benchmark comparison for GSM8K, MATH-500, and HumanEval.