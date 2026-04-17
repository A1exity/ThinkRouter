# RESULTS

No final official benchmark results have been produced yet. Most committed results are deterministic mock-model smoke tests for validating the experiment pipeline. A small provider-backed Qwen smoke result has now been recorded separately.

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
- official GSM8K subset export to JSONL,
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



## Official GSM8K Data Export

`prepare_data --source gsm8k` was used to download `openai/gsm8k` through the Hugging Face mirror and export a local JSONL subset:

| file | total | train | dev | test |
| --- | ---: | ---: | ---: | ---: |
| `data/splits/gsm8k.jsonl` | 100 | 60 | 20 | 20 |

The exported JSONL file and Hugging Face cache are local artifacts and are not committed. A 2-sample mock `run_grid --input data/splits/gsm8k.jsonl` smoke test passed, confirming the official GSM8K JSONL is compatible with the grid runner.
## Qwen Real-Model Smoke Tests

Small provider-backed smoke tests were run with `qwen3.5-flash-2026-02-23` through DashScope OpenAI-compatible mode. These are not benchmark results; they validate that the real model path works and that budget-level traces can be recorded.

| file | rows | task | split | budgets | accuracy | total estimated cost | avg latency |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `results/tables/qwen_gsm8k_dev_smoke.csv` | 4 | gsm8k | dev | 0 | 1.000 | 0.000500 | 3.022s |
| `results/tables/qwen_gsm8k_dev_budget_grid.csv` | 12 | gsm8k | dev | 0,256,1024 | 1.000 | 0.001778 | 3.461s |

Budget-grid summary:

| budget | accuracy | avg cost | p95 latency | n |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.000115 | 4.814s | 4 |
| 256 | 1.000 | 0.000131 | 3.147s | 4 |
| 1024 | 1.000 | 0.000199 | 6.496s | 4 |

Generated artifacts:

- `results/tables/qwen_gsm8k_dev_budget_grid.csv`
- `results/tables/qwen_gsm8k_dev_budget_summary.csv`
- `results/figures/qwen_gsm8k_dev_budget_pareto.png`

The local `.env` file contains the API key and is intentionally ignored by git. SQLite traces under `results/traces/` are also ignored.
## Final Reporting Targets

The final report should include:

- accuracy,
- average cost per query,
- p95 latency,
- cost reduction vs Always Strong + Max Budget,
- accuracy drop vs Always Strong + Max Budget,
- cost-accuracy Pareto plot,
- per-benchmark comparison for GSM8K, MATH-500, and HumanEval.