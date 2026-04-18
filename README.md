# ThinkRouter

ThinkRouter is an adaptive thinking-budget router for reasoning LLMs. It treats the pair `(model, thinking_budget)` as the routing decision, then records quality, cost, and latency for verifiable reasoning tasks.

## Key Results

On the committed Qwen GSM8K held-out `test20` split, the dev-calibrated safe policy matched the high-budget `1024` baseline accuracy while lowering average estimated cost:

| policy | accuracy | avg cost | p95 latency | cost vs budget 1024 |
| --- | ---: | ---: | ---: | ---: |
| fixed budget 1024 | 0.950 | 0.000542 | 23.390s | 100.0% |
| dev-calibrated safe policy | 0.950 | 0.000344 | 16.721s | 63.5% |
| oracle upper bound | 1.000 | 0.000288 | 10.325s | 53.2% |

For a resume/interview-oriented overview, see `PROJECT_BRIEF.md`. The consolidated experiment report is in `results/reports/qwen_gsm8k_final_policy_report.md`.

This repository currently implements the Day-1 MVP loop plus the first Week-2 trainable router, frozen-split experiment components, a JSONL benchmark interface, offline policy evaluation, learned policy replay, and real OpenAI-compatible model smoke-test tooling:

1. load built-in GSM8K-style, MATH-style, and HumanEval-style seed samples,
2. run model configs across fixed budgets,
3. evaluate exact-match or numeric exact-match outputs,
4. store traces in SQLite,
5. export benchmark samples to standard JSONL,
6. run train/dev/test trace CSVs from either built-in seed data or JSONL input,
7. train lightweight difficulty and budget models from train traces,
8. validate or call a real OpenAI-compatible model endpoint,
9. compare fixed-budget, oracle, and aggregate-utility policies from completed grids,
10. train a learned policy router on one split and replay it on another split,
11. inspect or run examples through Streamlit or FastAPI.

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate thinkrouter
```

Copy the environment template if you want to use a real OpenAI-compatible endpoint or trained router models:

```bash
copy .env.example .env
```

By default, the code works with mock models and does not require API keys.

## Configuration

Environment variables:

- `THINKROUTER_DB_PATH`: SQLite path. Default: `results/traces/thinkrouter.sqlite`.
- `THINKROUTER_OPENAI_BASE_URL`: OpenAI-compatible base URL.
- `THINKROUTER_OPENAI_API_KEY`: API key for that endpoint.
- `THINKROUTER_CHEAP_MODEL`: cheap model id. Defaults to `mock-cheap`.
- `THINKROUTER_STRONG_MODEL`: strong model id. Defaults to `mock-strong` in code, example env uses `gpt-4o-mini`.
- `THINKROUTER_CHEAP_COST_PER_1K`: estimated cheap model cost per 1K tokens.
- `THINKROUTER_STRONG_COST_PER_1K`: estimated strong model cost per 1K tokens.
- `THINKROUTER_DIFFICULTY_MODEL_PATH`: optional joblib difficulty estimator path.
- `THINKROUTER_BUDGET_MODEL_PATH`: optional joblib budget predictor path.

Any model id starting with `mock` uses the local `MockAdapter`. Other model ids use the OpenAI-compatible adapter.

## Real Model Smoke Test

Before running a real grid, validate the endpoint configuration without making a network call:

```bash
python -m thinkrouter.experiments.smoke_real_model --model gpt-4o-mini
```

To actually call the configured endpoint, pass `--run`. This can incur provider cost:

```bash
python -m thinkrouter.experiments.smoke_real_model --model gpt-4o-mini --budget 0 --run
```

Required `.env` values for a real endpoint:

```bash
THINKROUTER_OPENAI_BASE_URL=https://api.openai.com/v1
THINKROUTER_OPENAI_API_KEY=replace-me
THINKROUTER_STRONG_MODEL=gpt-4o-mini
THINKROUTER_STRONG_COST_PER_1K=0.0006
```

For OpenRouter, SiliconFlow, Qwen/DashScope, or vLLM, set `THINKROUTER_OPENAI_BASE_URL` to that provider's OpenAI-compatible `/v1` endpoint and use the provider model id.

Qwen/DashScope low-cost smoke-test example:

```bash
THINKROUTER_OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
THINKROUTER_STRONG_MODEL=qwen3.5-flash-2026-02-23
THINKROUTER_STRONG_COST_PER_1K=0.000287
```

Then validate and run one real call:

```bash
python -m thinkrouter.experiments.smoke_real_model --model qwen3.5-flash-2026-02-23
python -m thinkrouter.experiments.smoke_real_model --model qwen3.5-flash-2026-02-23 --budget 0 --run
```


Small Qwen budget grid smoke test:

```bash
python -m thinkrouter.experiments.run_grid --input data/splits/seed.jsonl --task gsm8k --split dev --budgets 0,256,1024 --models qwen3.5-flash-2026-02-23 --db results/traces/qwen_gsm8k_dev_budget_grid.sqlite --out results/tables/qwen_gsm8k_dev_budget_grid.csv
python -m thinkrouter.experiments.eval_baselines results/tables/qwen_gsm8k_dev_budget_grid.csv --out results/tables/qwen_gsm8k_dev_budget_summary.csv
python -m thinkrouter.experiments.make_plots results/tables/qwen_gsm8k_dev_budget_grid.csv --out results/figures/qwen_gsm8k_dev_budget_pareto.png
```
## Run Day-1 Grid

```bash
python -m thinkrouter.experiments.run_day1_grid --limit 20 --db results/traces/day1.sqlite --out results/tables/day1_grid.csv
```

This writes the original 20-sample GSM8K-style smoke-test traces to SQLite and CSV.

## Prepare JSONL Data

ThinkRouter uses a simple JSONL benchmark format so official datasets can be converted once and reused by the grid runner:

```json
{"sample_id":"gsm8k_train_001","task_type":"gsm8k","split":"train","query":"...","expected_answer":"..."}
```

Export the built-in seed suite to JSONL:

```bash
python -m thinkrouter.experiments.prepare_data --source seed --task all --split all --out data/splits/seed.jsonl --summary
```

`data/splits/` is ignored by git so local benchmark exports do not get committed accidentally.
Export a small official GSM8K subset using Hugging Face. The default split is 60 train / 20 dev / 20 test:

```bash
python -m thinkrouter.experiments.prepare_data --source gsm8k --out data/splits/gsm8k.jsonl --hf-endpoint https://hf-mirror.com --summary
```

The GSM8K export reads from `openai/gsm8k`, extracts the final numeric answer, and writes the same JSONL schema used by `run_grid --input`.

Export a small official MATH subset using Hugging Face. The default split is 60 train / 20 dev / 20 test:

```bash
python -m thinkrouter.experiments.prepare_data --source math --out data/splits/math.jsonl --hf-endpoint https://hf-mirror.com --summary
```

The MATH export reads from `Maxwell-Jia/MATH`, extracts the final `\boxed{...}` answer from each solution, and writes the standard benchmark JSONL schema.

## Run Frozen Split Grid

The general grid runner supports local frozen seed splits for `gsm8k`, `math`, and `humaneval` tasks, or an external JSONL file via `--input`. The built-in samples are deterministic seed examples for pipeline validation; they are not a substitute for official benchmark results.

Inspect split counts:

```bash
python -m thinkrouter.experiments.run_grid --summary
```

Run train/dev/test grids from built-in seed data:

```bash
python -m thinkrouter.experiments.run_grid --task all --split train --budgets 0,256,1024 --db results/traces/train_grid.sqlite --out results/tables/train_grid.csv
python -m thinkrouter.experiments.run_grid --task all --split dev --budgets 0,256,1024 --db results/traces/dev_grid.sqlite --out results/tables/dev_grid.csv
python -m thinkrouter.experiments.run_grid --task all --split test --budgets 0,256,1024 --db results/traces/test_grid.sqlite --out results/tables/test_grid.csv
```

Run a grid from JSONL input:

```bash
python -m thinkrouter.experiments.run_grid --input data/splits/seed.jsonl --task gsm8k --split dev --budgets 0,256 --models mock-cheap --db results/traces/jsonl_dev.sqlite --out results/tables/jsonl_dev.csv
```

Useful options:

- `--input path/to/samples.jsonl`
- `--task gsm8k|math|humaneval|all`
- `--split train|dev|test|all`
- `--budgets 0,256,1024,4096`
- `--models mock-cheap,mock-strong`
- `--limit N`
- `--resume` to skip sample/model/budget traces already present in `--db` and export all rows from that database

Small official GSM8K Qwen smoke test:

```bash
python -m thinkrouter.experiments.run_grid --input data/splits/gsm8k.jsonl --task gsm8k --split dev --limit 5 --budgets 0,256 --models qwen3.5-flash-2026-02-23 --db results/traces/qwen_gsm8k_official_dev5.sqlite --out results/tables/qwen_gsm8k_official_dev5.csv
python -m thinkrouter.experiments.eval_baselines results/tables/qwen_gsm8k_official_dev5.csv --out results/tables/qwen_gsm8k_official_dev5_summary.csv
python -m thinkrouter.experiments.make_plots results/tables/qwen_gsm8k_official_dev5.csv --out results/figures/qwen_gsm8k_official_dev5_pareto.png
```

Small official MATH Qwen smoke test:

```bash
python -m thinkrouter.experiments.run_grid --input data/splits/math.jsonl --task math --split dev --limit 5 --budgets 0,256 --models qwen3.5-flash-2026-02-23 --db results/traces/qwen_math_official_dev5.sqlite --out results/tables/qwen_math_official_dev5.csv --resume
python -m thinkrouter.experiments.regrade_traces results/tables/qwen_math_official_dev5.csv --out results/tables/qwen_math_official_dev5_regraded.csv
python -m thinkrouter.experiments.eval_baselines results/tables/qwen_math_official_dev5_regraded.csv --out results/tables/qwen_math_official_dev5_summary_regraded.csv
```

Official GSM8K dev20 Qwen budget grid:

```bash
python -m thinkrouter.experiments.run_grid --input data/splits/gsm8k.jsonl --task gsm8k --split dev --budgets 0,256,1024 --models qwen3.5-flash-2026-02-23 --db results/traces/qwen_gsm8k_official_dev20_budget_grid.sqlite --out results/tables/qwen_gsm8k_official_dev20_budget_grid.csv
python -m thinkrouter.experiments.eval_baselines results/tables/qwen_gsm8k_official_dev20_budget_grid.csv --out results/tables/qwen_gsm8k_official_dev20_budget_summary.csv
python -m thinkrouter.experiments.make_plots results/tables/qwen_gsm8k_official_dev20_budget_grid.csv --out results/figures/qwen_gsm8k_official_dev20_budget_pareto.png
```

Official GSM8K train60 Qwen budget grid with resume support:

```bash
python -m thinkrouter.experiments.run_grid --input data/splits/gsm8k.jsonl --task gsm8k --split train --budgets 0,256,1024 --models qwen3.5-flash-2026-02-23 --db results/traces/qwen_gsm8k_official_train60_budget_grid.sqlite --out results/tables/qwen_gsm8k_official_train60_budget_grid.csv --resume
python -m thinkrouter.experiments.regrade_traces results/tables/qwen_gsm8k_official_train60_budget_grid.csv --out results/tables/qwen_gsm8k_official_train60_budget_grid_regraded.csv
python -m thinkrouter.experiments.evaluate_policy results/tables/qwen_gsm8k_official_train60_budget_grid_regraded.csv --out results/tables/qwen_gsm8k_official_train60_policy_summary.csv --stats-out results/tables/qwen_gsm8k_official_train60_policy_stats.csv
```

Regrade an existing grid with the current evaluator:

```bash
python -m thinkrouter.experiments.regrade_traces results/tables/qwen_gsm8k_official_dev20_budget_grid.csv --out results/tables/qwen_gsm8k_official_dev20_budget_grid_regraded.csv
```

Evaluate offline routing policies from an existing grid:

```bash
python -m thinkrouter.experiments.evaluate_policy results/tables/qwen_gsm8k_official_dev20_budget_grid_regraded.csv --out results/tables/qwen_gsm8k_official_dev20_policy_summary.csv --stats-out results/tables/qwen_gsm8k_official_dev20_policy_stats.csv
```

Train and replay a learned policy router. The default evaluator uses the safe policy: it keeps raw classifier predictions for diagnostics, but falls back to the train-split aggregate-utility budget when the learned selector has not established a better policy.

```bash
python -m thinkrouter.experiments.train_learned_policy results/tables/qwen_gsm8k_official_train60_budget_grid_regraded.csv --out results/models/qwen_gsm8k_official_train60_learned_policy.joblib
python -m thinkrouter.experiments.calibrate_learned_policy results/tables/qwen_gsm8k_official_dev20_budget_grid_regraded.csv --model results/models/qwen_gsm8k_official_train60_learned_policy.joblib --out results/models/qwen_gsm8k_official_train60_dev20_calibrated_policy.joblib --summary-out results/tables/qwen_gsm8k_official_dev20_calibration_summary.csv
python -m thinkrouter.experiments.evaluate_learned_policy results/tables/qwen_gsm8k_official_dev20_budget_grid_regraded.csv --model results/models/qwen_gsm8k_official_train60_learned_policy.joblib --out results/tables/qwen_gsm8k_official_train60_to_dev20_safe_policy_summary.csv --selected-out results/tables/qwen_gsm8k_official_train60_to_dev20_safe_policy_selected.csv
python -m thinkrouter.experiments.evaluate_learned_policy results/tables/qwen_gsm8k_official_dev20_budget_grid_regraded.csv --model results/models/qwen_gsm8k_official_train60_learned_policy.joblib --unsafe --out results/tables/qwen_gsm8k_official_train60_to_dev20_raw_learned_policy_summary.csv --selected-out results/tables/qwen_gsm8k_official_train60_to_dev20_raw_learned_policy_selected.csv
python -m thinkrouter.experiments.evaluate_learned_policy results/tables/qwen_gsm8k_official_test20_budget_grid_regraded.csv --model results/models/qwen_gsm8k_official_train60_dev20_calibrated_policy.joblib --out results/tables/qwen_gsm8k_official_train60_dev20_to_test20_calibrated_policy_summary.csv --selected-out results/tables/qwen_gsm8k_official_train60_dev20_to_test20_calibrated_policy_selected.csv
```

Build the consolidated GSM8K report:

```bash
python -m thinkrouter.experiments.make_gsm8k_report
python -m thinkrouter.experiments.make_benchmark_report
```

Failure analysis for a grid CSV:

```bash
python -m thinkrouter.experiments.analyze_failures results/tables/qwen_gsm8k_official_dev20_budget_grid.csv --out results/tables/qwen_gsm8k_official_dev20_failures.csv
```
## Train Router Models

Train on the train split, not dev or test:

```bash
python -m thinkrouter.experiments.train_difficulty results/tables/train_grid.csv --out results/models/difficulty.joblib
python -m thinkrouter.experiments.train_budget results/tables/train_grid.csv --out results/models/budget.joblib
```

To make FastAPI or Streamlit use the trained models, set:

```bash
THINKROUTER_DIFFICULTY_MODEL_PATH=results/models/difficulty.joblib
THINKROUTER_BUDGET_MODEL_PATH=results/models/budget.joblib
```

Without these variables, the router falls back to the built-in heuristic difficulty estimator and utility policy.

## Analysis Scripts

```bash
python -m thinkrouter.experiments.eval_baselines results/tables/dev_grid.csv --out results/tables/dev_baseline_summary.csv
python -m thinkrouter.experiments.make_plots results/tables/dev_grid.csv --out results/figures/dev_pareto.png
```

The current training scripts are lightweight sklearn pipelines over trace CSVs. The later project stage should replace the built-in seed samples with frozen official GSM8K, MATH-500, and HumanEval splits.

## Run API

```bash
uvicorn thinkrouter.app.api:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /config`
- `POST /run`
- `GET /traces`

## Run Streamlit Demo

```bash
streamlit run thinkrouter/ui/streamlit_app.py
```

The demo lets you run a GSM8K-style query, choose a model and budget, optionally use the router, and inspect recent SQLite traces.

## Test

```bash
pytest
```
