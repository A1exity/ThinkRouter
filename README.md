# ThinkRouter

Adaptive thinking-budget routing for reasoning LLMs.

ThinkRouter treats `(model, thinking_budget)` as the routing decision. It records every run as a trace with correctness, tokens, estimated cost, latency, model id, budget, and benchmark metadata, then evaluates fixed-budget, oracle, aggregate-utility, and learned routing policies offline.

## Highlights

- OpenAI-compatible model adapter with mock-model fallback for zero-cost local testing.
- Ordered model-pool config with legacy tier vars and built-in Qwen `flash -> plus -> max` aliases.
- Resumable benchmark grid runner for interrupted API experiments.
- Official GSM8K and MATH JSONL export pipelines.
- Deterministic GSM8K and MATH evaluators with regrading support.
- Offline policy replay for fixed budget, oracle, aggregate utility, raw learned, safe fallback, and dev-calibrated policies.
- Consolidated report generation for GSM8K and multi-benchmark results.

## Key Results

Held-out Qwen `test20` results from committed traces:

| benchmark | policy | accuracy | avg cost | p95 latency | cost vs 1024 |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K | fixed budget 1024 | 0.950 | 0.000542 | 23.390s | 100.0% |
| GSM8K | dev-calibrated safe policy | 0.950 | 0.000344 | 16.721s | 63.5% |
| MATH | fixed budget 0 | 0.500 | 0.000604 | 32.348s | 59.1% |
| MATH | fixed budget 256 | 0.550 | 0.000876 | 54.652s | 85.7% |
| MATH | fixed budget 1024 | 0.250 | 0.001022 | 49.151s | 100.0% |

Main takeaway: budget is a real routing variable, but larger budget is not automatically better. On GSM8K, the calibrated safe policy matched the high-budget baseline at lower cost. On MATH, budget behavior was less stable and `1024` performed poorly.

See [FINAL_REPORT.md](FINAL_REPORT.md) for the full technical report and [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for a concise project overview.

## Architecture

```text
official dataset
  -> JSONL benchmark split
  -> run_grid.py
  -> model adapter
  -> evaluator
  -> SQLite / CSV traces
  -> regrade + failure analysis
  -> policy replay
  -> reports
```

Core files:

| path | purpose |
| --- | --- |
| `thinkrouter/adapters/` | mock/OpenAI-compatible adapters and model-pool config parsing |
| `thinkrouter/app/models.py` | compatibility exports for adapter/model config entrypoints |
| `thinkrouter/app/evaluators.py` | GSM8K, MATH, and exact-match evaluators |
| `thinkrouter/app/store.py` | SQLite trace persistence |
| `thinkrouter/app/router.py` | query features and online routing policy |
| `thinkrouter/experiments/run_grid.py` | resumable budget-grid runner |
| `thinkrouter/experiments/prepare_data.py` | seed, GSM8K, and MATH JSONL exporters |
| `thinkrouter/experiments/evaluate_policy.py` | fixed/oracle/aggregate policy replay |
| `thinkrouter/experiments/learned_policy_router.py` | learned budget selector and safe fallback |
| `thinkrouter/experiments/make_benchmark_report.py` | multi-benchmark report generation |

## Quick Start

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate thinkrouter
```

Run the test suite:

```bash
pytest
```

Run a zero-cost mock grid:

```bash
python -m thinkrouter.experiments.run_day1_grid --limit 20 --db results/traces/day1.sqlite --out results/tables/day1_grid.csv
python -m thinkrouter.experiments.eval_baselines results/tables/day1_grid.csv --out results/tables/baseline_summary.csv
python -m thinkrouter.experiments.make_plots results/tables/day1_grid.csv --out results/figures/pareto.png
```

Generate consolidated reports from committed CSV artifacts:

```bash
python -m thinkrouter.experiments.make_gsm8k_report
python -m thinkrouter.experiments.make_benchmark_report
```

Report outputs:

- `results/reports/qwen_gsm8k_final_policy_report.md`
- `results/reports/qwen_multi_benchmark_policy_report.md`
- `results/tables/qwen_multi_benchmark_policy_report.csv`
- `results/figures/qwen_gsm8k_test20_policy_comparison.png`

## Configuration

By default, the project works with mock models and does not require API keys.

For a real OpenAI-compatible endpoint, copy the template and fill in provider settings:

```powershell
copy .env.example .env
```

Important environment variables:

| variable | description |
| --- | --- |
| `THINKROUTER_OPENAI_BASE_URL` | OpenAI-compatible `/v1` endpoint |
| `THINKROUTER_OPENAI_API_KEY` | provider API key |
| `THINKROUTER_MODEL_POOL` | ordered candidate models, for example `qwen-flash,qwen-plus,qwen-max` |
| `THINKROUTER_STRONG_MODEL` | real model id, for example `qwen3.5-flash-2026-02-23` |
| `THINKROUTER_STRONG_COST_PER_1K` | estimated cost per 1K tokens |
| `THINKROUTER_DB_PATH` | default SQLite trace database path |
| `THINKROUTER_DIFFICULTY_MODEL_PATH` | optional sklearn difficulty model |
| `THINKROUTER_BUDGET_MODEL_PATH` | optional sklearn budget model |

Qwen pool support:

```text
THINKROUTER_MODEL_POOL=qwen-flash,qwen-plus,qwen-max
THINKROUTER_QWEN_FLASH_MODEL=qwen3.5-flash-2026-02-23
THINKROUTER_QWEN_PLUS_MODEL=qwen3.5-plus-2026-02-23
THINKROUTER_QWEN_MAX_MODEL=qwen3.5-max-2026-02-23
```

If `THINKROUTER_MODEL_POOL` is unset, the project keeps the legacy `THINKROUTER_CHEAP_MODEL` / `THINKROUTER_MID_MODEL` / `THINKROUTER_STRONG_MODEL` behavior.

Validate the strongest configured model in the pool without making a network call:

```bash
python -m thinkrouter.experiments.smoke_real_model --model qwen3.5-flash-2026-02-23
```

Run one real call:

```bash
python -m thinkrouter.experiments.smoke_real_model --model qwen3.5-flash-2026-02-23 --budget 0 --run
```

## Benchmark Data

ThinkRouter uses one JSON object per line:

```json
{"sample_id":"gsm8k_train_001","task_type":"gsm8k","split":"train","query":"...","expected_answer":"..."}
```

Export built-in seed data:

```bash
python -m thinkrouter.experiments.prepare_data --source seed --task all --split all --out data/splits/seed.jsonl --summary
```

Export official GSM8K and MATH subsets:

```bash
python -m thinkrouter.experiments.prepare_data --source gsm8k --out data/splits/gsm8k.jsonl --hf-endpoint https://hf-mirror.com --summary
python -m thinkrouter.experiments.prepare_data --source math --out data/splits/math.jsonl --hf-endpoint https://hf-mirror.com --summary
```

Default official subset sizes are `60 train / 20 dev / 20 test`. `data/splits/` is ignored by git.

## Running Experiments

Run a resumable grid from JSONL:

```bash
python -m thinkrouter.experiments.run_grid --input data/splits/gsm8k.jsonl --task gsm8k --split dev --budgets 0,256,1024 --models qwen3.5-flash-2026-02-23 --db results/traces/qwen_gsm8k_official_dev20_budget_grid.sqlite --out results/tables/qwen_gsm8k_official_dev20_budget_grid.csv --resume
```

Analyze a completed grid:

```bash
python -m thinkrouter.experiments.regrade_traces results/tables/qwen_gsm8k_official_dev20_budget_grid.csv --out results/tables/qwen_gsm8k_official_dev20_budget_grid_regraded.csv
python -m thinkrouter.experiments.eval_baselines results/tables/qwen_gsm8k_official_dev20_budget_grid_regraded.csv --out results/tables/qwen_gsm8k_official_dev20_budget_summary_regraded.csv
python -m thinkrouter.experiments.evaluate_policy results/tables/qwen_gsm8k_official_dev20_budget_grid_regraded.csv --out results/tables/qwen_gsm8k_official_dev20_policy_summary.csv --stats-out results/tables/qwen_gsm8k_official_dev20_policy_stats.csv
python -m thinkrouter.experiments.analyze_failures results/tables/qwen_gsm8k_official_dev20_budget_grid_regraded.csv --out results/tables/qwen_gsm8k_official_dev20_failures_regraded.csv
```

Useful `run_grid` options:

- `--input path/to/samples.jsonl`
- `--task gsm8k|math|humaneval|all`
- `--split train|dev|test|all`
- `--budgets 0,256,1024,4096`
- `--models mock-cheap,mock-strong,<real-model-id>`
- `--limit N`
- `--resume` to skip sample/model/budget traces already present in `--db`

## Learned Policy Workflow

Train on train, calibrate on dev, evaluate on held-out test:

```bash
python -m thinkrouter.experiments.train_learned_policy results/tables/qwen_gsm8k_official_train60_budget_grid_regraded.csv --out results/models/qwen_gsm8k_official_train60_learned_policy.joblib
python -m thinkrouter.experiments.calibrate_learned_policy results/tables/qwen_gsm8k_official_dev20_budget_grid_regraded.csv --model results/models/qwen_gsm8k_official_train60_learned_policy.joblib --out results/models/qwen_gsm8k_official_train60_dev20_calibrated_policy.joblib --summary-out results/tables/qwen_gsm8k_official_dev20_calibration_summary.csv
python -m thinkrouter.experiments.evaluate_learned_policy results/tables/qwen_gsm8k_official_test20_budget_grid_regraded.csv --model results/models/qwen_gsm8k_official_train60_dev20_calibrated_policy.joblib --out results/tables/qwen_gsm8k_official_train60_dev20_to_test20_calibrated_policy_summary.csv --selected-out results/tables/qwen_gsm8k_official_train60_dev20_to_test20_calibrated_policy_selected.csv
```

The default learned-policy evaluator uses safe fallback. Add `--unsafe` to replay raw classifier predictions.

## API and Demo

Run FastAPI:

```bash
uvicorn thinkrouter.app.api:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /config`
- `POST /run`
- `GET /traces`

Run Streamlit:

```bash
streamlit run thinkrouter/ui/streamlit_app.py
```

## Repository Notes

- `.env` is ignored and must not be committed.
- `results/traces/` is ignored because SQLite traces can grow and may contain raw outputs.
- `data/splits/` is ignored because benchmark exports and Hugging Face caches are local artifacts.
- Committed `results/tables/`, `results/figures/`, and `results/reports/` files are small reproducibility artifacts.

## Limitations

- Current real-model experiments use one provider model.
- Official subsets are small, not full benchmark-scale runs.
- Thinking budgets are prompt-level instructions, not guaranteed internal reasoning-token controls.
- MATH evaluation is deterministic but approximate; it does not perform full symbolic equivalence.
