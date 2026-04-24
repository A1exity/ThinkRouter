# ThinkRouter

ThinkRouter is a routing system for reasoning workloads where the decision is `(model, budget)`, not just `model`. The current codebase supports multi-model Qwen routing, structured budgets, deterministic evaluators, offline replay, Phase 2 learned routers, shared runtime cache/recovery, and an official protocol pipeline.

## Current System

- Model pool: `qwen-flash`, `qwen-plus`, `qwen-max`
- Budget set: `0`, `256`, `1024`
- Benchmarks in the frozen official protocol:
  - `GSM8K`
  - `MATH-500`
  - `HumanEval`
- Train/dev/test subset size per benchmark: `60 / 20 / 20`
- Learned router stack:
  - `threshold`
  - `logreg_joint`
  - `mlp_factorized`
  - `uncertainty_aware`
- Online default router:
  - `uncertainty_aware`
- Semantic feature backend:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - lexical SVD fallback is only a local resilience path when the embedding model cannot be loaded

## Current Main Committed Results

The main committed multi-model reference result is still the historical Qwen pool GSM8K slice at [`results/qwen35_pool_gsm8k_dev20_phase2_pareto.png`](results/qwen35_pool_gsm8k_dev20_phase2_pareto.png) and [`results/reports/qwen35_pool_gsm8k_dev20_phase2_report.md`](results/reports/qwen35_pool_gsm8k_dev20_phase2_report.md).

On that committed `dev20` slice:

| benchmark | policy | accuracy | avg cost | p95 latency | utility |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K dev20 | `qwen-max @ budget 0` strongest fixed point | 1.000 | 0.000849 | 8.263 | 0.904947 |
| GSM8K dev20 | Phase 2 learned routers | 0.950 | 0.000246 | 6.744 | 0.813895 |

Current interpretation:

- The system path is complete enough to run multi-model joint routing.
- The learned router stack is operational, but the committed historical GSM8K slice does **not** yet show a learned-router win over the strongest fixed baseline.
- The committed HumanEval slice is still only a wiring proof, not a final code-task result line.

## What Is Not Yet Completed

The repository now contains the frozen official protocol and one-command official pipeline, but the **final official rerun** has not yet been committed:

- no committed `final_official_results.csv`
- no committed `final_official_pareto.png`
- no committed `results/reports/final_official_report.md` generated from official protocol outputs
- no committed official `HumanEval` result line at the full `60 / 20 / 20` protocol scale

So the project status is:

- system architecture: implemented
- official protocol: frozen
- online default router: switched to Phase 2 stack
- real semantic features: implemented
- official multi-benchmark rerun: pending

## Frozen Official Protocol

The only formal protocol is defined in:

- [`configs/official_protocol.json`](configs/official_protocol.json)
- [`thinkrouter/official_protocol.py`](thinkrouter/official_protocol.py)

Formal settings:

| field | value |
| --- | --- |
| model pool | `qwen-flash,qwen-plus,qwen-max` |
| budgets | `0,256,1024` |
| benchmarks | `gsm8k`, `math500`, `humaneval` |
| split sizes | `60 train / 20 dev / 20 test` |
| baselines | `fixed_model_budget`, `model_only`, `budget_only`, `joint_aggregate_utility`, `joint_safe_fallback` |
| routers | `threshold`, `logreg_joint`, `mlp_factorized`, `uncertainty_aware` |
| utility | `1.0 * accuracy - 5.0 * cost - 0.02 * latency` |
| default online router | `uncertainty_aware` |

Historical smoke/dev slices remain in the repo, but they are appendix artifacts only.

## Official Pipeline

The official command chain is fixed:

```powershell
.\scripts\run_official_pipeline.ps1
```

Equivalent stages:

```bash
python -m thinkrouter.experiments.run_official_pipeline --stage prepare-data
python -m thinkrouter.experiments.run_official_pipeline --stage grids
python -m thinkrouter.experiments.run_official_pipeline --stage routers
python -m thinkrouter.experiments.run_official_pipeline --stage report
```

This pipeline is the only supported path for future formal results.

## Setup

```bash
conda env create -f environment.yml
conda activate thinkrouter
pytest
```

Environment variables for the official Qwen pool:

```text
THINKROUTER_MODEL_POOL=qwen-flash,qwen-plus,qwen-max
THINKROUTER_QWEN_FLASH_MODEL=qwen3.5-flash-2026-02-23
THINKROUTER_QWEN_PLUS_MODEL=qwen3.5-plus-2026-02-15
THINKROUTER_QWEN_MAX_MODEL=qwen3-max-2026-01-23
```

## Key Paths

- [`METHOD.md`](METHOD.md): current method and protocol definition
- [`RESULTS.md`](RESULTS.md): current result inventory and appendix/historical index
- [`FINAL_REPORT.md`](FINAL_REPORT.md): current repository-wide status summary
- [`results/reports/`](results/reports): generated report artifacts
- [`results/official/`](results/official): official protocol outputs after reruns
