# ThinkRouter Project Brief

## One-Liner

ThinkRouter is a complete experimental routing system for reasoning LLMs that jointly selects `model + thinking_budget` and measures the resulting quality, cost, and latency tradeoffs.

## Resume Version

Built ThinkRouter, a full-stack LLM routing system with provider adapters, structured budgets, deterministic benchmark evaluators, SQLite traces, offline replay, learned and uncertainty-aware routers, runtime cache/recovery, analysis dashboards, and reproducible reporting.

## What It Demonstrates

- LLM systems engineering from provider call to trace store to report
- joint routing over both model and budget rather than budget-only tuning
- deterministic evaluation and offline replay discipline
- practical runtime hardening: retries, request cache, resumable experiments, failure recording
- research workflow discipline: baselines, closeout summaries, failure taxonomy, stability summaries, reproducibility scripts

## Architecture

```text
benchmark export
  -> grid collection
  -> trace store
  -> evaluator
  -> replay / ranking / plotting / failure taxonomy / stability summaries
  -> reports and UI inspection
```

Core modules:

- `thinkrouter/adapters/`: provider abstraction and model-pool parsing
- `thinkrouter/runtime/`: shared cache and recovery
- `thinkrouter/features/`: routing features
- `thinkrouter/routers/`: routing policies
- `thinkrouter/analytics/`: cost/latency/failure/stability summaries
- `thinkrouter/experiments/`: grid runs, reports, closeout scripts
- `thinkrouter/ui/`: dashboard, failure browser, route inspector

## Key Results

Completed real Qwen pool Phase 2 slice on GSM8K `dev20`:

| policy | accuracy | avg cost | avg latency |
| --- | ---: | ---: | ---: |
| `phase2_threshold` | 0.950 | 0.000246 | 6.744s |
| `phase2_logreg_joint` | 0.950 | 0.000246 | 6.744s |
| `phase2_mlp_factorized` | 0.950 | 0.000246 | 6.744s |
| `phase2_uncertainty_aware` | 0.950 | 0.000246 | 6.744s |

Current strongest utility point on that slice is still `qwen-max @ budget 0`, which shows the system is complete enough to measure honest non-winning results rather than only positive ones.

## Project State

The implementation roadmap is complete:

- Phase 1 complete
- Phase 2 complete
- Phase 3 complete
- Phase 4 complete

Anything further would be a new extension, not unfinished work.

## Interview Talking Points

- Why routing on budget alone is incomplete without model choice
- Why deterministic evaluators matter for cost-quality comparisons
- Why offline replay is such a strong systems lever
- Why runtime cache/recovery matters when real provider experiments are unstable
- Why a complete experiment repo should include failure taxonomy and reproducibility assets, not just one headline plot

## Reproduce

```bash
pytest
python -m thinkrouter.experiments.run_eval results/tables/qwen35_pool_gsm8k_dev20_grid.csv --out-prefix results/eval/qwen35_pool_gsm8k_dev20 --phase2-router threshold --phase2-router logreg_joint=results/qwen35_pool_gsm8k_dev20_logreg_joint.joblib --phase2-router mlp_factorized=results/qwen35_pool_gsm8k_dev20_mlp_factorized.joblib --phase2-router uncertainty_aware=results/qwen35_pool_gsm8k_dev20_mlp_factorized.joblib
```
