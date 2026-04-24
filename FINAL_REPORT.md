# FINAL_REPORT

This file is the repository-wide status summary. It uses the same position as [`README.md`](README.md), [`RESULTS.md`](RESULTS.md), and [`METHOD.md`](METHOD.md).

## What The System Is Now

ThinkRouter is now a joint routing system with:

- structured budgets
- a three-tier Qwen model pool
- deterministic math and code evaluators
- Phase 2 learned routers
- shared runtime cache and recovery
- an official frozen protocol
- a one-command official pipeline

The online default route has been moved to the Phase 2 stack. The legacy heuristic policy remains only as a baseline path.

## Frozen Official Protocol

The only official protocol is:

- model pool: `qwen-flash,qwen-plus,qwen-max`
- budgets: `0,256,1024`
- benchmarks: `GSM8K`, `MATH-500`, `HumanEval`
- split sizes: `60 train / 20 dev / 20 test`
- routers: `threshold`, `logreg_joint`, `mlp_factorized`, `uncertainty_aware`
- utility: `accuracy - 5 * cost - 0.02 * latency`

Implementation paths:

- [`configs/official_protocol.json`](configs/official_protocol.json)
- [`thinkrouter/official_protocol.py`](thinkrouter/official_protocol.py)
- [`thinkrouter/experiments/run_official_pipeline.py`](thinkrouter/experiments/run_official_pipeline.py)
- [`scripts/run_official_pipeline.ps1`](scripts/run_official_pipeline.ps1)

## Current Main Committed Results

The main committed reference result is still the historical Qwen pool GSM8K `dev20` slice:

| benchmark | policy | accuracy | avg cost | p95 latency | utility |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K dev20 | strongest fixed: `qwen-max @ 0` | 1.000 | 0.000849 | 8.263 | 0.904947 |
| GSM8K dev20 | learned routers | 0.950 | 0.000246 | 14.738 | 0.813895 |

Current conclusion from the committed artifacts:

- the multi-model routing path works
- the learned router stack is fully wired
- the committed historical slice still favors the strongest fixed baseline on utility

The committed HumanEval multi-model slice is still only a wiring proof and should not be treated as the final code-task conclusion.

## What Is Still Not Committed

The repository still does **not** contain the final official rerun outputs:

- no committed `results/tables/final_official_results.csv`
- no committed `results/figures/final_official_pareto.png`
- no committed `results/tables/final_official_failures.csv`
- no committed `results/reports/final_official_report.md` generated from official protocol outputs

This means the project has completed the system refactor and protocol freeze, but the final official experiment run remains to be executed and committed.

## Historical Versus Official Results

The repo contains many older slices:

- seed runs
- smoke runs
- `dev5`
- `dev10`
- `dev20`
- old single-model held-out `test20` summaries

Those files are retained for debugging and provenance. They are no longer the official reporting target.

The official reporting target is only the protocol-driven output under:

- `results/official/`
- `results/tables/final_official_results.csv`
- `results/figures/final_official_pareto.png`
- `results/tables/final_official_failures.csv`
- `results/reports/final_official_report.md`
