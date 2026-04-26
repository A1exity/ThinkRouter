# FINAL_REPORT

This file is the repository-wide closeout summary. It uses the same position as `README.md`, `RESULTS.md`, and `METHOD.md`.

## What The System Is Now

ThinkRouter is now a complete joint routing system with:

- structured budgets
- a three-tier Qwen model pool
- deterministic math and code evaluators
- Phase 2 learned routers
- shared runtime cache and recovery
- an official frozen protocol
- completed official reruns
- a final official benchmark report

The online default route uses the Phase 2 stack. The legacy heuristic policy remains only as a comparison baseline.

## Final Official Protocol

The only official protocol is:

- model pool: `qwen-flash,qwen-plus,qwen-max`
- budgets: `0,256,1024`
- benchmarks: `GSM8K`, `MATH-500`, `HumanEval`
- split sizes: `60 train / 20 dev / 20 test`
- routers: `threshold`, `logreg_joint`, `mlp_factorized`, `uncertainty_aware`
- utility: `accuracy - 5 * cost - 0.02 * latency`
- default online router: `uncertainty_aware`

Implementation paths:

- `configs/official_protocol.json`
- `thinkrouter/official_protocol.py`
- `thinkrouter/experiments/run_official_pipeline.py`
- `scripts/run_official_pipeline.ps1`

## Final Official Results

The final official report is:

- `results/reports/final_official_report.md`

The final top-level assets are:

- `results/tables/final_official_results.csv`
- `results/figures/final_official_pareto.png`
- `results/tables/final_official_failures.csv`

Official summary:

| benchmark | official learned policy | router | utility | learned win |
| --- | --- | --- | ---: | --- |
| GSM8K | `phase2_logreg_joint` | `logreg_joint` | 0.801039 | yes |
| MATH-500 | `phase2_logreg_joint` | `logreg_joint` | -0.115803 | no |
| HumanEval | `phase2_logreg_joint` | `logreg_joint` | 0.217585 | no |

## Acceptance Outcome

The final acceptance conditions are now satisfied:

- documentation wording is unified
- a unique final official report exists
- the learned router beats strongest fixed and aggregate baselines on at least one official benchmark
- HumanEval is part of the main official report
- online default routing is on the Phase 2 router stack

The benchmark that satisfies the learned-router win condition is `GSM8K`.

## Historical Versus Official Results

The repository still contains older slices:

- seed runs
- smoke runs
- `dev5`
- `dev10`
- `dev20`
- old single-model held-out `test20` summaries

Those files are retained for debugging and provenance. They are not part of the final official result story.
