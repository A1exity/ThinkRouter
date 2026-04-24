# RESULTS

This file is the result inventory for the current repository state. It uses the same narrative as [`README.md`](README.md), [`FINAL_REPORT.md`](FINAL_REPORT.md), and [`METHOD.md`](METHOD.md).

## Current Status

ThinkRouter now has:

- a frozen official experiment protocol
- a Phase 2 router stack as the default online routing path
- real sentence-transformer semantic features in the learned-router pipeline
- deterministic evaluators for math and code
- a single official command chain for future formal reruns

What it does **not** yet have in the committed artifacts is the final official multi-benchmark rerun.

That means the repository is in this state:

- system implementation: current
- protocol definition: frozen
- official rerun outputs: not yet committed

## Frozen Official Protocol

The only official protocol is:

- config: [`configs/official_protocol.json`](configs/official_protocol.json)
- code: [`thinkrouter/official_protocol.py`](thinkrouter/official_protocol.py)
- pipeline entrypoint: [`thinkrouter/experiments/run_official_pipeline.py`](thinkrouter/experiments/run_official_pipeline.py)
- one-command wrapper: [`scripts/run_official_pipeline.ps1`](scripts/run_official_pipeline.ps1)

Formal settings:

| field | value |
| --- | --- |
| model pool | `qwen-flash,qwen-plus,qwen-max` |
| budgets | `0,256,1024` |
| benchmarks | `gsm8k`, `math500`, `humaneval` |
| split sizes | `60 train / 20 dev / 20 test` |
| learned routers | `logreg_joint`, `mlp_factorized`, `uncertainty_aware` |
| explanation baseline | `threshold` |
| utility | `accuracy - 5 * cost - 0.02 * latency` |
| online default router | `uncertainty_aware` |

## Current Main Committed Reference Results

The main committed multi-model reference slice is still the historical Qwen pool GSM8K `dev20` run:

- summary: [`results/qwen35_pool_gsm8k_dev20_baseline_phase2_summary.csv`](results/qwen35_pool_gsm8k_dev20_baseline_phase2_summary.csv)
- plot: [`results/qwen35_pool_gsm8k_dev20_phase2_pareto.png`](results/qwen35_pool_gsm8k_dev20_phase2_pareto.png)
- report: [`results/reports/qwen35_pool_gsm8k_dev20_phase2_report.md`](results/reports/qwen35_pool_gsm8k_dev20_phase2_report.md)

Reference rows from that committed slice:

| benchmark | policy | accuracy | avg cost | p95 latency | utility |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K dev20 | strongest fixed: `qwen-max @ 0` | 1.000 | 0.000849 | 8.263 | 0.904947 |
| GSM8K dev20 | learned routers | 0.950 | 0.000246 | 14.738 | 0.813895 |

Current interpretation:

- the learned router stack is operational and routed to the cheap flash tier on this slice
- the strongest fixed baseline is still better on utility in the committed reference run
- the repository therefore cannot yet claim an official learned-router win

## HumanEval Status

HumanEval is now part of the frozen official protocol and the exporter exists:

- official split export path: [`data/splits/official_humaneval.jsonl`](data/splits/official_humaneval.jsonl)

The currently committed multi-model code-task result is still the historical small slice:

- grid: [`results/tables/qwen35_pool_humaneval_dev2_budget256_grid.csv`](results/tables/qwen35_pool_humaneval_dev2_budget256_grid.csv)
- replay report: [`results/reports/qwen35_pool_humaneval_dev2_budget256_phase2_report.md`](results/reports/qwen35_pool_humaneval_dev2_budget256_phase2_report.md)

That slice is only a wiring proof. It is not the final official HumanEval result line.

## Official Outputs Expected From The Frozen Pipeline

When the official rerun is executed and committed, the main outputs will be:

- `results/tables/final_official_results.csv`
- `results/figures/final_official_pareto.png`
- `results/tables/final_official_failures.csv`
- `results/reports/final_official_report.md`

Per-benchmark official outputs will be written under:

- `results/official/gsm8k/`
- `results/official/math500/`
- `results/official/humaneval/`

## Historical Appendix

The following are historical or debugging artifacts and are no longer part of the main result story:

- Day-1 seed runs
- smoke runs
- `dev5`, `dev10`, `dev20` slices
- old single-model held-out `test20` summaries
- old `qwen_gsm8k_final_policy_report` / `qwen_multi_benchmark_policy_report` assets

They remain useful for debugging and provenance, but they are appendix-only.
