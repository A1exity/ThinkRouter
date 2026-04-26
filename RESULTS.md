# RESULTS

This file is the result inventory for the completed repository state. It uses the same narrative as `README.md`, `FINAL_REPORT.md`, and `METHOD.md`.

## Final Status

ThinkRouter now has:

- a frozen official experiment protocol
- a Phase 2 router stack as the default online routing path
- real sentence-transformer semantic features in the learned-router pipeline
- deterministic evaluators for math and code
- a single official command chain
- completed official reruns for `GSM8K`, `MATH-500`, and `HumanEval`
- a final official report and final top-level assets

## Final Official Outputs

The main official outputs are:

- `results/tables/final_official_results.csv`
- `results/figures/final_official_pareto.png`
- `results/tables/final_official_failures.csv`
- `results/reports/final_official_report.md`

Per-benchmark official outputs are under:

- `results/official/gsm8k/`
- `results/official/math500/`
- `results/official/humaneval/`

## Final Official Summary

| benchmark | official learned policy | router | accuracy | avg cost | p95 latency | utility | beats strongest fixed | beats aggregate |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| GSM8K | `phase2_logreg_joint` | `logreg_joint` | 0.95 | 0.000186 | 7.401475 | 0.801039 | yes | yes |
| MATH-500 | `phase2_logreg_joint` | `logreg_joint` | 0.25 | 0.000155 | 18.251530 | -0.115803 | no | no |
| HumanEval | `phase2_logreg_joint` | `logreg_joint` | 0.60 | 0.000377 | 19.026367 | 0.217585 | no | no |

Final interpretation:

- `GSM8K` is the benchmark where the learned router clearly wins under the official protocol.
- `MATH-500` and `HumanEval` do not support the same learned-router win claim.
- The repository therefore contains both positive and negative official outcomes rather than a single benchmark-specific success story.

## Frozen Official Protocol

The only official protocol is:

- config: `configs/official_protocol.json`
- code: `thinkrouter/official_protocol.py`
- pipeline entrypoint: `thinkrouter/experiments/run_official_pipeline.py`
- one-command wrapper: `scripts/run_official_pipeline.ps1`

Formal settings:

| field | value |
| --- | --- |
| model pool | `qwen-flash,qwen-plus,qwen-max` |
| budgets | `0,256,1024` |
| benchmarks | `GSM8K`, `MATH-500`, `HumanEval` |
| split sizes | `60 train / 20 dev / 20 test` |
| routers | `threshold`, `logreg_joint`, `mlp_factorized`, `uncertainty_aware` |
| utility | `accuracy - 5 * cost - 0.02 * latency` |
| online default router | `uncertainty_aware` |

## HumanEval Status

HumanEval is now a completed official result line, not just a wiring proof.

Official artifacts include:

- `results/official/humaneval/humaneval_router_selection.csv`
- `results/official/humaneval/humaneval_test_integrated_summary.csv`

On the official `test` split, the learned router does not beat the strongest fixed or aggregate baseline, but the benchmark is fully represented in the final official report.

## Historical Appendix

The following remain historical or debugging artifacts only:

- Day-1 seed runs
- smoke runs
- `dev5`, `dev10`, `dev20` slices
- old single-model held-out `test20` summaries
- old `qwen_gsm8k_final_policy_report` / `qwen_multi_benchmark_policy_report` assets

They remain useful for debugging and provenance, but they are appendix-only.
