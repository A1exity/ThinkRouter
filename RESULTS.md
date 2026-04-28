# RESULTS

This file is the result inventory and closeout summary for the completed repository state. It uses the same narrative as `README.md` and `METHOD.md`.

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

## Final Conclusion

ThinkRouter is now closed out as a model-and-budget routing system with a frozen official protocol, reproducible official reruns, deterministic math/code evaluation, and a Phase 2 router stack as the online default path.

## Acceptance Condition

The closeout acceptance conditions are satisfied:

- the documentation wording is unified around the official protocol
- the public repository keeps a single final official report plus official result assets
- the learned router beats strongest fixed and aggregate baselines on at least one official benchmark
- HumanEval is included as a formal benchmark line
- the online default route uses the Phase 2 router stack

The benchmark that satisfies the learned-router win condition is `GSM8K`.

## Negative Controls

`MATH-500` and `HumanEval` remain important negative controls. They confirm that the system can run the full official pipeline across multiple reasoning tasks even when the learned router is not the utility-optimal policy on those benchmarks.

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

## Historical Note

Historical smoke, `dev*`, and old `test20` assets have been pruned from the public repository. The public result tree now keeps only the official protocol outputs and the final official report.
