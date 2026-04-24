# ThinkRouter Final Technical Report

## Summary

ThinkRouter is now a complete joint `model + thinking_budget` routing system repository. It includes:

- real and mock model adapters
- structured budgets
- versioned traces
- deterministic evaluators
- offline replay and policy comparison
- Phase 2 routers with semantic and cheap-probe features
- Phase 3 analytics, UI inspection, unified evaluation CLI, and runtime cache/recovery
- Phase 4 closeout assets: ablation summaries, failure taxonomy reports, stability summaries, reproducibility docs, configs, and scripts

The repository is no longer just a budget-routing prototype. It is a full experiment and analysis system for verifiable reasoning tasks.

## Scope

Implemented task families:

- GSM8K-style math reasoning
- MATH-style symbolic/numeric reasoning
- HumanEval-style code generation

Implemented routing families:

- fixed model/budget baselines
- model-only and budget-only baselines
- aggregate utility joint baseline
- safe fallback joint baseline
- learned budget policy with calibration
- `threshold`
- `logreg_joint`
- `mlp_factorized`
- `uncertainty_aware`

## System

Current pipeline:

```text
dataset export
  -> structured benchmark JSONL
  -> run_grid / run_eval / run_phase2_eval
  -> provider or mock adapter
  -> deterministic evaluator
  -> SQLite trace + CSV artifacts
  -> replay / ranking / plotting / failure taxonomy / stability summaries
  -> reports
```

Major subsystems:

- `thinkrouter/adapters/`: provider abstraction and model-pool parsing
- `thinkrouter/runtime/`: shared request cache, retry behavior, and failure recovery
- `thinkrouter/features/`: surface, semantic-hash, and cheap-probe features
- `thinkrouter/routers/`: threshold, joint-logreg, factorized MLP, uncertainty-aware
- `thinkrouter/analytics/`: cost, latency, failures, stability
- `thinkrouter/experiments/`: data prep, grid runs, replay, reports, closeout summaries
- `thinkrouter/ui/`: Streamlit run/demo, dashboard, failure browser, route inspector

## Real Results

### Held-Out Legacy Qwen Results

Held-out test reporting from the earlier single-model Qwen path remains:

| benchmark | policy | accuracy | avg cost | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| GSM8K test20 | fixed budget 256 | 0.950 | 0.000344 | 16.721s |
| GSM8K test20 | fixed budget 1024 | 0.950 | 0.000542 | 23.390s |
| GSM8K test20 | dev-calibrated safe policy | 0.950 | 0.000344 | 16.721s |
| MATH test20 | fixed budget 0 | 0.500 | 0.000604 | 32.348s |
| MATH test20 | fixed budget 256 | 0.550 | 0.000876 | 54.652s |
| MATH test20 | fixed budget 1024 | 0.250 | 0.001022 | 49.151s |

Main interpretation: budget is a real decision variable, but larger budget is not reliably better.

### Real Qwen Pool Phase 2 Results

The strongest completed joint-pool Phase 2 slice in the repository is `qwen35_pool_gsm8k_dev20`.

Candidate trace summary:

- models: `qwen-flash`, `qwen-plus`, `qwen-max`
- budgets: `0`, `256`, `1024`
- traces: `180`

Phase 2 router replay summary on that slice:

| policy | accuracy | avg cost | avg latency | avg route confidence |
| --- | ---: | ---: | ---: | ---: |
| `phase2_threshold` | 0.950 | 0.000246 | 6.744s | 0.6381 |
| `phase2_logreg_joint` | 0.950 | 0.000246 | 6.744s | 0.7785 |
| `phase2_mlp_factorized` | 0.950 | 0.000246 | 6.744s | 0.9873 |
| `phase2_uncertainty_aware` | 0.950 | 0.000246 | 6.744s | 0.9873 |

The strongest utility winner on that slice is still `qwen-max @ budget 0`. That is the current empirical outcome, not a missing implementation.

### Code Task Status

The repository also contains a real code-task Qwen pool slice and a full Phase 2 offline replay on that slice. The code-task path is fully integrated through:

- structured budgets
- deterministic code evaluator
- feature extraction
- factorized and joint replay
- failure analysis
- integrated reporting

On the committed HumanEval slice, all policies remain incorrect. The code-task machinery is complete; the model performance is simply weak on that tiny slice.

## Analytics And Closeout Assets

The repository now includes:

- integrated Phase 2 ranking reports
- Phase 4 ablation summaries
- failure taxonomy reports
- stability summaries with bootstrap-style intervals
- reproducibility docs in `docs/`
- runnable scripts in `scripts/`
- standard config in `configs/`

Key closeout artifacts:

- `results/reports/qwen35_pool_phase2_closeout.md`
- `results/reports/qwen35_pool_gsm8k_dev20_phase2_report.md`
- `results/reports/qwen35_pool_humaneval_dev2_budget256_phase2_report.md`
- `results/reports/qwen35_pool_phase4_ablation.md`
- `results/reports/qwen35_pool_gsm8k_dev20_failure_taxonomy.md`

## Final Status

All planned repository work for Phases 1 through 4 has been completed within the current project scope:

- Phase 1: complete
- Phase 2: complete
- Phase 3: complete
- Phase 4: complete

What remains in the future would be new research work, larger experiments, or new benchmark/model additions, not unfinished implementation in the current roadmap.

## Reproduction

Main commands:

```bash
pytest
python -m thinkrouter.experiments.run_eval results/tables/qwen35_pool_gsm8k_dev20_grid.csv --out-prefix results/eval/qwen35_pool_gsm8k_dev20 --phase2-router threshold --phase2-router logreg_joint=results/qwen35_pool_gsm8k_dev20_logreg_joint.joblib --phase2-router mlp_factorized=results/qwen35_pool_gsm8k_dev20_mlp_factorized.joblib --phase2-router uncertainty_aware=results/qwen35_pool_gsm8k_dev20_mlp_factorized.joblib
python -m thinkrouter.experiments.make_ablation_report results/qwen35_pool_gsm8k_dev10_baseline_phase2_summary.csv results/qwen35_pool_gsm8k_dev20_baseline_phase2_summary.csv results/qwen35_pool_humaneval_dev2_budget256_phase2_baseline_phase2_summary.csv --summary-out results/tables/qwen35_pool_phase4_ablation.csv --markdown-out results/reports/qwen35_pool_phase4_ablation.md
python -m thinkrouter.experiments.make_failure_taxonomy results/tables/qwen35_pool_gsm8k_dev20_grid.csv --summary-out results/tables/qwen35_pool_gsm8k_dev20_failure_taxonomy.csv --markdown-out results/reports/qwen35_pool_gsm8k_dev20_failure_taxonomy.md
```
