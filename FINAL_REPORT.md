# ThinkRouter Final Technical Report

## Summary

ThinkRouter studies whether reasoning LLM inference should route not only across models, but also across discrete thinking budgets. The project implements a full traceable pipeline for `(model, budget)` experiments: official benchmark export, OpenAI-compatible inference, deterministic evaluation, SQLite trace storage, policy replay, learned budget selection, calibration, failure analysis, and report generation.

The current real-model experiments use `qwen3.5-flash-2026-02-23` through DashScope OpenAI-compatible mode on small official GSM8K and MATH subsets.

## System

Core pipeline:

```text
official dataset -> JSONL split -> run_grid -> model adapter -> evaluator
                 -> SQLite/CSV traces -> regrade/failure analysis
                 -> fixed/oracle/aggregate/learned policies -> reports
```

Implemented components:

- OpenAI-compatible adapter plus deterministic mock adapter.
- Resumable grid runner for interrupted API experiments.
- GSM8K numeric evaluator.
- MATH boxed-answer/final-expression evaluator with simple LaTeX and numeric equivalence handling.
- SQLite trace persistence with model, budget, answer, correctness, token, cost, latency, and metadata fields.
- Offline policy evaluator for fixed budget, aggregate utility, and oracle upper bound.
- Learned policy router trained on train-split traces, with safe fallback and dev calibration.
- Consolidated GSM8K and multi-benchmark report generation.

## Data

Committed real-model subsets:

| benchmark | source | train | dev | test |
| --- | --- | ---: | ---: | ---: |
| GSM8K | `openai/gsm8k` | 60 | 20 | 20 |
| MATH | `Maxwell-Jia/MATH` | 60 | 20 | 20 |

`data/splits/` is intentionally ignored by git. The committed CSV artifacts contain the results needed for review and report reproduction.

## Policies

Evaluated policy types:

- `fixed_budget_*`: always use one budget.
- `aggregate_utility_budget_*`: choose one global budget by utility.
- `learned_policy`: classifier predicts a budget from query-side features.
- `safe_learned_policy_*`: learned policy with conservative fallback.
- `oracle_lowest_cost_correct`: offline upper bound that can inspect all candidate traces for each sample.

Utility uses:

```text
U = alpha * accuracy - beta * cost - gamma * latency
```

Default weights are `alpha=1.0`, `beta=5.0`, `gamma=0.02`.

## Held-Out Results

### GSM8K Test20

| policy | accuracy | avg cost | p95 latency | cost vs 1024 |
| --- | ---: | ---: | ---: | ---: |
| fixed_budget_0 | 0.900 | 0.000242 | 19.652s | 44.6% |
| fixed_budget_256 | 0.950 | 0.000344 | 16.721s | 63.5% |
| fixed_budget_1024 | 0.950 | 0.000542 | 23.390s | 100.0% |
| dev-calibrated safe policy | 0.950 | 0.000344 | 16.721s | 63.5% |
| oracle | 1.000 | 0.000288 | 10.325s | 53.2% |

GSM8K conclusion: the dev-calibrated safe policy matched the high-budget baseline accuracy while reducing average estimated cost by about 36.5%. The oracle shows additional headroom if per-sample routing improves.

### MATH Test20

| policy | accuracy | avg cost | p95 latency | cost vs 1024 |
| --- | ---: | ---: | ---: | ---: |
| fixed_budget_0 | 0.500 | 0.000604 | 32.348s | 59.1% |
| fixed_budget_256 | 0.550 | 0.000876 | 54.652s | 85.7% |
| fixed_budget_1024 | 0.250 | 0.001022 | 49.151s | 100.0% |
| aggregate_utility_budget_0 | 0.500 | 0.000604 | 32.348s | 59.1% |
| oracle | 0.600 | 0.000611 | 36.246s | 59.8% |

MATH conclusion: MATH is harder and more evaluator-sensitive. Budget `256` had the highest fixed-budget accuracy, while aggregate utility selected budget `0` because it was substantially cheaper and faster. Budget `1024` performed poorly, reinforcing that blindly increasing thinking budget is unreliable for this Qwen setup.

## Main Findings

1. Thinking budget is a meaningful routing variable. Accuracy, cost, and latency change substantially across budgets, and the best budget differs by benchmark.
2. Larger budget is not automatically better. On both GSM8K and MATH, budget `1024` was often slower, more expensive, and not more accurate.
3. Offline replay is valuable. Once candidate budget traces are recorded, new routing policies can be evaluated without additional API calls.
4. Raw learned routing is unstable at this scale. The raw classifier underperformed on held-out GSM8K test, so safe fallback and dev calibration are necessary.
5. MATH evaluation is harder than GSM8K evaluation. Many failures are answer-format or equivalence issues, so deterministic evaluators must be benchmark-specific.

## Limitations

- Small official subsets: 60/20/20 per benchmark, not full benchmark-scale evaluation.
- One real model: Qwen only. Joint model-budget routing is structurally implemented, but the real experiments mostly evaluate budget routing for one provider model.
- Prompt-level budget control: the external API may not expose true internal reasoning-token budgets; the project maps budgets to prompt instructions.
- Lightweight learned features: current learned router uses simple text features, not embeddings or calibrated uncertainty.
- MATH evaluator remains approximate: it handles common boxed/fraction/numeric cases but does not perform full symbolic equivalence.

## Next Work

- Add a second real model to evaluate true joint `(model, budget)` routing.
- Train a MATH-specific learned policy using MATH train/dev grids.
- Increase benchmark sample sizes once budget permits.
- Add uncertainty-aware routing: only escalate budget when confidence or predicted utility gap justifies the extra cost.
- Improve MATH equivalence with symbolic normalization for sets, intervals, tuples, and polynomials.

## Reproduction

Generate the consolidated reports:

```bash
python -m thinkrouter.experiments.make_gsm8k_report
python -m thinkrouter.experiments.make_benchmark_report
```

Run tests:

```bash
pytest
```

Important report artifacts:

- `results/reports/qwen_gsm8k_final_policy_report.md`
- `results/reports/qwen_multi_benchmark_policy_report.md`
- `results/tables/qwen_multi_benchmark_policy_report.csv`
