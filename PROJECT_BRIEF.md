# ThinkRouter Project Brief

## One-Liner

ThinkRouter is an adaptive routing system for reasoning LLMs that treats `(model, thinking_budget)` as the routing decision and optimizes the quality-cost-latency tradeoff on verifiable reasoning tasks.

## Resume Version

Built an adaptive thinking-budget router for reasoning LLMs with a full experiment pipeline: benchmark export, OpenAI-compatible model calls, SQLite tracing, deterministic answer grading, offline policy replay, learned budget selection, dev calibration, and held-out GSM8K evaluation.

## What It Demonstrates

- End-to-end LLM systems engineering: model adapter, routing policy, trace store, benchmark runner, evaluator, and reporting pipeline.
- Cost-aware evaluation: every trace records correctness, tokens, estimated cost, latency, model id, and budget.
- Scientific workflow: train/dev/test separation, regrading, failure analysis, offline oracle upper bound, and held-out test reporting.
- Practical routing discipline: raw learned routing is not trusted blindly; a safe calibrated fallback prevents cost/latency regressions when the learned selector is weak.

## Architecture

```text
benchmark JSONL
    -> run_grid.py
    -> OpenAI-compatible adapter or mock adapter
    -> SQLite trace store
    -> evaluator / regrader / failure analysis
    -> fixed, oracle, aggregate, learned, safe calibrated policies
    -> final report tables and figures
```

Core modules:

- `thinkrouter/app/models.py`: mock and OpenAI-compatible adapters.
- `thinkrouter/app/router.py`: feature extraction and online policy engine.
- `thinkrouter/app/store.py`: SQLite trace persistence.
- `thinkrouter/app/evaluators.py`: deterministic GSM8K numeric grading.
- `thinkrouter/experiments/run_grid.py`: benchmark grid runner with resume support.
- `thinkrouter/experiments/learned_policy_router.py`: learned budget selector and safe fallback replay.
- `thinkrouter/experiments/calibrate_learned_policy.py`: dev-set calibration for deployment policy selection.
- `thinkrouter/experiments/make_gsm8k_report.py`: consolidated final report generation.

## Key Result

On the committed Qwen GSM8K held-out `test20` split:

| policy | accuracy | avg cost | p95 latency | cost vs budget 1024 |
| --- | ---: | ---: | ---: | ---: |
| fixed budget 1024 | 0.950 | 0.000542 | 23.390s | 100.0% |
| dev-calibrated safe policy | 0.950 | 0.000344 | 16.721s | 63.5% |
| oracle upper bound | 1.000 | 0.000288 | 10.325s | 53.2% |

The calibrated safe policy matched the high-budget test accuracy while reducing average estimated cost by about 36.5%. The oracle gap shows that per-sample budget routing still has headroom beyond the current simple feature set.

## Important Caveat

This is not a claim that the current learned classifier already beats all fixed-budget baselines. In fact, the raw learned router underperformed on held-out test. The implemented safety layer and dev calibration are part of the project contribution: they prevent an unstable learned selector from becoming the deployed policy.

## Interview Talking Points

- Why budget is a first-class routing variable: many providers expose reasoning intensity through prompts, parameters, or model variants, so routing only by model leaves cost savings unused.
- Why deterministic grading matters: GSM8K numeric exact match avoids LLM-as-judge noise and makes cost-quality comparisons reproducible.
- Why offline replay is useful: once candidate budgets have been run, new routing policies can be evaluated without more API cost.
- Why safe fallback matters: small training sets can make learned routers over-allocate expensive budgets; calibration protects deployment quality.
- What to improve next: add more benchmarks, richer query features, uncertainty-aware routing, and a second model to make joint model-budget routing more meaningful.

## Resume Bullets

- Built ThinkRouter, an adaptive LLM routing system that jointly selects model and thinking budget, with OpenAI-compatible inference, SQLite tracing, deterministic grading, and reproducible experiment reports.
- Ran Qwen GSM8K train/dev/test budget-grid experiments and implemented fixed-budget, oracle, aggregate-utility, raw learned, safe fallback, and dev-calibrated policy evaluation.
- Achieved 0.95 held-out GSM8K test accuracy with a dev-calibrated safe policy at 63.5% of the average estimated cost of the 1024-budget baseline.
- Added resume-safe engineering features including resumable API experiments, evaluator regrading, failure analysis, API-key leak checks, and one-command final report generation.

## Reproduce The Main Report

```bash
python -m thinkrouter.experiments.make_gsm8k_report
```

Outputs:

- `results/tables/qwen_gsm8k_final_policy_report.csv`
- `results/reports/qwen_gsm8k_final_policy_report.md`
- `results/figures/qwen_gsm8k_test20_policy_comparison.png`
