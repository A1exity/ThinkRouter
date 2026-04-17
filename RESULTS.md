# RESULTS

No final benchmark results have been produced yet. The current results are Day-1 MVP smoke-test results using deterministic mock models.

## Current Status

Implemented Day-1 MVP components:

- built-in 20-sample GSM8K-style development set,
- fixed budgets `0`, `256`, `1024`, `4096`,
- mock and OpenAI-compatible model adapters,
- GSM8K numeric exact-match evaluator,
- SQLite trace store,
- Day-1 grid runner,
- FastAPI endpoints,
- Streamlit trace demo,
- baseline summary and Pareto plotting scripts.

## Day-1 MVP Results

Generated artifacts:

- trace database: `results/traces/day1.sqlite`
- grid table: `results/tables/day1_grid.csv`
- baseline summary: `results/tables/baseline_summary.csv`
- Pareto figure: `results/figures/pareto.png`

Run command:

```bash
python -m thinkrouter.experiments.run_day1_grid --limit 20 --db results/traces/day1.sqlite --out results/tables/day1_grid.csv
python -m thinkrouter.experiments.eval_baselines results/tables/day1_grid.csv
python -m thinkrouter.experiments.make_plots results/tables/day1_grid.csv
```

The Day-1 grid covers 20 built-in GSM8K-style samples, 2 mock model configurations, and 3 budget levels:

| model | budget | accuracy | avg cost | p95 latency | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| mock-cheap | 0 | 1.000 | 0.000000 | 0.000021 | 20 |
| mock-cheap | 256 | 1.000 | 0.000000 | 0.000015 | 20 |
| mock-cheap | 1024 | 1.000 | 0.000000 | 0.000021 | 20 |
| mock-strong | 0 | 1.000 | 0.000014 | 0.000014 | 20 |
| mock-strong | 256 | 1.000 | 0.000014 | 0.000021 | 20 |
| mock-strong | 1024 | 1.000 | 0.000014 | 0.000015 | 20 |

Because these results use deterministic mock adapters, they validate the end-to-end pipeline rather than model quality. Real benchmark results should be generated with fixed train/dev/test splits and real model adapters.

## Final Reporting Targets

The final report should include:

- accuracy,
- average cost per query,
- p95 latency,
- cost reduction vs Always Strong + Max Budget,
- accuracy drop vs Always Strong + Max Budget,
- cost-accuracy Pareto plot,
- per-benchmark comparison for GSM8K, MATH-500, and HumanEval.
