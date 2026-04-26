# ThinkRouter

ThinkRouter is a routing system for reasoning workloads where the decision is `(model, budget)`, not just `model`. The repository now includes the completed official protocol, real semantic features, Phase 2 online routing defaults, full official reruns for `GSM8K`, `MATH-500`, and `HumanEval`, and a single final official report.

## Official Protocol

- Model pool: `qwen-flash`, `qwen-plus`, `qwen-max`
- Budget set: `0`, `256`, `1024`
- Official benchmarks:
  - `GSM8K`
  - `MATH-500`
  - `HumanEval`
- Split size per benchmark: `60 train / 20 dev / 20 test`
- Learned router stack:
  - `threshold`
  - `logreg_joint`
  - `mlp_factorized`
  - `uncertainty_aware`
- Online default router:
  - `uncertainty_aware`
- Semantic feature backend:
  - `sentence-transformers/all-MiniLM-L6-v2`

## Final Official Results

The only official top-level results are:

- [final_official_results.csv](C:/Users/23965/Desktop/ThinkRouter/results/tables/final_official_results.csv)
- [final_official_pareto.png](C:/Users/23965/Desktop/ThinkRouter/results/figures/final_official_pareto.png)
- [final_official_failures.csv](C:/Users/23965/Desktop/ThinkRouter/results/tables/final_official_failures.csv)
- [final_official_report.md](C:/Users/23965/Desktop/ThinkRouter/results/reports/final_official_report.md)

Official summary:

| benchmark | official learned policy | router | accuracy | avg cost | p95 latency | utility | beats strongest fixed | beats aggregate |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| GSM8K | `phase2_logreg_joint` | `logreg_joint` | 0.95 | 0.000186 | 7.401475 | 0.801039 | yes | yes |
| MATH-500 | `phase2_logreg_joint` | `logreg_joint` | 0.25 | 0.000155 | 18.251530 | -0.115803 | no | no |
| HumanEval | `phase2_logreg_joint` | `logreg_joint` | 0.60 | 0.000377 | 19.026367 | 0.217585 | no | no |

Main conclusion:

- The final acceptance condition is satisfied because the learned router wins on at least one official benchmark.
- That benchmark is `GSM8K`.
- `MATH-500` and `HumanEval` remain valuable negative controls: the learned router is cheaper there, but not utility-optimal.

## Frozen Official Protocol

The only formal protocol is defined in:

- [`configs/official_protocol.json`](C:/Users/23965/Desktop/ThinkRouter/configs/official_protocol.json)
- [`official_protocol.py`](C:/Users/23965/Desktop/ThinkRouter/thinkrouter/official_protocol.py)

Formal settings:

| field | value |
| --- | --- |
| model pool | `qwen-flash,qwen-plus,qwen-max` |
| budgets | `0,256,1024` |
| benchmarks | `GSM8K`, `MATH-500`, `HumanEval` |
| split sizes | `60 train / 20 dev / 20 test` |
| baselines | `fixed_model_budget`, `model_only`, `budget_only`, `joint_aggregate_utility`, `joint_safe_fallback` |
| routers | `threshold`, `logreg_joint`, `mlp_factorized`, `uncertainty_aware` |
| utility | `accuracy - 5 * cost - 0.02 * latency` |
| default online router | `uncertainty_aware` |

Historical smoke/dev slices remain in the repo, but they are appendix artifacts only.

## Official Reproducible Pipeline

The only official command chain is fixed:

```powershell
.\scripts\run_official_pipeline.ps1
```

Expanded stages:

```bash
python -m thinkrouter.experiments.run_official_pipeline --stage prepare-data
python -m thinkrouter.experiments.run_official_pipeline --stage grids
python -m thinkrouter.experiments.run_official_pipeline --stage routers
python -m thinkrouter.experiments.run_official_pipeline --stage report
```

Stage meanings:

- `prepare-data`: exports the official `60/20/20` splits for `GSM8K`, `MATH-500`, and `HumanEval`
- `grids`: collects the official model-budget traces
- `routers`: trains, calibrates, selects, and replays the learned routers
- `report`: writes `final_official_results.csv`, `final_official_pareto.png`, `final_official_failures.csv`, and `final_official_report.md`

Historical smoke scripts, debug scripts, and slice-specific reruns remain in the repository for debugging only. They are not part of the official reproduction path.

## Setup

```bash
conda env create -f environment.yml
conda activate thinkrouter
pytest
```

Environment variables for the official Qwen pool:

```text
THINKROUTER_MODEL_POOL=qwen-flash,qwen-plus,qwen-max
THINKROUTER_QWEN_FLASH_MODEL=qwen3.5-flash-2026-02-23
THINKROUTER_QWEN_PLUS_MODEL=qwen3.5-plus-2026-02-15
THINKROUTER_QWEN_MAX_MODEL=qwen3-max-2026-01-23
```

## Key Paths

- [`METHOD.md`](C:/Users/23965/Desktop/ThinkRouter/METHOD.md): current method and protocol definition
- [`RESULTS.md`](C:/Users/23965/Desktop/ThinkRouter/RESULTS.md): final result inventory and appendix/historical index
- [`FINAL_REPORT.md`](C:/Users/23965/Desktop/ThinkRouter/FINAL_REPORT.md): final repository-wide closeout summary
- [`results/reports/final_official_report.md`](C:/Users/23965/Desktop/ThinkRouter/results/reports/final_official_report.md): final official benchmark report
- [`scripts/run_official_pipeline.ps1`](C:/Users/23965/Desktop/ThinkRouter/scripts/run_official_pipeline.ps1): one-command official rerun entrypoint
- [`thinkrouter/experiments/run_official_pipeline.py`](C:/Users/23965/Desktop/ThinkRouter/thinkrouter/experiments/run_official_pipeline.py): staged official pipeline implementation
