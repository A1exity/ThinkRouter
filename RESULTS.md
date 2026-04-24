# RESULTS

No final multi-benchmark official report has been produced yet. The repository now includes deterministic mock-model pipeline checks, provider-backed Qwen GSM8K train/dev/test experiments, and an initial provider-backed Qwen MATH smoke test.

## Current Status

Implemented Day-1 and Week-2 pipeline components:

- built-in GSM8K-style, MATH-style, and HumanEval-style seed samples,
- fixed budgets `0`, `256`, `1024`, `4096`,
- mock and OpenAI-compatible model adapters,
- GSM8K numeric exact-match evaluator,
- MATH boxed-answer and final-expression evaluator,
- exact-match evaluator for non-GSM8K seed tasks,
- SQLite trace store,
- Day-1 grid runner,
- frozen train/dev/test grid runner,
- benchmark JSONL export and input loader,
- official GSM8K subset export to JSONL,
- official MATH subset export to JSONL,
- real OpenAI-compatible endpoint configuration preflight,
- FastAPI endpoints,
- Streamlit trace demo,
- baseline summary and Pareto plotting scripts,
- failure analysis and trace regrading scripts,
- offline fixed-budget, oracle, and aggregate-utility policy evaluation,
- learned policy router training and cross-split replay,
- sklearn difficulty estimator training,
- sklearn budget predictor training,
- optional router loading from joblib model paths.

## Frozen Seed Splits

The local seed suite is for deterministic pipeline validation only. It is not a replacement for official GSM8K, MATH-500, or HumanEval benchmark reporting.

| task | train | dev | test |
| --- | ---: | ---: | ---: |
| gsm8k | 12 | 4 | 4 |
| math | 6 | 2 | 2 |
| humaneval | 4 | 2 | 2 |

Generated trace tables:

| table | traces | task coverage |
| --- | ---: | --- |
| `results/tables/train_grid.csv` | 132 | gsm8k, math, humaneval |
| `results/tables/dev_grid.csv` | 48 | gsm8k, math, humaneval |
| `results/tables/test_grid.csv` | 48 | gsm8k, math, humaneval |

The trace counts reflect 2 mock models and 3 budget levels per sample.

## Generated Artifacts

- Day-1 trace database: `results/traces/day1.sqlite`
- frozen train/dev/test trace databases: `results/traces/train_grid.sqlite`, `results/traces/dev_grid.sqlite`, `results/traces/test_grid.sqlite`
- Day-1 grid table: `results/tables/day1_grid.csv`
- frozen split tables: `results/tables/train_grid.csv`, `results/tables/dev_grid.csv`, `results/tables/test_grid.csv`
- Day-1 baseline summary: `results/tables/baseline_summary.csv`
- dev baseline summary: `results/tables/dev_baseline_summary.csv`
- Day-1 Pareto figure: `results/figures/pareto.png`
- dev Pareto figure: `results/figures/dev_pareto.png`
- train-split difficulty model: `results/models/difficulty.joblib`
- train-split budget model: `results/models/budget.joblib`

## Reproduction Commands

```bash
python -m thinkrouter.experiments.run_day1_grid --limit 20 --db results/traces/day1.sqlite --out results/tables/day1_grid.csv
python -m thinkrouter.experiments.eval_baselines results/tables/day1_grid.csv --out results/tables/baseline_summary.csv
python -m thinkrouter.experiments.make_plots results/tables/day1_grid.csv --out results/figures/pareto.png

python -m thinkrouter.experiments.prepare_data --source seed --task all --split all --out data/splits/seed.jsonl --summary
python -m thinkrouter.experiments.run_grid --input data/splits/seed.jsonl --task all --split train --budgets 0,256,1024 --db results/traces/train_grid.sqlite --out results/tables/train_grid.csv
python -m thinkrouter.experiments.run_grid --input data/splits/seed.jsonl --task all --split dev --budgets 0,256,1024 --db results/traces/dev_grid.sqlite --out results/tables/dev_grid.csv
python -m thinkrouter.experiments.run_grid --input data/splits/seed.jsonl --task all --split test --budgets 0,256,1024 --db results/traces/test_grid.sqlite --out results/tables/test_grid.csv

python -m thinkrouter.experiments.train_difficulty results/tables/train_grid.csv --out results/models/difficulty.joblib
python -m thinkrouter.experiments.train_budget results/tables/train_grid.csv --out results/models/budget.joblib
python -m thinkrouter.experiments.eval_baselines results/tables/dev_grid.csv --out results/tables/dev_baseline_summary.csv
python -m thinkrouter.experiments.make_plots results/tables/dev_grid.csv --out results/figures/dev_pareto.png
```

The JSONL interface was smoke-tested by exporting 38 seed samples to `data/splits/seed.jsonl` and running a small `run_grid --input` job. `data/splits/` and SQLite traces are intentionally ignored by git.

A real endpoint can be checked with `python -m thinkrouter.experiments.smoke_real_model --model <model-id>`. Passing `--run` performs an actual provider call and can incur cost.

Because the early committed results use deterministic mock adapters, all mock accuracies are expected to be perfect. The value of those stages is validating the train/dev/test and JSONL experiment plumbing, not measuring model quality.

## Official GSM8K Data Export

`prepare_data --source gsm8k` was used to download `openai/gsm8k` through the Hugging Face mirror and export a local JSONL subset:

| file | total | train | dev | test |
| --- | ---: | ---: | ---: | ---: |
| `data/splits/gsm8k.jsonl` | 100 | 60 | 20 | 20 |

The exported JSONL file and Hugging Face cache are local artifacts and are not committed. A 2-sample mock `run_grid --input data/splits/gsm8k.jsonl` smoke test passed, confirming the official GSM8K JSONL is compatible with the grid runner.

## Official MATH Data Export

`prepare_data --source math` was used to download `Maxwell-Jia/MATH` through the Hugging Face mirror and export a local JSONL subset:

| file | total | train | dev | test |
| --- | ---: | ---: | ---: | ---: |
| `data/splits/math.jsonl` | 100 | 60 | 20 | 20 |

The exporter extracts the final `\boxed{...}` answer from each official solution. The exported JSONL file and Hugging Face cache are local artifacts and are not committed. A 3-sample mock grid passed, confirming the official MATH JSONL is compatible with the grid runner and MATH evaluator.

Generated mock artifacts:

- `results/tables/math_mock_dev3_grid.csv`
- `results/tables/math_mock_dev3_grid_regraded.csv`
- `results/tables/math_mock_dev3_summary.csv`
- `results/figures/math_mock_dev3_pareto.png`

## Qwen Real-Model Smoke Tests

Small provider-backed smoke tests were run with `qwen3.5-flash-2026-02-23` through DashScope OpenAI-compatible mode. These are not benchmark results; they validate that the real model path works and that budget-level traces can be recorded.

| file | rows | task | split | budgets | accuracy | total estimated cost | avg latency |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `results/tables/qwen_gsm8k_dev_smoke.csv` | 4 | gsm8k | dev | 0 | 1.000 | 0.000500 | 3.022s |
| `results/tables/qwen_gsm8k_dev_budget_grid.csv` | 12 | gsm8k | dev | 0,256,1024 | 1.000 | 0.001778 | 3.461s |

Budget-grid summary:

| budget | accuracy | avg cost | p95 latency | n |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.000115 | 4.814s | 4 |
| 256 | 1.000 | 0.000131 | 3.147s | 4 |
| 1024 | 1.000 | 0.000199 | 6.496s | 4 |

Generated artifacts:

- `results/tables/qwen_gsm8k_dev_budget_grid.csv`
- `results/tables/qwen_gsm8k_dev_budget_summary.csv`
- `results/figures/qwen_gsm8k_dev_budget_pareto.png`

The local `.env` file contains the API key and is intentionally ignored by git. SQLite traces under `results/traces/` are also ignored.

## Official GSM8K Qwen Dev Smoke Test

A small official GSM8K provider-backed run was executed on the exported `data/splits/gsm8k.jsonl` dev split using `qwen3.5-flash-2026-02-23`. This is still a smoke test, not a final benchmark, because it covers only 5 examples and one real model.

| file | rows | task | split | limit | budgets | accuracy | total estimated cost |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: |
| `results/tables/qwen_gsm8k_official_dev5.csv` | 10 | gsm8k | dev | 5 | 0,256 | 1.000 | 0.002365 |

Budget summary:

| budget | accuracy | avg cost | p95 latency | n |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.000203 | 9.785s | 5 |
| 256 | 1.000 | 0.000270 | 10.777s | 5 |

Generated artifacts:

- `results/tables/qwen_gsm8k_official_dev5.csv`
- `results/tables/qwen_gsm8k_official_dev5_summary.csv`
- `results/figures/qwen_gsm8k_official_dev5_pareto.png`

## Official MATH Qwen Dev Smoke Test

A small official MATH provider-backed run was executed on the exported `data/splits/math.jsonl` dev split using `qwen3.5-flash-2026-02-23`. This is a smoke test, not a benchmark result, because it covers only 5 examples and one real model.

Original evaluator accuracy was 0.500 because several outputs contained the correct answer without a boxed or marked final answer. After improving the MATH final-expression extractor and regrading, all 10 rows were correct:

| budget | regraded accuracy | avg cost | p95 latency | n |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.000428 | 18.620s | 5 |
| 256 | 1.000 | 0.000428 | 17.919s | 5 |

Generated artifacts:

- `results/tables/qwen_math_official_dev5.csv`
- `results/tables/qwen_math_official_dev5_regraded.csv`
- `results/tables/qwen_math_official_dev5_summary_regraded.csv`
- `results/tables/qwen_math_official_dev5_failures_regraded.csv`
- `results/figures/qwen_math_official_dev5_pareto_regraded.png`

## Official MATH Qwen Dev20 Budget Grid

A larger official MATH dev run was executed with `qwen3.5-flash-2026-02-23` over all 20 exported dev examples and three budget levels. This is the second real benchmark path in the project, but it is still a dev-set result rather than a held-out MATH test result.

After improving simple MATH equivalence checks and regrading, the budget summary is:

| budget | regraded accuracy | avg cost | p95 latency | n |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.800 | 0.001702 | 58.935s | 20 |
| 256 | 0.700 | 0.000649 | 42.538s | 20 |
| 1024 | 0.350 | 0.000779 | 32.662s | 20 |

Offline policy summary:

| policy | accuracy | avg cost | total cost | avg latency | p95 latency | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_budget_0 | 0.800 | 0.001702 | 0.034038 | 39.063s | 58.935s | 20 |
| fixed_budget_256 | 0.700 | 0.000649 | 0.012984 | 15.822s | 42.538s | 20 |
| fixed_budget_1024 | 0.350 | 0.000779 | 0.015573 | 17.870s | 32.662s | 20 |
| oracle_lowest_cost_correct | 0.800 | 0.000486 | 0.009720 | 11.274s | 29.652s | 20 |
| aggregate_utility_budget_256 | 0.700 | 0.000649 | 0.012984 | 15.822s | 42.538s | 20 |

Interpretation: MATH is harder and more evaluator-sensitive than GSM8K. On this dev subset, budget `0` had the best accuracy but much higher cost and latency, while aggregate utility selected budget `256`. The oracle result shows cost/latency headroom from per-sample routing, but the current learned router has not yet been trained or calibrated on MATH.

Generated artifacts:

- `results/tables/qwen_math_official_dev20_budget_grid.csv`
- `results/tables/qwen_math_official_dev20_budget_grid_regraded.csv`
- `results/tables/qwen_math_official_dev20_budget_summary_regraded.csv`
- `results/tables/qwen_math_official_dev20_policy_summary.csv`
- `results/tables/qwen_math_official_dev20_policy_stats.csv`
- `results/tables/qwen_math_official_dev20_failures_regraded.csv`
- `results/figures/qwen_math_official_dev20_budget_pareto_regraded.png`

## Official MATH Qwen Test20 Budget Grid

A held-out official MATH test run was executed with `qwen3.5-flash-2026-02-23` over all 20 exported test examples and three budget levels.

Regraded budget summary:

| budget | regraded accuracy | avg cost | p95 latency | n |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.500 | 0.000604 | 32.348s | 20 |
| 256 | 0.550 | 0.000876 | 54.652s | 20 |
| 1024 | 0.250 | 0.001022 | 49.151s | 20 |

Offline policy summary:

| policy | accuracy | avg cost | total cost | avg latency | p95 latency | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_budget_0 | 0.500 | 0.000604 | 0.012073 | 13.984s | 32.348s | 20 |
| fixed_budget_256 | 0.550 | 0.000876 | 0.017518 | 28.854s | 54.652s | 20 |
| fixed_budget_1024 | 0.250 | 0.001022 | 0.020439 | 23.268s | 49.151s | 20 |
| oracle_lowest_cost_correct | 0.600 | 0.000611 | 0.012228 | 14.560s | 36.246s | 20 |
| aggregate_utility_budget_0 | 0.500 | 0.000604 | 0.012073 | 13.984s | 32.348s | 20 |

Interpretation: on held-out MATH, budget `256` has the best fixed-budget accuracy, while aggregate utility chooses budget `0` because it is much cheaper and faster. The `1024` setting again performs poorly, reinforcing that blindly increasing the thinking budget is not reliable for this Qwen setup.

Generated artifacts:

- `results/tables/qwen_math_official_test20_budget_grid.csv`
- `results/tables/qwen_math_official_test20_budget_grid_regraded.csv`
- `results/tables/qwen_math_official_test20_budget_summary_regraded.csv`
- `results/tables/qwen_math_official_test20_policy_summary.csv`
- `results/tables/qwen_math_official_test20_policy_stats.csv`
- `results/tables/qwen_math_official_test20_failures_regraded.csv`
- `results/figures/qwen_math_official_test20_budget_pareto_regraded.png`

## Official GSM8K Qwen Dev20 Budget Grid

A larger official GSM8K dev run was executed with `qwen3.5-flash-2026-02-23` over all 20 exported dev examples and three budget levels. This is the first run where budget differences become visible.

Original extractor result:

| file | rows | task | split | budgets | accuracy | total estimated cost |
| --- | ---: | --- | --- | --- | ---: | ---: |
| `results/tables/qwen_gsm8k_official_dev20_budget_grid.csv` | 60 | gsm8k | dev | 0,256,1024 | 0.883 | 0.023382 |

Original budget summary:

| budget | accuracy | correct | avg cost | p95 latency | n |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.950 | 19 | 0.000297 | 14.334s | 20 |
| 256 | 0.950 | 19 | 0.000319 | 11.370s | 20 |
| 1024 | 0.750 | 15 | 0.000553 | 28.879s | 20 |

Failure analysis found 7 incorrect traces across 5 unique samples. Four of the five `1024` failures contained the correct numeric answer in the output but ended with another number, causing the earlier extractor to select the wrong value. Those same samples were correct at budgets `0` and `256`.

After improving the answer-heading extractor and regrading, 3 rows changed from incorrect to correct. The regraded summary is:

| budget | regraded accuracy | correct | avg cost | p95 latency | n |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.950 | 19 | 0.000297 | 14.334s | 20 |
| 256 | 0.950 | 19 | 0.000319 | 11.370s | 20 |
| 1024 | 0.900 | 18 | 0.000553 | 28.879s | 20 |

Regraded interpretation: higher budget still did not improve accuracy on this subset, and `1024` remained substantially more expensive and slower. The strongest conclusion is therefore not "1024 is always worse," but "blindly increasing budget is not cost-effective and can introduce answer-format instability."

Offline policy evaluation on the regraded grid:

| policy | accuracy | avg cost | total cost | avg latency | p95 latency | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_budget_0 | 0.950 | 0.000297 | 0.005939 | 6.996s | 14.334s | 20 |
| fixed_budget_256 | 0.950 | 0.000319 | 0.006385 | 6.592s | 11.370s | 20 |
| fixed_budget_1024 | 0.900 | 0.000553 | 0.011059 | 13.159s | 28.879s | 20 |
| oracle_lowest_cost_correct | 0.950 | 0.000233 | 0.004670 | 4.895s | 8.396s | 20 |
| aggregate_utility_budget_256 | 0.950 | 0.000319 | 0.006385 | 6.592s | 11.370s | 20 |

The oracle policy is an offline upper bound that can inspect all completed candidate traces for each sample. It is not deployable as-is, but it shows the routing headroom: per-sample selection can match the best observed accuracy while reducing cost and latency. The aggregate utility baseline selected budget `256`, which matched budget `0` accuracy with lower p95 latency but slightly higher cost.

Generated artifacts:

- `results/tables/qwen_gsm8k_official_dev20_budget_grid.csv`
- `results/tables/qwen_gsm8k_official_dev20_budget_summary.csv`
- `results/figures/qwen_gsm8k_official_dev20_budget_pareto.png`
- `results/tables/qwen_gsm8k_official_dev20_failures.csv`
- `results/tables/qwen_gsm8k_official_dev20_budget_grid_regraded.csv`
- `results/tables/qwen_gsm8k_official_dev20_budget_summary_regraded.csv`
- `results/tables/qwen_gsm8k_official_dev20_failures_regraded.csv`
- `results/figures/qwen_gsm8k_official_dev20_budget_pareto_regraded.png`
- `results/tables/qwen_gsm8k_official_dev20_policy_summary.csv`
- `results/tables/qwen_gsm8k_official_dev20_policy_stats.csv`

## Official GSM8K Qwen Train60 Learned Policy

The 60-example train split was run with the same Qwen model and budgets. The first network attempt was interrupted after 26 traces; `run_grid --resume` then skipped existing `(sample_id, model, budget)` combinations and completed the 180-row grid.

Regraded train60 budget summary:

| budget | regraded accuracy | avg cost | p95 latency | n |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.983 | 0.000266 | 10.907s | 60 |
| 256 | 0.983 | 0.000352 | 20.436s | 60 |
| 1024 | 0.917 | 0.000495 | 36.716s | 60 |

Train60 offline policy summary:

| policy | accuracy | avg cost | total cost | avg latency | p95 latency | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_budget_0 | 0.983 | 0.000266 | 0.015967 | 6.703s | 10.907s | 60 |
| fixed_budget_256 | 0.983 | 0.000352 | 0.021121 | 8.493s | 20.436s | 60 |
| fixed_budget_1024 | 0.917 | 0.000495 | 0.029706 | 12.852s | 36.716s | 60 |
| oracle_lowest_cost_correct | 0.983 | 0.000239 | 0.014354 | 5.785s | 10.788s | 60 |
| aggregate_utility_budget_0 | 0.983 | 0.000266 | 0.015967 | 6.703s | 10.907s | 60 |

The learned policy router was trained on train60 labels derived from per-sample utility. Label counts were `0: 41`, `256: 17`, `1024: 2`. Training selected a safe fallback budget of `0`, matching the train-split aggregate-utility policy.

Replaying the raw learned classifier and the safe learned policy on the separate dev20 grid produced:

| policy | train split | eval split | accuracy | avg cost | total cost | avg latency | p95 latency | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_policy_raw | train60 | dev20 | 0.950 | 0.000373 | 0.007465 | 8.377s | 14.455s | 20 |
| safe_learned_policy_fallback_budget_0 | train60 | dev20 | 0.950 | 0.000297 | 0.005939 | 6.996s | 14.334s | 20 |

The raw learned router predicted budget `0` for 10 dev samples, `256` for 7, and `1024` for 3. It matched the best fixed-budget accuracy on dev20, but did not beat fixed budget `0` or `256` on cost/latency. Dev calibration selected fallback budget `256`, which had the highest dev utility among raw learned and fixed-budget candidates.

Held-out test20 results:

| policy | train split | calibration split | eval split | accuracy | avg cost | total cost | avg latency | p95 latency | n |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_budget_0 | - | - | test20 | 0.900 | 0.000242 | 0.004836 | 6.424s | 19.652s | 20 |
| fixed_budget_256 | - | - | test20 | 0.950 | 0.000344 | 0.006883 | 8.725s | 16.721s | 20 |
| fixed_budget_1024 | - | - | test20 | 0.950 | 0.000542 | 0.010840 | 13.555s | 23.390s | 20 |
| learned_policy_raw | train60 | - | test20 | 0.900 | 0.000284 | 0.005677 | 7.243s | 19.652s | 20 |
| safe_learned_policy_fallback_budget_0 | train60 | - | test20 | 0.900 | 0.000242 | 0.004836 | 6.424s | 19.652s | 20 |
| safe_learned_policy_fallback_budget_256 | train60 | dev20 | test20 | 0.950 | 0.000344 | 0.006883 | 8.725s | 16.721s | 20 |
| oracle_lowest_cost_correct | - | - | test20 | 1.000 | 0.000288 | 0.005765 | 7.674s | 10.325s | 20 |

The calibrated policy matches the best fixed-budget test accuracy and avoids the train-only fallback's accuracy drop. It still does not close the gap to the oracle, so the remaining router opportunity is per-sample budget selection rather than global fallback selection.

A consolidated report was generated from the committed dev/test policy CSVs. The multi-benchmark report uses held-out `test20` rows for both GSM8K and MATH.

- `results/tables/qwen_gsm8k_final_policy_report.csv`
- `results/reports/qwen_gsm8k_final_policy_report.md`
- `results/figures/qwen_gsm8k_test20_policy_comparison.png`
- `results/tables/qwen_multi_benchmark_policy_report.csv`
- `results/reports/qwen_multi_benchmark_policy_report.md`

Generated artifacts:

- `results/tables/qwen_gsm8k_official_train60_budget_grid.csv`
- `results/tables/qwen_gsm8k_official_train60_budget_summary.csv`
- `results/tables/qwen_gsm8k_official_train60_budget_grid_regraded.csv`
- `results/tables/qwen_gsm8k_official_train60_budget_summary_regraded.csv`
- `results/tables/qwen_gsm8k_official_train60_policy_summary.csv`
- `results/tables/qwen_gsm8k_official_train60_policy_stats.csv`
- `results/tables/qwen_gsm8k_official_train60_failures_regraded.csv`
- `results/figures/qwen_gsm8k_official_train60_budget_pareto_regraded.png`
- `results/models/qwen_gsm8k_official_train60_learned_policy.joblib`
- `results/tables/qwen_gsm8k_official_train60_to_dev20_learned_policy_summary.csv`
- `results/tables/qwen_gsm8k_official_train60_to_dev20_learned_policy_selected.csv`
- `results/tables/qwen_gsm8k_official_train60_to_dev20_raw_learned_policy_summary.csv`
- `results/tables/qwen_gsm8k_official_train60_to_dev20_raw_learned_policy_selected.csv`
- `results/tables/qwen_gsm8k_official_train60_to_dev20_safe_policy_summary.csv`
- `results/tables/qwen_gsm8k_official_train60_to_dev20_safe_policy_selected.csv`
- `results/tables/qwen_gsm8k_official_test20_budget_grid.csv`
- `results/tables/qwen_gsm8k_official_test20_budget_grid_regraded.csv`
- `results/tables/qwen_gsm8k_official_test20_budget_summary_regraded.csv`
- `results/tables/qwen_gsm8k_official_test20_policy_summary.csv`
- `results/tables/qwen_gsm8k_official_test20_policy_stats.csv`
- `results/tables/qwen_gsm8k_official_test20_failures_regraded.csv`
- `results/figures/qwen_gsm8k_official_test20_budget_pareto_regraded.png`
- `results/models/qwen_gsm8k_official_train60_dev20_calibrated_policy.joblib`
- `results/tables/qwen_gsm8k_official_dev20_calibration_summary.csv`
- `results/tables/qwen_gsm8k_official_train60_to_test20_raw_learned_policy_summary.csv`
- `results/tables/qwen_gsm8k_official_train60_to_test20_raw_learned_policy_selected.csv`
- `results/tables/qwen_gsm8k_official_train60_to_test20_safe_policy_summary.csv`
- `results/tables/qwen_gsm8k_official_train60_to_test20_safe_policy_selected.csv`
- `results/tables/qwen_gsm8k_official_train60_dev20_to_test20_calibrated_policy_summary.csv`
- `results/tables/qwen_gsm8k_official_train60_dev20_to_test20_calibrated_policy_selected.csv`
- `results/tables/qwen_gsm8k_final_policy_report.csv`
- `results/reports/qwen_gsm8k_final_policy_report.md`
- `results/figures/qwen_gsm8k_test20_policy_comparison.png`

## Qwen 3.5 Pool Phase 1 Completion Slice

Phase 1 was closed out with a real three-tier Qwen pool on DashScope:

- `qwen3.5-flash-2026-02-23`
- `qwen3.5-plus-2026-02-15`
- `qwen3-max-2026-01-23`

Two final small real-benchmark slices were added to verify the multi-model system path end to end:

1. GSM8K `dev5` with `flash / plus / max` and budgets `0 / 256 / 1024`
2. HumanEval seed `dev2` with `flash / plus / max` at budget `256`

The GSM8K slice completed all `45` traces and all runs were correct. The strongest utility winner on this small slice was `qwen3-max-2026-01-23` at budget `0`, while the cheapest always-correct point was still the flash tier at budget `0`.

The HumanEval slice completed `6` traces and all were incorrect, which is still useful for Phase 1 because it validates the real code-generation evaluation path, failure taxonomy, and multi-model trace/report flow. On this small code slice, `qwen3-max-2026-01-23` was the least costly failed option and the flash `budget 0` configuration was intentionally not used because it produced pathological runtimes on the seed tasks.

Added artifacts:

- `results/tables/qwen35_pool_gsm8k_dev5_grid.csv`
- `results/tables/qwen35_pool_gsm8k_dev5_baseline_summary.csv`
- `results/tables/qwen35_pool_gsm8k_dev5_policy_summary.csv`
- `results/tables/qwen35_pool_gsm8k_dev5_policy_stats.csv`
- `results/tables/qwen35_pool_gsm8k_dev5_failures.csv`
- `results/figures/qwen35_pool_gsm8k_dev5_pareto.png`
- `results/tables/qwen35_pool_humaneval_dev2_budget256_grid.csv`
- `results/tables/qwen35_pool_humaneval_dev2_budget256_baseline_summary.csv`
- `results/tables/qwen35_pool_humaneval_dev2_budget256_policy_summary.csv`
- `results/tables/qwen35_pool_humaneval_dev2_budget256_policy_stats.csv`
- `results/tables/qwen35_pool_humaneval_dev2_budget256_failures.csv`
- `results/figures/qwen35_pool_humaneval_dev2_budget256_pareto.png`

## Final Reporting Targets

The final report should include:

- accuracy,
- average cost per query,
- p95 latency,
- cost reduction vs Always Strong + Max Budget,
- accuracy drop vs Always Strong + Max Budget,
- cost-accuracy Pareto plot,
- per-benchmark comparison for GSM8K, MATH-500, and HumanEval.
