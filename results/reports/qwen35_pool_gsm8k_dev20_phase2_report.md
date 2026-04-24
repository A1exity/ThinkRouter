# ThinkRouter Phase 2 Report

This report ranks the integrated baseline and Phase 2 router summaries by utility.

- Best overall policy: `fixed_model_budget_qwen3-max-2026-01-23_0`
- Best overall utility: `0.904947`
- Best Phase 2 policy: `phase2_logreg_joint`
- Best Phase 2 utility: `0.813895`

| policy                                                      | policy_family            | router_name       | selected_model           | selected_model_tier   | selected_budget   |   accuracy |   avg_cost |   avg_latency |   avg_route_confidence |   fallback_rate |   utility |
|:------------------------------------------------------------|:-------------------------|:------------------|:-------------------------|:----------------------|:------------------|-----------:|-----------:|--------------:|-----------------------:|----------------:|----------:|
| fixed_model_budget_qwen3-max-2026-01-23_0                   | fixed_model_budget       | nan               | qwen3-max-2026-01-23     | strong                | 0                 |   1        |   0.000849 |       4.54046 |             nan        |             nan |  0.904947 |
| aggregate_utility_qwen3-max-2026-01-23_0                    | joint_aggregate_utility  | nan               | qwen3-max-2026-01-23     | strong                | 0                 |   1        |   0.000849 |       4.54046 |             nan        |             nan |  0.904947 |
| aggregate_utility_safe_fallback_qwen3-max-2026-01-23_0_to_0 | joint_safe_fallback      | nan               | qwen3-max-2026-01-23     | strong                | 0                 |   1        |   0.000849 |       4.54046 |             nan        |             nan |  0.904947 |
| model_only_best_qwen3-max-2026-01-23                        | model_only               | nan               | qwen3-max-2026-01-23     | strong                | nan               |   0.983333 |   0.001243 |       7.38904 |             nan        |             nan |  0.829338 |
| phase2_logreg_joint                                         | phase2_logreg_joint      | logreg_joint      | qwen3.5-flash-2026-02-23 | cheap                 | mixed             |   0.95     |   0.000246 |       6.74378 |               0.77851  |               0 |  0.813895 |
| phase2_mlp_factorized                                       | phase2_mlp_factorized    | mlp_factorized    | qwen3.5-flash-2026-02-23 | cheap                 | mixed             |   0.95     |   0.000246 |       6.74378 |               0.987323 |               0 |  0.813895 |
| phase2_threshold                                            | phase2_threshold         | threshold         | qwen3.5-flash-2026-02-23 | cheap                 | mixed             |   0.95     |   0.000246 |       6.74378 |               0.638125 |               0 |  0.813895 |
| phase2_uncertainty_aware                                    | phase2_uncertainty_aware | uncertainty_aware | qwen3.5-flash-2026-02-23 | cheap                 | mixed             |   0.95     |   0.000246 |       6.74378 |               0.987323 |               0 |  0.813895 |
| fixed_model_budget_qwen3-max-2026-01-23_1024                | fixed_model_budget       | nan               | qwen3-max-2026-01-23     | strong                | 1024              |   1        |   0.001471 |       9.00719 |             nan        |             nan |  0.812501 |
| fixed_model_budget_qwen3-max-2026-01-23_256                 | fixed_model_budget       | nan               | qwen3-max-2026-01-23     | strong                | 256               |   0.95     |   0.001409 |       8.61948 |             nan        |             nan |  0.770566 |
| fixed_model_budget_qwen3.5-flash-2026-02-23_256             | fixed_model_budget       | nan               | qwen3.5-flash-2026-02-23 | cheap                 | 256               |   0.95     |   0.000328 |       9.32627 |             nan        |             nan |  0.761833 |
| fixed_model_budget_qwen3.5-flash-2026-02-23_0               | fixed_model_budget       | nan               | qwen3.5-flash-2026-02-23 | cheap                 | 0                 |   0.95     |   0.000343 |       9.91816 |             nan        |             nan |  0.749923 |
| budget_only_best_0                                          | budget_only              | nan               | nan                      | nan                   | 0                 |   0.933333 |   0.000867 |      11.6069  |             nan        |             nan |  0.69686  |
| fixed_model_budget_qwen3.5-flash-2026-02-23_1024            | fixed_model_budget       | nan               | qwen3.5-flash-2026-02-23 | cheap                 | 1024              |   0.9      |   0.000483 |      13.9868  |             nan        |             nan |  0.617852 |
| fixed_model_budget_qwen3.5-plus-2026-02-15_256              | fixed_model_budget       | nan               | qwen3.5-plus-2026-02-15  | mid                   | 256               |   0.95     |   0.00137  |      19.9549  |             nan        |             nan |  0.544052 |
| fixed_model_budget_qwen3.5-plus-2026-02-15_0                | fixed_model_budget       | nan               | qwen3.5-plus-2026-02-15  | mid                   | 0                 |   0.85     |   0.00141  |      20.3619  |             nan        |             nan |  0.435709 |
| fixed_model_budget_qwen3.5-plus-2026-02-15_1024             | fixed_model_budget       | nan               | qwen3.5-plus-2026-02-15  | mid                   | 1024              |   0.85     |   0.002866 |      42.8367  |             nan        |             nan | -0.021066 |

