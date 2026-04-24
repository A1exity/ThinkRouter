# ThinkRouter Phase 2 Report

This report ranks the integrated baseline and Phase 2 router summaries by utility.

- Best overall policy: `fixed_model_budget_qwen3-max-2026-01-23_256`
- Best overall utility: `-0.050306`
- Best Phase 2 policy: `phase2_logreg_joint`
- Best Phase 2 utility: `-0.050306`

| policy                                                          | policy_family            | router_name       | selected_model           | selected_model_tier   |   selected_budget |   accuracy |   avg_cost |   avg_latency |   avg_route_confidence |   fallback_rate |   utility |
|:----------------------------------------------------------------|:-------------------------|:------------------|:-------------------------|:----------------------|------------------:|-----------:|-----------:|--------------:|-----------------------:|----------------:|----------:|
| fixed_model_budget_qwen3-max-2026-01-23_256                     | fixed_model_budget       | nan               | qwen3-max-2026-01-23     | strong                |               256 |          0 |   0.000358 |       2.42581 |               nan      |             nan | -0.050306 |
| aggregate_utility_qwen3-max-2026-01-23_256                      | joint_aggregate_utility  | nan               | qwen3-max-2026-01-23     | strong                |               256 |          0 |   0.000358 |       2.42581 |               nan      |             nan | -0.050306 |
| aggregate_utility_safe_fallback_qwen3-max-2026-01-23_256_to_256 | joint_safe_fallback      | nan               | qwen3-max-2026-01-23     | strong                |               256 |          0 |   0.000358 |       2.42581 |               nan      |             nan | -0.050306 |
| model_only_best_qwen3-max-2026-01-23                            | model_only               | nan               | qwen3-max-2026-01-23     | strong                |               nan |          0 |   0.000358 |       2.42581 |               nan      |             nan | -0.050306 |
| phase2_logreg_joint                                             | phase2_logreg_joint      | logreg_joint      | qwen3-max-2026-01-23     | strong                |               256 |          0 |   0.000358 |       2.42581 |                 1      |               0 | -0.050306 |
| phase2_mlp_factorized                                           | phase2_mlp_factorized    | mlp_factorized    | qwen3-max-2026-01-23     | strong                |               256 |          0 |   0.000358 |       2.42581 |                 1      |               0 | -0.050306 |
| phase2_threshold                                                | phase2_threshold         | threshold         | qwen3-max-2026-01-23     | strong                |               256 |          0 |   0.000358 |       2.42581 |                 0.8125 |               0 | -0.050306 |
| phase2_uncertainty_aware                                        | phase2_uncertainty_aware | uncertainty_aware | qwen3-max-2026-01-23     | strong                |               256 |          0 |   0.000358 |       2.42581 |                 1      |               0 | -0.050306 |
| fixed_model_budget_qwen3.5-flash-2026-02-23_256                 | fixed_model_budget       | nan               | qwen3.5-flash-2026-02-23 | cheap                 |               256 |          0 |   0.000616 |      21.9966  |               nan      |             nan | -0.44301  |
| budget_only_best_256                                            | budget_only              | nan               | nan                      | nan                   |               256 |          0 |   0.001987 |      33.8115  |               nan      |             nan | -0.686165 |
| fixed_model_budget_qwen3.5-plus-2026-02-15_256                  | fixed_model_budget       | nan               | qwen3.5-plus-2026-02-15  | mid                   |               256 |          0 |   0.004987 |      77.0122  |               nan      |             nan | -1.56518  |

