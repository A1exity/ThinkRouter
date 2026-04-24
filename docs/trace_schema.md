# Trace Schema

Trace rows are stored in SQLite and exported to CSV.

Important fields:

- `query_id`
- `benchmark`
- `query_text`
- `selected_model`
- `selected_model_provider`
- `selected_model_tier`
- `selected_budget`
- `selected_budget_id`
- `budget_config`
- `output_text`
- `parsed_output`
- `is_correct`
- `judge_metadata`
- `cost_usd`
- `latency_s`
- `error_type`
- `route_confidence`
- `fallback_triggered`
- `fallback_reason`
- `provider_response_meta`
- `metadata`

Operational notes:

- failed provider calls are now recorded as failed traces instead of aborting the whole run
- API and experiments share the same request cache path unless caching is disabled
- `resume` keys are based on sample id, model id, and budget
