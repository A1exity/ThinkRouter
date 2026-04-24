# Failure Taxonomy

ThinkRouter uses a small fixed failure taxonomy for post-hoc analysis.

Current categories:

- `parse_error`
- `execution_error`
- `timeout`
- `empty_output`
- `malformed_answer`
- `wrong_answer`
- `answer_format_extraction_error`
- `no_final_answer`
- `no_numeric_answer`
- `missing_expected_answer`

Higher-level routing failure themes:

- difficult query sent to an overly cheap model
- easy query over-routed to an expensive model
- budget increase with no accuracy gain
- confidence too high on a wrong route
- evaluator extraction failure hiding a correct answer

This taxonomy is used by the failure browser and should remain stable across future result refreshes.
