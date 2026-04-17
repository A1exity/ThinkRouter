# METHOD

## Current MVP

ThinkRouter currently implements the first project milestone: a complete traceable loop for GSM8K-style numeric reasoning tasks, plus the first trainable router and frozen-split experiment components for Week 2.

The system has four stable boundaries:

- model adapters convert a `(query, model_id, budget)` request into a normalized response,
- evaluators convert model output plus expected answer into a verifiable score,
- the trace store persists model, budget, output, correctness, token, cost, and latency fields,
- the policy engine chooses `(model, budget)` when router mode is enabled.

## Thinking Budgets

The public budget levels are fixed:

- `0`: direct/no explicit thinking,
- `256`: brief reasoning,
- `1024`: medium reasoning,
- `4096`: deep reasoning.

For providers that do not expose exact reasoning-token control, the adapter maps each budget to a prompt instruction. The external API still records the selected discrete budget.

## Seed Splits

The repository now includes deterministic seed samples for three task types:

- `gsm8k`: arithmetic word problems,
- `math`: compact algebra, arithmetic, and geometry prompts,
- `humaneval`: code-generation-style prompts with exact-match seed answers.

These are local seed examples for pipeline validation, not official benchmark subsets. They provide stable `train`, `dev`, and `test` splits so the router training code can avoid training on dev/test traces before official datasets are wired in.

The general grid runner records split metadata in each trace and supports task, split, model, budget, and limit filters. It can read either the built-in seed samples or an external benchmark JSONL file with `sample_id`, `task_type`, `split`, `query`, and `expected_answer` fields.

## Benchmark JSONL Interface

External datasets should be converted to one JSON object per line:

```json
{"sample_id":"gsm8k_train_001","task_type":"gsm8k","split":"train","query":"...","expected_answer":"..."}
```

`prepare_data.py` exports the built-in seed suite and a small official GSM8K subset to this format. For GSM8K it reads `openai/gsm8k`, extracts the final numeric answer after `####`, and creates a fixed 60 train / 20 dev / 20 test subset by default. `run_grid.py --input` consumes the same format, so MATH-500 and HumanEval loaders can be added without changing the model, evaluator, trace store, or router training layers.

## Model Adapters

The MVP includes two adapter types and a real-endpoint smoke-test path:

- `MockAdapter`: deterministic local adapter used for development and tests.
- `OpenAICompatibleAdapter`: calls `/v1/chat/completions` through the OpenAI Python client and supports OpenAI-compatible providers, including vLLM servers.
- `smoke_real_model.py`: validates endpoint configuration by default and only calls the provider when `--run` is passed.

This keeps the local RTX 4050 machine out of the critical path during development while preserving a controlled path to real LLM calls. A lab server with 4090 GPUs can be connected later by exposing a vLLM OpenAI-compatible endpoint and setting `THINKROUTER_OPENAI_BASE_URL` plus model ids.

## Query Features

The trainable router uses lightweight, explainable features instead of fine-tuning an LLM:

- character and word counts,
- digit count and digit density,
- math-symbol count,
- punctuation count,
- code-marker count,
- average word length,
- task type,
- selected model id for budget prediction.

## Difficulty Estimator

`train_difficulty.py` builds a sklearn classifier from trace CSVs. Current seed traces do not contain human difficulty labels, so the script derives pseudo-labels from query complexity features and task type. This is sufficient for testing the train-save-load loop; official benchmark traces should replace pseudo-labels with labels derived from train-set performance statistics.

The trained model is saved as a joblib file and can be loaded by setting:

```text
THINKROUTER_DIFFICULTY_MODEL_PATH=results/models/difficulty.joblib
```

If no model path is configured, the router falls back to the built-in heuristic difficulty estimator.

## Budget Predictor

`train_budget.py` derives a target budget for each `(query, task_type, model)` group by choosing the lowest-cost, lowest-latency correct trace. If no trace is correct, it falls back to the highest-scoring trace. The resulting sklearn classifier predicts one of the fixed budget levels.

The trained model is saved as a joblib file and can be loaded by setting:

```text
THINKROUTER_BUDGET_MODEL_PATH=results/models/budget.joblib
```

When loaded, the budget prediction is used as a soft hint inside the joint utility policy. It does not hard-force the selected budget.

## Routing Policy

The router estimates difficulty, optionally predicts a budget hint for each candidate model, evaluates utility for each model-budget pair, and selects the best option:

```text
U = alpha * estimated_accuracy - beta * estimated_cost - gamma * estimated_latency - hint_penalty
```

The current accuracy, cost, and latency estimates are still simple policy estimates. The next implementation step is to replace them with train-split aggregate statistics from real benchmark traces.

## Evaluation

The implemented GSM8K evaluator extracts the final numeric answer, normalizes commas and trailing decimals, and uses exact match. Non-GSM8K seed tasks currently use exact-match final-answer evaluation. This follows the project's verifiable-first rule and avoids LLM-as-judge evaluation.