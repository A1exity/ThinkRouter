# METHOD

## Current MVP

ThinkRouter currently implements the first project milestone: a complete traceable loop for GSM8K-style numeric reasoning tasks.

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

## Model Adapters

The MVP includes two adapter types:

- `MockAdapter`: deterministic local adapter used for development and tests.
- `OpenAICompatibleAdapter`: calls `/v1/chat/completions` through the OpenAI Python client and supports OpenAI-compatible providers, including vLLM servers.

This keeps the local RTX 4050 machine out of the critical path. A lab server with 4090 GPUs can be connected later by exposing a vLLM OpenAI-compatible endpoint and setting `THINKROUTER_OPENAI_BASE_URL` plus model ids.

## Routing Policy

The current router is a heuristic policy, not the final trained router. It estimates difficulty from simple query features, evaluates utility for each model-budget pair, and selects the best option:

```text
U = alpha * estimated_accuracy - beta * estimated_cost - gamma * estimated_latency
```

The Week-2 implementation should replace heuristic estimates with train-split statistics, a learned difficulty estimator, and a learned budget classifier.

## Evaluation

The implemented GSM8K evaluator extracts the final numeric answer, normalizes commas and trailing decimals, and uses exact match. This follows the project's verifiable-first rule and avoids LLM-as-judge evaluation.
