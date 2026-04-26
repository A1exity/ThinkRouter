# METHOD

## System Definition

ThinkRouter is a routing system for verifiable reasoning tasks where the decision is:

```text
(model, budget)
```

The repository now targets the Phase 2 router stack as the main runtime path. The older `JointPolicyEngine` remains only as a legacy baseline.

## Frozen Official Protocol

The only official protocol is defined in:

- [`configs/official_protocol.json`](configs/official_protocol.json)
- [`thinkrouter/official_protocol.py`](thinkrouter/official_protocol.py)

Protocol settings:

| field | value |
| --- | --- |
| model pool | `qwen-flash,qwen-plus,qwen-max` |
| budgets | `0,256,1024` |
| benchmarks | `GSM8K`, `MATH-500`, `HumanEval` |
| split sizes | `60 train / 20 dev / 20 test` |
| routers | `threshold`, `logreg_joint`, `mlp_factorized`, `uncertainty_aware` |
| utility | `accuracy - 5 * cost - 0.02 * latency` |

All future formal results are expected to come from the official pipeline, not from ad hoc `dev5/dev10/dev20` slices.

## Budgets

Budgets are structured configurations rather than plain integers. The active public set is:

- `0`
- `256`
- `1024`

Each budget compiles to a structured config with:

- effort level
- max output token cap
- provider control fields when available

## Model Layer

The current implementation uses a unified adapter boundary with a Qwen three-tier pool:

- `qwen-flash`
- `qwen-plus`
- `qwen-max`

Adapters normalize:

- query input
- budget config
- token accounting
- latency
- provider metadata
- recoverable runtime errors

## Feature Pipeline

The feature pipeline now has three parts:

1. Surface features
2. Semantic features
3. Cheap-probe features

### Surface Features

Surface features include:

- character count
- word count
- digit density
- math symbol count
- punctuation count
- code marker count
- average word length

### Semantic Features

The semantic path is now based on real sentence embeddings:

- backend: `sentence-transformers`
- default model: `sentence-transformers/all-MiniLM-L6-v2`

The active semantic columns are dense embedding dimensions:

- `semantic_embedding_00`
- ...
- `semantic_embedding_15`

If the embedding model cannot be loaded in a local environment, the code falls back to a lexical SVD representation. That fallback is a resilience path only; the intended official backend is sentence-transformers.

### Cheap-Probe Features

Cheap-probe features remain lightweight heuristics:

- difficulty score
- confidence
- consistency

They are auxiliary features, not the semantic backbone.

## Routers

The Phase 2 router stack contains:

- `threshold`: interpretable baseline
- `logreg_joint`: joint `(model, budget)` classifier
- `mlp_factorized`: separate model head and budget head
- `uncertainty_aware`: factorized router with confidence-triggered fallback

The learned routers are trained on `train` grids, selected/calibrated on `dev`, and intended to be reported on `test`.

## Online Routing

`POST /run` now defaults to the Phase 2 runtime stack when `use_router=true` and no router name is supplied. The runtime default is:

```text
uncertainty_aware
```

`legacy_joint_policy` remains available only for comparison and regression checks.

## Evaluation

Evaluation remains deterministic:

- GSM8K: numeric extraction and normalization
- MATH / MATH-500: boxed/final-expression extraction plus lightweight normalization
- HumanEval: code extraction and deterministic test execution

The repository does not use LLM-as-judge in the official scoring path.

## Trace And Replay

Every run is stored as a trace with:

- query and benchmark metadata
- selected model and budget
- raw and parsed outputs
- correctness and judge metadata
- tokens, cost, latency
- route confidence and fallback metadata

Offline replay supports:

- fixed model-budget baselines
- model-only baseline
- budget-only baseline
- aggregate utility baseline
- safe fallback baseline
- Phase 2 router replay

## Current Project State

Methodologically, the system is now aligned with the completed official protocol and its final report artifacts. The remaining work, if any, is extension work rather than unfinished core routing/evaluation work.
