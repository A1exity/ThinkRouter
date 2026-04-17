# ThinkRouter Final Project Plan

## 1. Project Overview

**Project Name:** ThinkRouter  
**Subtitle:** Adaptive Thinking-Budget Router for Reasoning LLMs

### One-sentence definition
ThinkRouter is a routing system for reasoning LLMs that jointly decides **which model to use** and **how much thinking budget to allocate** for each query, with the goal of reducing inference cost and latency while preserving most of the accuracy of a strong always-max-thinking baseline.

### Core motivation
Modern reasoning models often show two opposite failure modes:
- **Overthinking:** simple queries consume unnecessary reasoning tokens, increasing cost and latency.
- **Underthinking:** difficult queries fail when the allocated reasoning budget is too small.

Existing routing systems usually focus on **model selection only**. ThinkRouter reframes routing as a **joint `(model × thinking_budget)` decision problem**.

### Project goal
Build a complete, demoable, resume-ready system that:
1. predicts query difficulty,
2. chooses a model and thinking budget,
3. executes the query,
4. evaluates output with verifiable graders,
5. reports the quality-cost-latency tradeoff.

---

## 2. Scope Control

This is a **focused MVP-style project**, not a large platform product.

### This project will do
- Text-only reasoning tasks
- Public benchmarks with verifiable answers
- A 3-model pool
- 4 fixed budget levels
- A lightweight difficulty estimator
- A lightweight budget predictor
- A joint routing policy
- A simple web demo and experiment report

### This project will NOT do
- Multimodal tasks
- Image editing benchmarks
- Judge calibration
- LLM-as-judge pipelines
- Complex regression infrastructure
- Retrieval-based router in the first version
- Multi-page frontend productization
- Too many providers or too many models

---

## 3. Final Problem Formulation

Given an input query `x`, ThinkRouter chooses a pair:

- `m` = model
- `b` = thinking budget

from the candidate set:

`(m, b) ∈ M × B`

with the objective of maximizing expected task quality under cost and latency constraints.

### Utility objective
A simple first-version utility function:

```text
U = α * predicted_accuracy - β * cost - γ * latency
```

Where:
- `predicted_accuracy` comes from historical train-set statistics and the budget predictor
- `cost` is estimated from token usage / API price
- `latency` is measured runtime

The router returns the `(model, budget)` pair with the highest utility.

---

## 4. Benchmark Suite

The benchmark suite is frozen to **three tasks only**.

### A. GSM8K
- Type: grade-school math reasoning
- Purpose: expose overthinking on relatively easier reasoning queries
- Metric: exact match after numeric normalization
- Sample count: **100**

### B. MATH-500
- Type: harder mathematical reasoning
- Purpose: expose underthinking and the need for larger budgets
- Metric: exact match after answer normalization
- Sample count: **100**

### C. HumanEval
- Type: code generation
- Purpose: measure how reasoning budget affects executable coding performance
- Metric: pass@1 using unit tests
- Sample count: **80**

### Total dataset size
- **280 examples total**

### Data split
Use a fixed split per benchmark:
- **Train: 60%**
- **Dev: 20%**
- **Test: 20%**

Approximate split:
- GSM8K: 60 / 20 / 20
- MATH-500: 60 / 20 / 20
- HumanEval: 48 / 16 / 16

### Important rule
Only **train** data may be used to build difficulty and budget predictors.  
**Dev** is for tuning.  
**Test** is for final one-shot reporting only.

---

## 5. Model Pool

The model pool is frozen to **three models**.

### M1: Cheap local model
**Qwen3-4B**
- Role: low-cost baseline and cheap router option
- Requirement: should support thinking on/off or prompt-controlled shallow/deep reasoning

### M2: Mid-tier reasoning model
**DeepSeek-R1-Distill-Qwen-7B** or **DeepSeek-R1-Distill-Qwen-14B**
- Role: middle layer for tradeoff
- Use local deployment if hardware allows, otherwise use API

### M3: Strong frontier reasoning model
Choose **one** strong API model only:
- GPT family **or**
- Claude family **or**
- Gemini family

### Model pool rule
- The project uses **exactly 3 models** in the main results.
- Do not expand the pool during Week 1.
- Optional extra models may be added later only as supplementary comparison, not as part of the core MVP.

---

## 6. Thinking Budget Levels

Budget levels are frozen to **4 discrete options**:

- **0** → no explicit thinking / no-think mode
- **256**
- **1024**
- **4096**

If a model does not support exact token-level control, simulate the levels with prompt-based modes:
- `0` → no-think
- `256` → brief
- `1024` → medium
- `4096` → deep

Externally, all of them should still be presented as **budget levels**.

---

## 7. Evaluation Protocol

The project uses **verifiable-first evaluation only**.

### GSM8K / MATH-500
- Extract final answer
- Normalize numeric or symbolic form
- Use exact match

### HumanEval
- Run official or simplified unit tests
- Use pass@1

### Explicitly excluded
- No LLM-as-judge
- No rubric grader
- No judge disagreement handling
- No judge calibration

This keeps the results methodologically clean and much easier to defend.

---

## 8. Core System Modules

The project contains **three algorithmic modules** and several supporting system modules.

### 8.1 Difficulty Estimator

#### Input
- Query text

#### Output
- Difficulty label: `easy`, `medium`, or `hard`

#### Implementation
Use a lightweight ML pipeline:
- sentence-transformers embedding
- simple handcrafted features:
  - input length
  - digit density
  - code markers
  - math symbols
  - punctuation / structural cues
- classifier:
  - Logistic Regression, LightGBM, or XGBoost

#### Rule
Do **not** fine-tune a language model in V1.

---

### 8.2 Budget Predictor

#### Input
- task type
- predicted difficulty
- model ID

#### Output
- one of the 4 budget levels: `0 / 256 / 1024 / 4096`

#### Implementation
Use a **classifier**, not a regression model, for the first version.

This is intentionally simpler and easier to stabilize than quantile regression.

---

### 8.3 Joint Policy Engine

#### Input
- difficulty estimate
- candidate models
- candidate budgets
- historical train-set performance statistics
- cost and latency statistics

#### Output
- selected `(model, budget)` pair
- routing explanation

#### First-version policy
1. Predict difficulty.
2. Estimate the suitable budget.
3. Compute utility for feasible `(model, budget)` combinations.
4. Select the highest-utility option.

This module should remain simple, explainable, and easy to visualize.

---

## 9. Supporting System Modules

### 9.1 Model Adapter Layer
Responsible for:
- local model calls via vLLM
- API model calls
- unified request / response schema
- budget-mode translation

### 9.2 Evaluator Layer
Responsible for:
- GSM8K grader
- MATH grader
- HumanEval runner

### 9.3 Trace Store
Use SQLite to store:
- query
- task type
- selected model
- selected budget
- output text
- score / pass result
- token usage
- cost
- latency
- timestamp

### 9.4 Demo UI
A simple Streamlit app that shows:
- input query
- selected route
- model output
- score
- latency
- cost
- comparison against selected baselines

---

## 10. Baselines

Only **5 baselines / methods** will be reported in the main results.

### Baseline 1: Always Cheap + No Thinking
- model = M1
- budget = 0
- Purpose: lower-cost lower-bound baseline

### Baseline 2: Always Strong + Max Budget
- model = M3
- budget = 4096
- Purpose: high-quality upper baseline

### Baseline 3: Always Strong + Fixed Medium Budget
- model = M3
- budget = 1024
- Purpose: strong fixed-budget production-style baseline

### Baseline 4: Model-only Router
- chooses model only
- budget fixed globally
- Purpose: direct comparison with routing methods that ignore budget allocation

### Baseline 5: ThinkRouter
- joint `(model × budget)` routing
- main proposed method

### Optional: Oracle
- theoretical upper bound using held-out truth information
- optional, not required for MVP

---

## 11. Main Metrics

The final report is frozen to the following metrics:

1. **Accuracy**
2. **Average cost per query**
3. **p95 latency**
4. **Cost reduction vs Always Strong + Max Budget**
5. **Accuracy drop vs Always Strong + Max Budget**

### Required figures

#### Figure 1: Cost–Accuracy Pareto plot
This is the most important summary figure.

#### Figure 2: Per-benchmark comparison chart
Separate results for:
- GSM8K
- MATH-500
- HumanEval

### Optional figure
#### Figure 3: Budget usage distribution
Show how the router allocates budget across difficulty levels.

---

## 12. Technical Stack

The stack is fixed as follows.

### Backend
- **FastAPI**

### Frontend
- **Streamlit**

### Database
- **SQLite**

### Analysis
- **Pandas**
- **DuckDB** (optional)

### Local inference
- **vLLM**

### ML components
- **scikit-learn**
- **LightGBM**
- **sentence-transformers**

### Visualization
- **matplotlib** or **plotly**

This stack is intentionally lightweight and implementation-oriented.

---

## 13. Repository Structure

```text
thinkrouter/
├── app/
│   ├── api.py
│   ├── router.py
│   ├── models.py
│   ├── budgets.py
│   ├── evaluators.py
│   └── store.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── experiments/
│   ├── run_grid.py
│   ├── train_difficulty.py
│   ├── train_budget.py
│   ├── eval_baselines.py
│   └── make_plots.py
├── ui/
│   └── streamlit_app.py
├── results/
│   ├── tables/
│   ├── figures/
│   └── traces/
├── README.md
├── METHOD.md
└── RESULTS.md
```

---

## 14. Execution Plan

The project is planned as a **3-week delivery cycle**.

## Week 1 — End-to-End Pipeline

### Goal
Get a complete minimal loop running for any `(model, budget)` pair.

### Tasks
- prepare benchmark subsets and train/dev/test splits
- build 3 model adapters
- implement 4 budget levels
- implement GSM8K / MATH / HumanEval evaluators
- build SQLite trace store
- create first Streamlit page

### Week 1 success criterion
For at least one query, the system can:
- run any selected model
- apply any selected budget level
- return output
- evaluate correctness
- store cost and latency in SQLite
- display the full record in Streamlit

---

## Week 2 — Router Core

### Goal
Build a non-trivial routing system that beats at least two baselines on the dev set.

### Tasks
- run the training grid over train split
- train difficulty estimator
- train budget classifier
- implement joint policy engine
- evaluate on dev set
- inspect failure cases

### Week 2 success criterion
ThinkRouter shows a meaningful quality-cost tradeoff improvement over:
- Always Cheap + No Thinking
- at least one fixed strong baseline

---

## Week 3 — Final Results

### Goal
Produce stable final test results and project-ready artifacts.

### Tasks
- run final one-shot test evaluation
- generate Pareto plot
- generate per-benchmark comparison charts
- write failure analysis notes
- finalize RESULTS.md
- polish demo UI

### Week 3 success criterion
You have:
- final tables
- final figures
- test-set results
- a working demo page
- material ready for README and resume usage

---

## 15. Stop Condition

The project is considered **complete enough** when all 4 conditions are satisfied:

1. ThinkRouter significantly reduces cost compared with **Always Strong + Max Budget**.
2. Accuracy loss remains acceptable.
3. At least one clear Pareto figure is produced.
4. The Streamlit demo can show a single query’s routing decision and result comparison.

Anything beyond this is a bonus, not a requirement.

---

## 16. Day-1 Mini Goal

The very first implementation milestone should be:

> Run **20 GSM8K samples** with **2 models × 3 budget levels**, store output, correctness, cost, and latency in SQLite, and visualize the records in Streamlit.

Do **not** start with the full benchmark suite or all model combinations.
Start with the smallest complete loop, then scale up.

---

## 17. Final Deliverables

By the end of the project, the required deliverables are:

1. **GitHub repository** with runnable code
2. **README.md** with project summary and quickstart
3. **METHOD.md** describing the routing method
4. **RESULTS.md** with final tables and figures
5. **Streamlit demo**
6. **Resume-ready project description**

### Optional but recommended
- short demo video
- technical blog post
- extra ablation results

---

## 18. Resume Positioning

### Suggested Chinese title
**ThinkRouter：推理大模型自适应思考预算路由系统**

### Suggested English title
**ThinkRouter — Adaptive Thinking-Budget Router for Reasoning LLMs**

### Resume angle
This project should be presented as:
- an **LLM systems + evaluation + efficiency** project,
- not as a generic chatbot or agent demo,
- and not as a reproduction of an existing benchmark.

The strongest positioning is:
> a production-style reasoning inference optimization project with a clear algorithmic core and verifiable evaluation.

---

## 19. Final Guidance

ThinkRouter should stay focused on one central claim:

> **Reasoning budget is a first-class routing variable.**

If the implementation starts drifting into too many engineering extras, cut them.
If the algorithm starts becoming too complicated, simplify it.
If the benchmark starts expanding too much, shrink it.

A smaller but sharper project is more valuable than a large but shallow one.
