# ThinkRouter

ThinkRouter is an adaptive thinking-budget router for reasoning LLMs. It treats the pair `(model, thinking_budget)` as the routing decision, then records quality, cost, and latency for verifiable reasoning tasks.

This repository currently implements the Day-1 MVP loop:

1. load 20 built-in GSM8K-style samples,
2. run 2 model configs across budgets `0`, `256`, and `1024`,
3. evaluate numeric exact match,
4. store traces in SQLite,
5. inspect or run examples through Streamlit or FastAPI.

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate thinkrouter
```

Copy the environment template if you want to use a real OpenAI-compatible endpoint:

```bash
copy .env.example .env
```

By default, the code works with mock models and does not require API keys.

## Configuration

Environment variables:

- `THINKROUTER_DB_PATH`: SQLite path. Default: `results/traces/thinkrouter.sqlite`.
- `THINKROUTER_OPENAI_BASE_URL`: OpenAI-compatible base URL.
- `THINKROUTER_OPENAI_API_KEY`: API key for that endpoint.
- `THINKROUTER_CHEAP_MODEL`: cheap model id. Defaults to `mock-cheap`.
- `THINKROUTER_STRONG_MODEL`: strong model id. Defaults to `mock-strong` in code, example env uses `gpt-4o-mini`.
- `THINKROUTER_CHEAP_COST_PER_1K`: estimated cheap model cost per 1K tokens.
- `THINKROUTER_STRONG_COST_PER_1K`: estimated strong model cost per 1K tokens.

Any model id starting with `mock` uses the local `MockAdapter`. Other model ids use the OpenAI-compatible adapter.

## Run Day-1 Grid

```bash
python -m thinkrouter.experiments.run_day1_grid --db results/traces/day1.sqlite --out results/tables/day1_grid.csv
```

This writes traces to SQLite and a CSV table for analysis.

## Run API

```bash
uvicorn thinkrouter.app.api:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /config`
- `POST /run`
- `GET /traces`

## Run Streamlit Demo

```bash
streamlit run thinkrouter/ui/streamlit_app.py
```

The demo lets you run a GSM8K-style query, choose a model and budget, optionally use the router, and inspect recent SQLite traces.

## Analysis Scripts

```bash
python -m thinkrouter.experiments.eval_baselines results/tables/day1_grid.csv
python -m thinkrouter.experiments.make_plots results/tables/day1_grid.csv
python -m thinkrouter.experiments.train_difficulty results/tables/day1_grid.csv
python -m thinkrouter.experiments.train_budget results/tables/day1_grid.csv
```

The training scripts are lightweight placeholders over trace CSVs. The later project stage should replace the Day-1 built-in samples with frozen GSM8K, MATH-500, and HumanEval splits.

## Test

```bash
pytest
```
