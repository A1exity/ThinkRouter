from __future__ import annotations

import joblib
import pandas as pd

from thinkrouter.app.router import JointPolicyEngine, SklearnBudgetPredictor, SklearnDifficultyEstimator
from thinkrouter.experiments.train_budget import derive_budget_training_examples, train_budget_from_traces
from thinkrouter.experiments.train_difficulty import train_difficulty_from_traces


def _write_trace_csv(path) -> None:
    rows = []
    samples = [
        ("Add 2 and 3.", "gsm8k", "5"),
        ("A shop sold 12 red pens and 15 blue pens, then returned 4. How many remain?", "gsm8k", "23"),
        ("Solve x^2 + 5*x + 6 = 0 and report the larger root.", "math", "-2"),
        ("Write a Python function def add(a, b): return the sum and assert it works.", "humaneval", "pass"),
    ]
    for query, task_type, expected in samples:
        for model in ["mock-cheap", "mock-strong"]:
            for budget in [0, 256, 1024]:
                is_correct = budget == 0 or model == "mock-strong"
                rows.append(
                    {
                        "query": query,
                        "task_type": task_type,
                        "selected_model": model,
                        "selected_budget": budget,
                        "output_text": f"Final answer: {expected}",
                        "score": 1.0 if is_correct else 0.0,
                        "is_correct": is_correct,
                        "expected_answer": expected,
                        "extracted_answer": expected if is_correct else "0",
                        "prompt_tokens": 10,
                        "completion_tokens": 5 + budget // 256,
                        "total_tokens": 15 + budget // 256,
                        "cost_usd": 0.0 if model == "mock-cheap" else 0.0001 * (1 + budget / 1024),
                        "latency_s": 0.01 + budget / 100000,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_trainers_save_load_and_route(tmp_path) -> None:
    csv_path = tmp_path / "traces.csv"
    difficulty_path = tmp_path / "difficulty.joblib"
    budget_path = tmp_path / "budget.joblib"
    _write_trace_csv(csv_path)

    difficulty_model = train_difficulty_from_traces(str(csv_path))
    budget_model = train_budget_from_traces(str(csv_path))
    joblib.dump(difficulty_model, difficulty_path)
    joblib.dump(budget_model, budget_path)

    difficulty = SklearnDifficultyEstimator(difficulty_path)
    budget = SklearnBudgetPredictor(budget_path)
    engine = JointPolicyEngine(["mock-cheap", "mock-strong"], difficulty_estimator=difficulty, budget_predictor=budget)
    decision = engine.route("Add 1 and 2.", "gsm8k")

    assert decision.difficulty in {"easy", "medium", "hard"}
    assert decision.budget in {0, 256, 1024, 4096}
    assert "budget_hint" in decision.explanation


def test_derive_budget_examples_picks_lowest_cost_correct_budget(tmp_path) -> None:
    csv_path = tmp_path / "traces.csv"
    _write_trace_csv(csv_path)
    examples = derive_budget_training_examples(pd.read_csv(csv_path))

    assert not examples.empty
    assert set(examples["selected_budget"]).issubset({0, 256, 1024, 4096})
    assert {"query", "task_type", "selected_model", "difficulty", "selected_budget"}.issubset(examples.columns)
