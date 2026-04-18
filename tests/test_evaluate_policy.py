from __future__ import annotations

import pandas as pd

from thinkrouter.experiments.evaluate_policy import evaluate_policies


def test_evaluate_policies_summarizes_fixed_oracle_and_utility(tmp_path) -> None:
    csv_path = tmp_path / "grid.csv"
    rows = []
    for sample_id, expected in [("s1", "1"), ("s2", "2")]:
        rows.append(
            {
                "id": len(rows) + 1,
                "query": "q",
                "task_type": "gsm8k",
                "selected_model": "model",
                "selected_budget": 0,
                "output_text": f"Final answer: {expected}",
                "score": 1.0,
                "is_correct": True,
                "expected_answer": expected,
                "extracted_answer": expected,
                "total_tokens": 10,
                "cost_usd": 0.1,
                "latency_s": 1.0,
                "metadata": f"{{'sample_id': '{sample_id}', 'split': 'dev'}}",
            }
        )
        rows.append(
            {
                "id": len(rows) + 1,
                "query": "q",
                "task_type": "gsm8k",
                "selected_model": "model",
                "selected_budget": 1024,
                "output_text": "Final answer: 0",
                "score": 0.0,
                "is_correct": False,
                "expected_answer": expected,
                "extracted_answer": "0",
                "total_tokens": 20,
                "cost_usd": 0.2,
                "latency_s": 2.0,
                "metadata": f"{{'sample_id': '{sample_id}', 'split': 'dev'}}",
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary, stats = evaluate_policies(str(csv_path))

    policies = set(summary["policy"])
    assert "fixed_budget_0" in policies
    assert "fixed_budget_1024" in policies
    assert "oracle_lowest_cost_correct" in policies
    assert "aggregate_utility_budget_0" in policies
    assert summary.loc[summary["policy"] == "fixed_budget_0", "accuracy"].iloc[0] == 1.0
    assert summary.loc[summary["policy"] == "fixed_budget_1024", "accuracy"].iloc[0] == 0.0
    assert stats["utility"].notna().all()


def test_evaluate_policies_falls_back_to_id_without_metadata(tmp_path) -> None:
    csv_path = tmp_path / "grid.csv"
    pd.DataFrame(
        [
            {
                "id": "s1",
                "selected_model": "model",
                "selected_budget": 0,
                "is_correct": True,
                "cost_usd": 0.1,
                "latency_s": 1.0,
            }
        ]
    ).to_csv(csv_path, index=False)

    summary, _ = evaluate_policies(str(csv_path))

    assert summary.loc[summary["policy"] == "oracle_lowest_cost_correct", "n"].iloc[0] == 1
