from __future__ import annotations

import pandas as pd

from thinkrouter.experiments.make_failure_taxonomy import build_failure_taxonomy_report


def test_build_failure_taxonomy_report_writes_outputs(tmp_path) -> None:
    csv_path = tmp_path / "grid.csv"
    pd.DataFrame(
        [
            {
                "id": 1,
                "query": "Add 1 and 2.",
                "task_type": "gsm8k",
                "selected_model": "cheap",
                "selected_budget": 0,
                "is_correct": False,
                "cost_usd": 0.1,
                "latency_s": 1.0,
                "expected_answer": "3",
                "extracted_answer": "",
                "output_text": "I think the answer is three.",
                "metadata": "{'sample_id': 's1'}",
            }
        ]
    ).to_csv(csv_path, index=False)

    report = build_failure_taxonomy_report(str(csv_path), str(tmp_path / "taxonomy.csv"), str(tmp_path / "taxonomy.md"))

    assert "count" in report.columns
    assert (tmp_path / "taxonomy.csv").exists()
    assert (tmp_path / "taxonomy.md").exists()
