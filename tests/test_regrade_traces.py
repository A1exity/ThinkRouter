from __future__ import annotations

import pandas as pd

from thinkrouter.experiments.regrade_traces import regrade_dataframe


def test_regrade_dataframe_updates_extraction_error() -> None:
    df = pd.DataFrame(
        [
            {
                "task_type": "gsm8k",
                "output_text": "**Answer:**\nIt takes **160 minutes** (or 2 hours and 40 minutes).",
                "expected_answer": "160",
                "extracted_answer": "40",
                "score": 0.0,
                "is_correct": False,
            }
        ]
    )

    regraded = regrade_dataframe(df)

    assert regraded.iloc[0]["extracted_answer"] == "160"
    assert bool(regraded.iloc[0]["is_correct"])
    assert regraded.iloc[0]["score"] == 1.0