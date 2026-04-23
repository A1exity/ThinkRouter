from __future__ import annotations

from thinkrouter.app.schemas import TraceRecord
from thinkrouter.app.store import TraceStore


def test_trace_store_insert_and_list(tmp_path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    record = TraceRecord(
        query_id="sample-1",
        benchmark="gsm8k",
        query="What is 1 + 1?",
        task_type="gsm8k",
        selected_model="mock-cheap",
        selected_budget=0,
        output_text="Final answer: 2",
        parsed_output="2",
        score=1.0,
        is_correct=True,
        expected_answer="2",
        extracted_answer="2",
        error_type=None,
        total_tokens=5,
    )
    inserted = store.insert_trace(record)
    traces = store.list_traces()
    assert inserted.id == 1
    assert len(traces) == 1
    assert traces[0].is_correct is True
    assert traces[0].query_id == "sample-1"
    assert traces[0].selected_budget_id == "budget-0"
    assert traces[0].budget_config["legacy_budget"] == 0
