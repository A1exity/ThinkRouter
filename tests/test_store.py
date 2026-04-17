from __future__ import annotations

from thinkrouter.app.schemas import TraceRecord
from thinkrouter.app.store import TraceStore


def test_trace_store_insert_and_list(tmp_path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    record = TraceRecord(
        query="What is 1 + 1?",
        task_type="gsm8k",
        selected_model="mock-cheap",
        selected_budget=0,
        output_text="Final answer: 2",
        score=1.0,
        is_correct=True,
        expected_answer="2",
        extracted_answer="2",
        total_tokens=5,
    )
    inserted = store.insert_trace(record)
    traces = store.list_traces()
    assert inserted.id == 1
    assert len(traces) == 1
    assert traces[0].is_correct is True
