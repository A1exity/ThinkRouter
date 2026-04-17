from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from thinkrouter.app.schemas import TraceRecord, model_copy_update


SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    task_type TEXT NOT NULL,
    selected_model TEXT NOT NULL,
    selected_budget INTEGER NOT NULL,
    output_text TEXT NOT NULL,
    score REAL NOT NULL,
    is_correct INTEGER NOT NULL,
    expected_answer TEXT,
    extracted_answer TEXT,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_s REAL NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL
);
"""


class TraceStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def insert_trace(self, record: TraceRecord) -> TraceRecord:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO traces (
                    query, task_type, selected_model, selected_budget, output_text,
                    score, is_correct, expected_answer, extracted_answer, prompt_tokens,
                    completion_tokens, total_tokens, cost_usd, latency_s, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.query,
                    record.task_type,
                    record.selected_model,
                    record.selected_budget,
                    record.output_text,
                    record.score,
                    int(record.is_correct),
                    record.expected_answer,
                    record.extracted_answer,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.cost_usd,
                    record.latency_s,
                    record.created_at.isoformat(),
                    json.dumps(record.metadata),
                ),
            )
            return model_copy_update(record, {"id": int(cursor.lastrowid)})

    def list_traces(self, limit: int = 100) -> list[TraceRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM traces ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [row_to_trace(row) for row in rows]

    def iter_traces(self) -> Iterable[TraceRecord]:
        with self._connect() as conn:
            for row in conn.execute("SELECT * FROM traces ORDER BY id"):
                yield row_to_trace(row)


def row_to_trace(row: sqlite3.Row) -> TraceRecord:
    data = dict(row)
    data["is_correct"] = bool(data["is_correct"])
    data["metadata"] = json.loads(data.get("metadata") or "{}")
    return TraceRecord(**data)

