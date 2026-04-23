from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from thinkrouter.app.budgets import budget_to_dict, compile_budget_config
from thinkrouter.app.schemas import TraceRecord, model_copy_update


SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_version INTEGER NOT NULL DEFAULT 2,
    query_id TEXT,
    benchmark TEXT,
    query TEXT NOT NULL,
    query_text TEXT,
    task_type TEXT NOT NULL,
    selected_model TEXT NOT NULL,
    selected_budget INTEGER NOT NULL,
    selected_budget_id TEXT,
    budget_config TEXT NOT NULL DEFAULT '{}',
    output_text TEXT NOT NULL,
    parsed_output TEXT,
    score REAL NOT NULL,
    is_correct INTEGER NOT NULL,
    expected_answer TEXT,
    extracted_answer TEXT,
    judge_metadata TEXT NOT NULL DEFAULT '{}',
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_s REAL NOT NULL,
    error_type TEXT,
    router_features TEXT NOT NULL DEFAULT '{}',
    route_reason TEXT,
    candidate_set_signature TEXT,
    prompt_template_version TEXT,
    provider_response_meta TEXT NOT NULL DEFAULT '{}',
    route_confidence REAL,
    fallback_triggered INTEGER NOT NULL DEFAULT 0,
    fallback_reason TEXT,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL
);
"""

REQUIRED_COLUMNS: dict[str, str] = {
    "schema_version": "INTEGER NOT NULL DEFAULT 2",
    "query_id": "TEXT",
    "benchmark": "TEXT",
    "query_text": "TEXT",
    "selected_budget_id": "TEXT",
    "budget_config": "TEXT NOT NULL DEFAULT '{}'",
    "parsed_output": "TEXT",
    "judge_metadata": "TEXT NOT NULL DEFAULT '{}'",
    "error_type": "TEXT",
    "router_features": "TEXT NOT NULL DEFAULT '{}'",
    "route_reason": "TEXT",
    "candidate_set_signature": "TEXT",
    "prompt_template_version": "TEXT",
    "provider_response_meta": "TEXT NOT NULL DEFAULT '{}'",
    "route_confidence": "REAL",
    "fallback_triggered": "INTEGER NOT NULL DEFAULT 0",
    "fallback_reason": "TEXT",
}


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
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(traces)").fetchall()}
            for column, ddl in REQUIRED_COLUMNS.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE traces ADD COLUMN {column} {ddl}")

    def insert_trace(self, record: TraceRecord) -> TraceRecord:
        budget_config = record.budget_config or budget_to_dict(compile_budget_config(record.selected_budget))
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO traces (
                    schema_version, query_id, benchmark, query, query_text, task_type, selected_model, selected_budget,
                    selected_budget_id, budget_config, output_text, parsed_output, score, is_correct, expected_answer,
                    extracted_answer, judge_metadata, prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_s,
                    error_type, router_features, route_reason, candidate_set_signature, prompt_template_version,
                    provider_response_meta, route_confidence, fallback_triggered, fallback_reason, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.schema_version,
                    record.query_id or record.metadata.get("sample_id"),
                    record.benchmark,
                    record.query,
                    record.query_text or record.query,
                    record.task_type,
                    record.selected_model,
                    record.selected_budget,
                    record.selected_budget_id or budget_config.get("budget_id"),
                    json.dumps(budget_config),
                    record.output_text,
                    record.parsed_output,
                    record.score,
                    int(record.is_correct),
                    record.expected_answer,
                    record.extracted_answer,
                    json.dumps(record.judge_metadata),
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.cost_usd,
                    record.latency_s,
                    record.error_type,
                    json.dumps(record.router_features),
                    record.route_reason,
                    record.candidate_set_signature,
                    record.prompt_template_version or budget_config.get("prompt_template_version"),
                    json.dumps(record.provider_response_meta),
                    record.route_confidence,
                    int(record.fallback_triggered),
                    record.fallback_reason,
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
    data["fallback_triggered"] = bool(data.get("fallback_triggered", 0))
    for field in ("budget_config", "judge_metadata", "router_features", "provider_response_meta", "metadata"):
        data[field] = json.loads(data.get(field) or "{}")
    if not data.get("query_text"):
        data["query_text"] = data.get("query")
    if not data.get("query_id"):
        data["query_id"] = str(data["metadata"].get("sample_id") or data["metadata"].get("query_id") or "")
    if not data.get("selected_budget_id"):
        budget_config = data.get("budget_config") or budget_to_dict(compile_budget_config(int(data["selected_budget"])))
        data["budget_config"] = budget_config
        data["selected_budget_id"] = budget_config.get("budget_id")
    return TraceRecord(**data)

