from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from thinkrouter.app.schemas import ModelRequest, ModelResponse, model_to_dict


SCHEMA = """
CREATE TABLE IF NOT EXISTS request_cache (
    cache_key TEXT PRIMARY KEY,
    response_json TEXT NOT NULL
);
"""


class RequestCache:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def key_for(self, request: ModelRequest) -> str:
        payload = json.dumps(model_to_dict(request), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, request: ModelRequest) -> ModelResponse | None:
        cache_key = self.key_for(request)
        with self._connect() as conn:
            row = conn.execute("SELECT response_json FROM request_cache WHERE cache_key = ?", (cache_key,)).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        provider_meta = dict(data.get("provider_meta") or {})
        provider_meta["cache_hit"] = True
        data["provider_meta"] = provider_meta
        return ModelResponse(**data)

    def set(self, request: ModelRequest, response: ModelResponse) -> None:
        cache_key = self.key_for(request)
        payload = json.dumps(model_to_dict(response), sort_keys=True, ensure_ascii=True)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO request_cache (cache_key, response_json) VALUES (?, ?)",
                (cache_key, payload),
            )
