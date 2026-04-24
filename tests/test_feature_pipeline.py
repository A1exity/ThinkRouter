from __future__ import annotations

from thinkrouter.features import extract_query_features, make_feature_frame


def test_feature_pipeline_includes_surface_semantic_and_probe_features() -> None:
    features = extract_query_features("Write a Python function add(a, b) and assert it works.", "humaneval")

    assert "char_count" in features
    assert "embedding_hash_mean" in features
    assert "cheap_probe_confidence" in features


def test_make_feature_frame_preserves_extra_columns() -> None:
    frame = make_feature_frame([{"query": "Add 1 and 2.", "task_type": "gsm8k", "selected_model": "mock-cheap"}])

    assert "selected_model" in frame.columns
    assert "semantic_bucket" in frame.columns
