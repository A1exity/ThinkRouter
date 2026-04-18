from __future__ import annotations

import joblib
import pandas as pd

from thinkrouter.experiments.learned_policy_router import (
    derive_policy_training_examples,
    evaluate_learned_policy,
    replay_learned_policy,
    train_learned_policy,
)


def _write_policy_grid(path) -> None:
    rows = []
    samples = [
        ("s1", "Add 1 and 2.", 0),
        ("s2", "A box has 10 red balls and 12 blue balls. How many total?", 256),
        ("s3", "Compute 2 + 2.", 0),
    ]
    for sample_id, query, best_budget in samples:
        for budget in [0, 256, 1024]:
            is_correct = budget == best_budget or (sample_id == "s1" and budget == 256)
            rows.append(
                {
                    "id": f"{sample_id}-{budget}",
                    "query": query,
                    "task_type": "gsm8k",
                    "selected_model": "model",
                    "selected_budget": budget,
                    "is_correct": is_correct,
                    "score": 1.0 if is_correct else 0.0,
                    "cost_usd": 0.01 + budget / 100000,
                    "latency_s": 1.0 + budget / 1000,
                    "metadata": f"{{'sample_id': '{sample_id}', 'split': 'train'}}",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_derive_policy_training_examples_uses_best_utility_budget(tmp_path) -> None:
    csv_path = tmp_path / "grid.csv"
    _write_policy_grid(csv_path)

    examples = derive_policy_training_examples(pd.read_csv(csv_path))

    assert len(examples) == 3
    assert set(examples["selected_budget"]).issubset({0, 256, 1024})
    assert examples.loc[examples["sample_id"] == "s1", "selected_budget"].iloc[0] == 0


def test_train_save_load_and_replay_learned_policy(tmp_path) -> None:
    csv_path = tmp_path / "grid.csv"
    model_path = tmp_path / "policy.joblib"
    _write_policy_grid(csv_path)

    artifact = train_learned_policy(str(csv_path))
    joblib.dump(artifact, model_path)
    summary, selected = evaluate_learned_policy(str(csv_path), str(model_path))
    replayed = replay_learned_policy(artifact, pd.read_csv(csv_path))

    assert summary["policy"].iloc[0] == "learned_policy"
    assert summary["n"].iloc[0] == 3
    assert set(selected["predicted_budget"]).issubset({0, 256, 1024})
    assert len(replayed) == 3
