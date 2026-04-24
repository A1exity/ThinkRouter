from __future__ import annotations

from thinkrouter.experiments import prepare_data


def test_load_humaneval_samples_builds_metadata(monkeypatch) -> None:
    fake_dataset = {
        "test": [
            {
                "prompt": "Write a function add(a, b).",
                "entry_point": "add",
                "canonical_solution": "def add(a, b):\n    return a + b\n",
                "test": "assert add(1, 2) == 3\n",
            }
            for _ in range(3)
        ]
    }

    monkeypatch.setattr(prepare_data, "_load_first_available_dataset", lambda _: fake_dataset)
    samples = prepare_data.load_humaneval_samples(train_count=1, dev_count=1, test_count=1)

    assert [sample.split for sample in samples] == ["train", "dev", "test"]
    assert samples[0].metadata["entry_point"] == "add"
    assert samples[0].metadata["benchmark"] == "humaneval"
