from __future__ import annotations

from typing import Any

from src import energy


def test_mace_model_identifier_is_shared_by_worker_and_sequential_paths(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_mace_mp(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(energy, "mace_mp", fake_mace_mp)
    monkeypatch.setattr(energy, "_relax_batch", lambda *args, **kwargs: [])
    monkeypatch.setattr(energy.torch.cuda, "device_count", lambda: 0)

    energy.relax_structures([], device="cpu")
    energy._gpu_worker(([], "cuda:0", 1e-3, 1000, 64))

    assert [call["model"] for call in calls] == [
        "medium-mpa-0",
        "medium-mpa-0",
    ]
