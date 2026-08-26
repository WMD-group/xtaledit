from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.config import ROOT


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", ["wyckoff_match", "substitute_structures"])
def test_cost_config_accepts_cs(monkeypatch, tmp_path, script: str) -> None:
    module = _load_script(script)
    config = tmp_path / "config.yaml"
    config.write_text("model: test\ncost: cs\n")
    monkeypatch.setattr(sys, "argv", [str(module.__file__), str(config)])

    assert module.parse_args().cost == "cs"


def test_top_k_entries_ranks_by_cs() -> None:
    module = _load_script("substitute_structures")
    matches = [
        SimpleNamespace(
            idx1=0,
            idx2=1,
            cost_uniform=0.1,
            cost_mod_petti=0.2,
            cost_cs=0.4,
        ),
        SimpleNamespace(
            idx1=0,
            idx2=2,
            cost_uniform=0.9,
            cost_mod_petti=0.8,
            cost_cs=0.3,
        ),
    ]

    entries = module._top_k_entries(
        matches,
        1,
        "cost_cs",
        [],
        [],
        lambda match, _gen, _train: match.idx2,
    )

    assert len(entries) == 1
    assert entries[0].train_idx == 2
    assert entries[0].cost_cs == 0.3
