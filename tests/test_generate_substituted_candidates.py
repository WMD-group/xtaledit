from __future__ import annotations

import gzip
import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pymatgen.core import Lattice, Structure

from src.wyckoff_match import precompute


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "generate_substituted_candidates.py"
    spec = importlib.util.spec_from_file_location(
        "generate_substituted_candidates", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hf_rocksalt() -> Structure:
    return Structure(
        Lattice.cubic(5.0),
        ["H"] * 4 + ["F"] * 4,
        [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
            [0.5, 0.5, 0.5],
        ],
    )


def test_main_generates_whole_orbit_substitution_and_protects_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    source = _hf_rocksalt()
    assert module._replacement_candidates([source]) == {
        "F": {1: ["H"]},
        "H": {1: ["F"]},
    }
    train_path = tmp_path / "train.pkl.gz"
    output_dir = tmp_path / "raw"
    with gzip.open(train_path, "wb") as file:
        pickle.dump([source], file)

    monkeypatch.setattr(module, "TRAIN_PATH", train_path)
    monkeypatch.setattr(module, "GEN_RAW_DIR", output_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(module.__file__), "--N", "1", "--p", "0.1"],
    )
    module.main()

    output_path = output_dir / "subst_p=0.1.pkl.gz"
    with gzip.open(output_path, "rb") as file:
        generated = pickle.load(file)  # noqa: S301

    result = generated[0]
    orbits = precompute(source, symprec=0.01).orbit_atom_indices
    changed = 0
    for orbit in orbits:
        original = {source[index].specie.symbol for index in orbit}
        replacement = {result[index].specie.symbol for index in orbit}
        assert len(original) == len(replacement) == 1
        if original != replacement:
            changed += 1
    assert changed >= 1
    assert [site.specie.symbol for site in source] == ["H"] * 4 + ["F"] * 4
    output = capsys.readouterr().out
    assert (
        f"Average fraction of orbits substituted: {changed / len(orbits):.6f}" in output
    )
    assert "Mean changed-atom modified-Pettifor cost: 1.000000" in output

    with pytest.raises(SystemExit, match="output exists"):
        module.main()

    monkeypatch.setattr(module, "GEN_RAW_DIR", tmp_path / "second_raw")
    module.main()
    with gzip.open(tmp_path / "second_raw" / output_path.name, "rb") as file:
        generated_again = pickle.load(file)  # noqa: S301
    assert [structure.as_dict() for structure in generated_again] == [
        structure.as_dict() for structure in generated
    ]


def test_substitute_samples_cost_before_element() -> None:
    module = _load_script()
    source = Structure(Lattice.cubic(3.0), ["O"], [[0, 0, 0]])

    class StubRandom:
        def random(self) -> float:
            return 0.0

        def choices(
            self, population: list[int], weights: list[int], k: int
        ) -> list[int]:
            assert population == [5, 9]
            assert weights == [module.COST_COUNTS[5], module.COST_COUNTS[9]]
            assert k == 1
            return [9]

        def choice(self, population: list[str]) -> str:
            assert population == ["N"]
            return "N"

    result, orbit_fraction, changed_atom_cost, changed_atoms = module._substitute(
        source,
        {"O": {5: ["F"], 9: ["N"]}},
        p=1.0,
        symprec=0.01,
        rng=StubRandom(),
    )

    assert result[0].specie.symbol == "N"
    assert orbit_fraction == 1.0
    assert changed_atom_cost == 9
    assert changed_atoms == 1


def test_parse_args_uses_notebook_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    monkeypatch.setattr(sys, "argv", [str(module.__file__)])

    args = module.parse_args()
    total = sum(module.COST_COUNTS.values())

    assert args.N == 10_000
    assert args.p == pytest.approx(0.3148565564765805)
    assert args.seed == 42
    assert total == 28_455
    assert sum(cost * count for cost, count in module.COST_COUNTS.items()) / total == (
        pytest.approx(6.251555086979441)
    )

    monkeypatch.setattr(sys, "argv", [str(module.__file__), "--seed", "7"])
    assert module.parse_args().seed == 7


@pytest.mark.parametrize(
    "arguments",
    [
        ["--N", "0"],
        ["--p", "0"],
        ["--p", "1.1"],
        ["--symprec", "0"],
    ],
)
def test_parse_args_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    module = _load_script()
    monkeypatch.setattr(sys, "argv", [str(module.__file__), *arguments])

    with pytest.raises(SystemExit, match="must"):
        module.parse_args()
