from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

PREPROCESS_PATH = Path(__file__).resolve().parent.parent / "scripts" / "preprocess.py"
SPEC = importlib.util.spec_from_file_location("preprocess", PREPROCESS_PATH)
assert SPEC is not None and SPEC.loader is not None
preprocess = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preprocess)


def test_compute_smact_validities_returns_bool_array() -> None:
    structures = [
        Structure(
            Lattice.cubic(3.0),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
    ]

    valid = preprocess.compute_smact_validities(structures)

    assert valid.dtype == np.bool_
    assert valid.tolist() == [True]


def test_gen_sources_defaults_to_all_raw_pkls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    b_path = raw_dir / "b.pkl.gz"
    a_path = raw_dir / "a.pkl.gz"
    txt_path = raw_dir / "ignore.txt"
    b_path.touch()
    a_path.touch()
    txt_path.touch()
    monkeypatch.setattr(preprocess, "GEN_RAW_DIR", raw_dir)

    assert preprocess._gen_sources() == [a_path, b_path]


def test_gen_sources_uses_one_given_file(tmp_path: Path) -> None:
    gen_file = tmp_path / "sample.pkl.gz"
    gen_file.touch()

    assert preprocess._gen_sources(gen_file) == [gen_file]


def test_gen_sources_rejects_non_pkl_gz(tmp_path: Path) -> None:
    gen_file = tmp_path / "sample.pkl"
    gen_file.touch()

    with pytest.raises(SystemExit, match="must end with .pkl.gz"):
        preprocess._gen_sources(gen_file)


def test_gen_sources_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="generated file not found"):
        preprocess._gen_sources(tmp_path / "missing.pkl.gz")


def test_gen_out_dir_requires_gen_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="gen_out_dir requires gen_file"):
        preprocess.preprocess_gen(gen_out_dir=tmp_path)


def test_preprocess_gen_uses_configured_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = tmp_path / "raw" / "model.pkl.gz"
    src.parent.mkdir()
    preprocess._save([], src)

    default_out_dir = tmp_path / "default"
    configured_out_dir = tmp_path / "configured"
    monkeypatch.setattr(preprocess, "GEN_OUT_DIR", default_out_dir)
    monkeypatch.setattr(preprocess, "relax_structures", lambda structures: ([], []))
    monkeypatch.setattr(preprocess, "_get_ppd", object)
    monkeypatch.setattr(
        preprocess,
        "compute_ehulls",
        lambda structures, energies, ppd: [],
    )

    preprocess.preprocess_gen(src, configured_out_dir)

    assert not default_out_dir.exists()
    assert {path.name for path in configured_out_dir.iterdir()} == {
        "ehull_relaxed.pkl.gz",
        "ehull_unrelaxed.pkl.gz",
        "relax_infos.pkl.gz",
        "relaxed.pkl.gz",
        "relaxed_niggli.pkl.gz",
        "smact_validity.npz",
    }


def test_real_test_config_uses_nested_output_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = PREPROCESS_PATH.parent.parent
    config_path = root / "configs" / "test" / "01_preprocess_real.yaml"
    monkeypatch.setattr(sys, "argv", [str(PREPROCESS_PATH), str(config_path)])

    args = preprocess.parse_args()

    assert args.gen_file == root / "input" / "gen" / "raw" / "test.pkl.gz"
    assert args.gen_out_dir == (
        root / "input" / "gen" / "preprocessed" / "test" / "real"
    )


def test_regenerated_relaxation_overwrites_dependent_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    relaxed_structure = structure.copy()
    relaxed_structure.scale_lattice(30.0)
    infos = [
        {
            "energy_per_atom_eV_initial": 1.0,
            "energy_per_atom_eV": 2.0,
        }
    ]

    src = tmp_path / "raw" / "model.pkl.gz"
    src.parent.mkdir()
    preprocess._save([structure], src)

    out_dir = tmp_path / "preprocessed" / "model"
    out_dir.mkdir(parents=True)
    preprocess._save(infos, out_dir / "relax_infos.pkl.gz")
    preprocess._save(["old"], out_dir / "ehull_unrelaxed.pkl.gz")
    preprocess._save(["old"], out_dir / "ehull_relaxed.pkl.gz")
    preprocess._save(["old"], out_dir / "relaxed_niggli.pkl.gz")
    np.savez_compressed(out_dir / "smact_validity.npz", valid=[True])

    monkeypatch.setattr(preprocess, "GEN_OUT_DIR", out_dir.parent)
    monkeypatch.setattr(
        preprocess,
        "relax_structures",
        lambda structures: ([relaxed_structure], infos),
    )
    monkeypatch.setattr(preprocess, "_get_ppd", object)
    monkeypatch.setattr(
        preprocess,
        "compute_ehulls",
        lambda structures, energies, ppd: list(energies),
    )
    monkeypatch.setattr(preprocess, "_reduce", lambda structure: structure)

    preprocess.preprocess_gen(src)

    assert preprocess._load(out_dir / "ehull_unrelaxed.pkl.gz") == [1.0]
    assert preprocess._load(out_dir / "ehull_relaxed.pkl.gz") == [2.0]
    reduced = preprocess._load(out_dir / "relaxed_niggli.pkl.gz")
    assert reduced == [relaxed_structure]
