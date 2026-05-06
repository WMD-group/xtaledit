from __future__ import annotations

import importlib.util
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
