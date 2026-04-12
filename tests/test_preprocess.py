from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
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
