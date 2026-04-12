from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from src.sm_fit import match_paired_structures


def _nacl_structure() -> Structure:
    return Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


def test_match_paired_structures_matches_identical_structure() -> None:
    structure = _nacl_structure()

    matches = match_paired_structures([structure], [structure.copy()])

    assert matches.dtype == np.bool_
    assert matches.tolist() == [True]


def test_match_paired_structures_rejects_different_chemistry() -> None:
    a = _nacl_structure()
    b = a.copy()
    b.replace_species({"Na": "K"})

    matches = match_paired_structures([a], [b])

    assert matches.tolist() == [False]


def test_match_paired_structures_requires_equal_lengths() -> None:
    structure = _nacl_structure()

    with pytest.raises(ValueError, match="differ in length"):
        match_paired_structures([structure], [])
