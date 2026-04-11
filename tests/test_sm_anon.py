from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure

from src.sm_anon import match_anonymous


def _small_structure() -> Structure:
    return Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


def _large_structure_from_small() -> Structure:
    large = _small_structure().copy()
    large.make_supercell([2, 1, 1])
    large.replace_species({"Na": "K", "Cl": "Br"})
    return large


def test_match_anonymous_mapping_is_consistent_for_supercell_cases() -> None:
    large = _large_structure_from_small()
    small = _small_structure()

    m_large_small = match_anonymous([large], [small])[0]
    assert m_large_small.s1_supercell is False
    assert m_large_small.fu == 2
    assert len(m_large_small.mapping) == len(large)
    assert np.array_equal(np.sort(m_large_small.mapping), np.arange(len(large)))
    assert np.all((np.arange(len(large)) // m_large_small.fu) < len(small))

    m_small_large = match_anonymous([small], [large])[0]
    assert m_small_large.s1_supercell is True
    assert m_small_large.fu == 2
    assert len(m_small_large.mapping) == len(large)
    assert np.array_equal(np.sort(m_small_large.mapping), np.arange(len(large)))
    assert np.all((m_small_large.mapping // m_small_large.fu) < len(small))


def test_match_anonymous_rejects_non_divisible_site_counts() -> None:
    three_site = Structure(
        Lattice.cubic(4.0),
        ["Li", "O", "F"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
    )
    two_site = _small_structure()

    assert match_anonymous([three_site], [two_site]) == []
