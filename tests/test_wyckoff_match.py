from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure

from src.mod_petti import MOD_PETTI
from src.wyckoff_match import WyckoffData, WyckoffMatch, match_wyckoff, precompute


def _nacl_conventional() -> Structure:
    """Rock-salt NaCl conventional cell (Fm-3m, spg 225)."""
    a = 5.64
    return Structure(
        Lattice.cubic(a),
        ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
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


def _simple_cubic() -> Structure:
    return Structure(Lattice.cubic(3.0), ["Cu"], [[0, 0, 0]])


def _kbr_conventional() -> Structure:
    """Rock-salt KBr conventional cell (Fm-3m, spg 225)."""
    a = 6.60
    return Structure(
        Lattice.cubic(a),
        ["K", "K", "K", "K", "Br", "Br", "Br", "Br"],
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


# ---------------------------------------------------------------------------
# precompute tests
# ---------------------------------------------------------------------------


def test_precompute_returns_wyckoff_data() -> None:
    data = precompute(_nacl_conventional())
    assert isinstance(data, WyckoffData)
    assert data.spg_num == 225
    # Two inequivalent orbits (Na and Cl).
    assert len(data.orbit_elements) == 2
    assert set(data.orbit_elements) == {"Na", "Cl"}


def test_precompute_nacl_has_two_distinct_representations() -> None:
    data = precompute(_nacl_conventional())
    # Euclidean normalizer of Fm-3m swaps Wyckoff a ↔ b → 2 distinct representations.
    num_distinct = max(data.repr_idx_for_relabeling) + 1
    assert num_distinct == 2


def test_precompute_result_type() -> None:
    data = precompute(_simple_cubic())
    assert isinstance(data.spg_num, int)
    assert isinstance(data.relabeled_letters, list)
    assert isinstance(data.orbit_elements, list)
    assert isinstance(data.orbit_atom_indices, list)


def test_precompute_coords() -> None:
    data = precompute(_nacl_conventional())
    assert isinstance(data.orbit_repr_coords, list)
    assert len(data.orbit_repr_coords) == len(data.orbit_elements)
    assert all(c.shape == (3,) for c in data.orbit_repr_coords)
    assert data.frac_coords.shape == (8, 3)


# ---------------------------------------------------------------------------
# match_wyckoff tests
# ---------------------------------------------------------------------------


def test_match_wyckoff_nacl_kbr_uniform_cost() -> None:
    nacl = _nacl_conventional()
    kbr = _kbr_conventional()
    matches, _, _ = match_wyckoff([nacl], [kbr], cost="uniform")
    assert len(matches) == 1
    m = matches[0]
    assert m.idx1 == 0 and m.idx2 == 0
    assert m.cost == 1.0  # two orbits substituted, each weighted 4/8
    assert isinstance(m.atom_map, np.ndarray)
    assert m.atom_map.shape == (len(kbr),)
    # atom_map must be a valid permutation of struct1 indices
    assert sorted(m.atom_map.tolist()) == list(range(len(nacl)))


def test_match_wyckoff_nacl_kbr_mod_petti_cost() -> None:
    nacl = _nacl_conventional()
    kbr = _kbr_conventional()
    matches, _, _ = match_wyckoff([nacl], [kbr], cost="mod_petti")
    assert len(matches) == 1
    m = matches[0]
    # Each orbit has 4/8 atoms; both orbits are substituted.
    expected = 0.5 * abs(MOD_PETTI["Na"] - MOD_PETTI["K"]) + 0.5 * abs(  # type: ignore[operator]
        MOD_PETTI["Cl"] - MOD_PETTI["Br"]  # type: ignore[operator]
    )
    assert m.cost == float(expected)


def test_match_wyckoff_no_match_different_sg() -> None:
    nacl = _nacl_conventional()
    sc = _simple_cubic()
    matches, _, _ = match_wyckoff([nacl], [sc])
    assert matches == []


def test_match_wyckoff_self_match_zero_cost() -> None:
    nacl = _nacl_conventional()
    matches, _, _ = match_wyckoff([nacl], [nacl], cost="uniform")
    assert len(matches) == 1
    m = matches[0]
    assert m.cost == 0.0
    # Self-match: each struct2 atom maps back to the same-index struct1 atom.
    assert np.array_equal(m.atom_map, np.arange(len(nacl), dtype=np.int64))


def test_match_wyckoff_atom_map_is_valid_permutation() -> None:
    nacl = _nacl_conventional()
    kbr = _kbr_conventional()
    matches, _, _ = match_wyckoff([nacl], [kbr])
    m = matches[0]
    n = len(nacl)
    assert m.atom_map.shape == (n,)
    assert sorted(m.atom_map.tolist()) == list(range(n))


def test_match_wyckoff_returns_wyckoff_data() -> None:
    nacl = _nacl_conventional()
    kbr = _kbr_conventional()
    _, data1, data2 = match_wyckoff([nacl], [kbr])
    assert len(data1) == 1 and isinstance(data1[0], WyckoffData)
    assert len(data2) == 1 and isinstance(data2[0], WyckoffData)
    assert data1[0].spg_num == 225
    assert data2[0].spg_num == 225


def test_match_wyckoff_accepts_precomputed_data() -> None:
    nacl = _nacl_conventional()
    kbr = _kbr_conventional()
    # Precompute once, then reuse.
    data_nacl = precompute(nacl)
    data_kbr = precompute(kbr)
    matches_raw, _, _ = match_wyckoff([nacl], [kbr])
    matches_pre, _, _ = match_wyckoff([data_nacl], [data_kbr])
    assert len(matches_pre) == len(matches_raw)
    m_raw = matches_raw[0]
    m_pre = matches_pre[0]
    assert m_pre.cost == m_raw.cost
    assert np.array_equal(m_pre.atom_map, m_raw.atom_map)
    assert isinstance(m_pre, WyckoffMatch)
