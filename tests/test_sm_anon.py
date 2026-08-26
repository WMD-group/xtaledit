from __future__ import annotations

import math
import time
import warnings

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.structure_matcher import FrameworkComparator

import src.sm_anon as sm_anon
from src.sm_anon import AnonMatch, match_anonymous


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


def _dummy_match(idx1: int) -> AnonMatch:
    return AnonMatch(
        idx1=idx1,
        idx2=0,
        fu=1,
        s1_supercell=True,
        supercell_matrix=np.eye(3, dtype=int),
        mapping=np.array([0], dtype=np.int32),
        translation=np.zeros(3),
        rms=0.0,
        cost_uniform=0.0,
        cost_mod_petti=0.0,
    )


def _partial_then_sleep(args: tuple[int, Structure]):
    i, _s1 = args
    yield _dummy_match(i)
    if i == 0:
        time.sleep(5)


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


def test_match_anonymous_costs_self_match() -> None:
    small = _small_structure()
    matches = match_anonymous([small], [small])
    assert len(matches) == 1
    m = matches[0]
    assert not math.isnan(m.cost_uniform)
    assert not math.isnan(m.cost_mod_petti)
    assert not math.isnan(m.cost_cs)
    assert m.cost_uniform == 0.0
    assert m.cost_mod_petti == 0.0
    assert m.cost_cs == 0.0


def test_match_anonymous_costs_substituted_supercell() -> None:
    large = _large_structure_from_small()  # KBr supercell (2×NaCl), all atoms differ
    small = _small_structure()

    # large in structs1, small in structs2 (s1_supercell=False)
    matches = match_anonymous([large], [small])
    assert len(matches) == 1
    m = matches[0]
    assert m.cost_uniform == 1.0  # every atom is substituted
    assert m.cost_mod_petti > 0.0
    assert m.cost_cs == pytest.approx(0.322)

    # small in structs1, large in structs2 (s1_supercell=True)
    matches2 = match_anonymous([small], [large])
    assert len(matches2) == 1
    m2 = matches2[0]
    assert m2.cost_uniform == 1.0
    assert m2.cost_mod_petti > 0.0
    assert m2.cost_cs == pytest.approx(0.322)


def test_match_anonymous_rejects_non_divisible_site_counts() -> None:
    three_site = Structure(
        Lattice.cubic(4.0),
        ["Li", "O", "F"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
    )
    two_site = _small_structure()

    assert match_anonymous([three_site], [two_site]) == []


def test_match_anonymous_parallel_timeout_keeps_normal_matches() -> None:
    small = _small_structure()

    matches = match_anonymous([small], [small], n_jobs=2, timeout_sec=30.0)

    assert len(matches) == 1
    assert matches[0].idx1 == 0


def test_match_anonymous_timeout_keeps_partial_matches(monkeypatch) -> None:
    monkeypatch.setattr(sm_anon, "_iter_struct1_matches", _partial_then_sleep)
    small = _small_structure()

    matches = match_anonymous(
        [small, small],
        [small],
        n_jobs=2,
        timeout_sec=0.5,
    )

    assert sorted(m.idx1 for m in matches) == [0, 1]


def test_match_anonymous_timeout_requires_parallel_jobs() -> None:
    small = _small_structure()

    try:
        match_anonymous([small], [small], n_jobs=1, timeout_sec=1.0)
    except ValueError as exc:
        assert "n_jobs != 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


@pytest.mark.parametrize(
    ("matcher_kwarg", "warning_text"),
    [
        ({"primitive_cell": True}, "primitive_cell=True"),
        ({"attempt_supercell": False}, "attempt_supercell=False"),
        ({"allow_subset": True}, "allow_subset=True"),
        ({"supercell_size": "volume"}, "supercell_size='volume'"),
        ({"ignored_species": ["Na"]}, "ignored_species"),
        ({"comparator": object()}, "only supports FrameworkComparator"),
    ],
)
def test_match_anonymous_warns_and_normalizes_incompatible_matcher_options(
    matcher_kwarg: dict,
    warning_text: str,
) -> None:
    small = _small_structure()

    with pytest.warns(UserWarning, match=warning_text):
        matches = match_anonymous([small], [small], **matcher_kwarg)

    assert len(matches) == 1


def test_match_anonymous_accepts_supported_matcher_options_without_warning() -> None:
    small = _small_structure()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        matches = match_anonymous(
            [small],
            [small],
            primitive_cell=False,
            attempt_supercell=True,
            allow_subset=False,
            supercell_size="num_sites",
            ignored_species=(),
            comparator=FrameworkComparator(),
            scale=False,
        )

    assert len(matches) == 1


def test_match_anonymous_normalizes_disabled_supercell_matching() -> None:
    small = _small_structure()
    large = _large_structure_from_small()

    with pytest.warns(UserWarning, match="attempt_supercell=False"):
        matches = match_anonymous(
            [small],
            [large],
            attempt_supercell=False,
        )

    assert len(matches) == 1
    assert matches[0].fu == 2


def test_match_anonymous_does_not_copy_structures_for_anonymization(
    monkeypatch,
) -> None:
    small = _small_structure()
    seen: list[tuple[Structure, Structure]] = []
    original_match_pair = sm_anon._match_pair

    def recording_match_pair(matcher, s1, s2, *args):
        seen.append((s1, s2))
        return original_match_pair(matcher, s1, s2, *args)

    monkeypatch.setattr(sm_anon, "_match_pair", recording_match_pair)

    matches = match_anonymous([small], [small])

    assert len(matches) == 1
    assert len(seen) == 1
    assert seen[0][0] is small
    assert seen[0][1] is small
