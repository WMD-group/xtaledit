from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.structure_matcher import FrameworkComparator, StructureMatcher

from src.sm_fit import match_paired_structures, match_structures


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


def test_match_structures_matches_brute_force_default_matcher() -> None:
    nacl = _nacl_structure()
    kcl = nacl.copy()
    kcl.replace_species({"Na": "K"})
    references = [nacl.copy(), kcl.copy()]

    matches = match_structures([nacl, kcl], references)
    matcher = StructureMatcher()
    expected = np.array(
        [
            (i, j)
            for i, structure in enumerate([nacl, kcl])
            for j, reference in enumerate(references)
            if matcher.fit(structure, reference)
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(matches, expected)


@pytest.mark.parametrize(
    ("matcher_kwargs", "warning"),
    [
        ({"ignored_species": ["Cl"]}, "does not support ignored_species"),
        ({"allow_subset": True}, "does not support allow_subset"),
        (
            {"comparator": FrameworkComparator()},
            "does not support comparator",
        ),
    ],
)
def test_match_structures_warns_and_overrides_unsupported_options(
    matcher_kwargs: dict[str, object],
    warning: str,
) -> None:
    nacl = _nacl_structure()
    different = nacl.copy()
    different.replace_species({"Na": "K"})

    with pytest.warns(UserWarning, match=warning):
        matches = match_structures([nacl], [different], **matcher_kwargs)

    assert matches.shape == (0, 2)
    assert matches.dtype == np.int32


def test_match_structures_filters_different_stoichiometries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nacl = _nacl_structure()
    na2cl = Structure(
        Lattice.cubic(3.0),
        ["Na", "Na", "Cl"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
    )
    original_fit = StructureMatcher.fit
    calls = 0

    def counting_fit(
        self: StructureMatcher,
        struct1: Structure,
        struct2: Structure,
        *args: object,
        **kwargs: object,
    ) -> bool:
        nonlocal calls
        calls += 1
        return original_fit(self, struct1, struct2, *args, **kwargs)

    monkeypatch.setattr(StructureMatcher, "fit", counting_fit)

    matches = match_structures([nacl], [nacl.copy(), na2cl])

    assert matches.tolist() == [[0, 0]]
    assert calls == 1
