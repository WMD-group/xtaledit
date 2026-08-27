"""Wyckoff-representation-based structure matching.

Two structures are **compatible** when they share the same space group number,
or belong to an enantiomorphic space-group pair after point inversion, and have
the same multiset of Wyckoff letters (regardless of which elements occupy each
letter).  The set of equivalent letter assignments for a structure is generated
by applying the coset representatives of the **Euclidean normalizer** of the
space group.  Pymatgen ships this data in
:data:`~pymatgen.analysis.prototypes.WYCKOFF_POSITION_RELAB_DICT` (from
``wyckoff-position-relabelings.json.gz``), which encodes each coset as a
:meth:`str.translate`-compatible mapping ``{ord(letter): new_letter}``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal

import numpy as np
from pymatgen.analysis.prototypes import WYCKOFF_POSITION_RELAB_DICT
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ._subst_cost import subst_cost_cs, subst_cost_mod_petti, subst_cost_uniform

_ENANTIOMORPHIC_SPACE_GROUP = {
    spg: partner
    for left, right in (
        (76, 78),
        (91, 95),
        (92, 96),
        (144, 145),
        (151, 153),
        (152, 154),
        (169, 170),
        (171, 172),
        (178, 179),
        (180, 181),
        (212, 213),
    )
    for spg, partner in ((left, right), (right, left))
}


@dataclass
class WyckoffData:
    """Precomputed Wyckoff orbit data for one structure.

    Call :func:`precompute` to construct this.  Pass instances directly to
    :func:`match_wyckoff` to avoid repeating the symmetry analysis.

    Attributes:
        spg_num: International space group number (1–230).
        relabeled_letters: ``relabeled_letters[k][i]`` is the Wyckoff letter of
            orbit *i* under Euclidean-normalizer coset *k*.
        letter_key: ``letter_key[k]`` is ``tuple(sorted(relabeled_letters[k]))``
            — used as a hash key for fast compatibility lookups.
        repr_idx_for_relabeling: ``repr_idx_for_relabeling[k]`` maps coset *k*
            to a deduplicated representation index (two cosets that produce the
            same ``(letter, element)`` multiset share the same index).
        orbit_elements: Element symbol for each orbit (invariant across cosets).
        orbit_atom_indices: Atom indices in the original structure belonging to
            each orbit (invariant across cosets).
        orbit_repr_coords: Lex-minimum fractional coordinate among the atoms of
            each orbit (shape ``(3,)`` per orbit, invariant across cosets).
            Used to disambiguate orbits that share the same Wyckoff letter.
        frac_coords: Fractional coordinates of all atoms, shape ``(N, 3)``.
            Used together with ``orbit_atom_indices`` for within-orbit atom
            matching.
    """

    spg_num: int
    relabeled_letters: list[list[str]]
    letter_key: list[tuple[str, ...]]
    repr_idx_for_relabeling: list[int]
    orbit_elements: list[str]
    orbit_atom_indices: list[list[int]]
    orbit_repr_coords: list[np.ndarray]
    frac_coords: np.ndarray


@dataclass
class WyckoffMatch:
    """Result of a Wyckoff-based structure match between one pair of structures.

    Two structures match when they share the same space group number, or belong
    to an enantiomorphic space-group pair after point inversion, and have the
    same multiset of Wyckoff letters (regardless of which elements occupy each
    letter).  Within each pair of normalizer representations, repeated
    occurrences of the same letter are paired by minimising the minimum-image
    fractional-coordinate distance between orbit representatives.  The
    requested element-substitution cost then selects the best pair of normalizer
    representations.

    Attributes:
        idx1: Index of the matched structure in the first input list.
        idx2: Index of the matched structure in the second input list.
        cost_uniform: Atom-fraction-weighted uniform substitution cost (0 if
            elements match, 1 if they differ), summed over all orbit pairs.
            Each orbit's contribution is scaled by
            ``orbit_size / total_atoms_in_struct1``.
        cost_mod_petti: Same weighting as ``cost_uniform`` but using the
            absolute difference of modified Pettifor numbers as the per-orbit
            cost.
        cost_cs: Same weighting as ``cost_uniform`` but using the absolute
            difference of chemical-scale values as the per-orbit cost.
        atom_map: Integer array of shape ``(N,)`` (``N`` = number of atoms,
            equal for both structures).  ``atom_map[j]`` is the index in
            ``structs1[idx1]`` of the atom that atom *j* in ``structs2[idx2]``
            corresponds to.  To substitute elements::

                new = structs2[m.idx2].copy()
                for j2, j1 in enumerate(m.atom_map):
                    new.replace(j2, structs1[m.idx1][j1].specie.symbol)

        repr1_idx: Deduplicated representation index for struct1 (indexes into
            the unique coset equivalence classes of ``structs1[idx1]``).
        repr2_idx: Deduplicated representation index for struct2.
    """

    idx1: int
    idx2: int
    cost_uniform: float
    cost_mod_petti: float
    atom_map: np.ndarray
    repr1_idx: int
    repr2_idx: int
    cost_cs: float = float("nan")

    def __setstate__(self, state: dict) -> None:
        state.setdefault("cost_cs", float("nan"))
        self.__dict__.update(state)


def precompute(struct: Structure, symprec: float = 0.1) -> WyckoffData:
    """Precompute Wyckoff orbit data for *struct*.

    Args:
        struct: Structure to analyse.
        symprec: Symmetry precision forwarded to
            :class:`~pymatgen.symmetry.analyzer.SpacegroupAnalyzer`.

    Returns:
        A :class:`WyckoffData` instance containing all relabeling variants and
        orbit-to-atom mappings needed by :func:`match_wyckoff`.
    """
    analyzer = SpacegroupAnalyzer(struct, symprec=symprec)
    spg_num = analyzer.get_space_group_number()
    symm = analyzer.get_symmetrized_structure()

    base_letters: list[str] = []
    orbit_elements: list[str] = []
    orbit_atom_indices: list[list[int]] = []
    orbit_repr_coords: list[np.ndarray] = []

    for wyck_sym, sites_group, atom_idxs in zip(
        symm.wyckoff_symbols, symm.equivalent_sites, symm.equivalent_indices
    ):
        letter = wyck_sym.lstrip("0123456789")
        orbit_elements.append(sites_group[0].specie.symbol)
        orbit_atom_indices.append(list(atom_idxs))
        base_letters.append(letter)
        repr_coord = min(
            (np.array(site.frac_coords) for site in sites_group), key=tuple
        )
        orbit_repr_coords.append(repr_coord)

    raw_relabelings = WYCKOFF_POSITION_RELAB_DICT.get(str(spg_num), [])

    relabeled_letters: list[list[str]] = []
    letter_key: list[tuple[str, ...]] = []
    repr_idx_for_relabeling: list[int] = []

    # Map canonical sites tuple → deduplicated representation index.
    seen: dict[tuple[tuple[str, str], ...], int] = {}

    if raw_relabelings:
        for trans in raw_relabelings:
            rletters = [lt.translate(trans) for lt in base_letters]
            relabeled_letters.append(rletters)
            letter_key.append(tuple(sorted(rletters)))
            sites = tuple(sorted(zip(rletters, orbit_elements)))
            if sites not in seen:
                seen[sites] = len(seen)
            repr_idx_for_relabeling.append(seen[sites])
    else:
        # Fallback: no relabeling data (identity only).
        relabeled_letters.append(list(base_letters))
        letter_key.append(tuple(sorted(base_letters)))
        sites = tuple(sorted(zip(base_letters, orbit_elements)))
        seen[sites] = 0
        repr_idx_for_relabeling.append(0)

    return WyckoffData(
        spg_num=spg_num,
        relabeled_letters=relabeled_letters,
        letter_key=letter_key,
        repr_idx_for_relabeling=repr_idx_for_relabeling,
        orbit_elements=orbit_elements,
        orbit_atom_indices=orbit_atom_indices,
        orbit_repr_coords=orbit_repr_coords,
        frac_coords=struct.frac_coords.copy(),
    )


def _min_img_frac_dist(c1: np.ndarray, c2: np.ndarray) -> float:
    """L2 norm of minimum-image fractional-coordinate difference."""
    d = c1 - c2
    d = d - np.round(d)
    return float(np.linalg.norm(d))


def _inverted_data(data: WyckoffData) -> WyckoffData:
    """Return an atom-order-preserving point-inverted view of *data*."""
    frac_coords = np.mod(-data.frac_coords, 1.0)
    orbit_repr_coords = [
        min((frac_coords[i] for i in indices), key=tuple)
        for indices in data.orbit_atom_indices
    ]
    return replace(
        data,
        spg_num=_ENANTIOMORPHIC_SPACE_GROUP[data.spg_num],
        orbit_repr_coords=orbit_repr_coords,
        frac_coords=frac_coords,
    )


def _orbit_cost(
    orbit_map: list[int],
    elems1: list[str],
    elems2: list[str],
    orbit_sizes1: list[int],
    cost_fn: Callable[[str, str], float],
) -> float:
    total = sum(orbit_sizes1)
    return sum(
        orbit_sizes1[k] / total * cost_fn(elems1[k], elems2[m])
        for k, m in enumerate(orbit_map)
    )


def _match_orbits(
    letters1: list[str],
    elems1: list[str],
    letters2: list[str],
    elems2: list[str],
    orbit_sizes1: list[int],
    cost_fn: Callable[[str, str], float],
    repr_coords1: list[np.ndarray],
    repr_coords2: list[np.ndarray],
) -> tuple[float, list[int]]:
    """Match orbits by Wyckoff letter and fractional-coordinate proximity.

    When a Wyckoff letter appears exactly once in each structure the match is
    trivial.  When it appears multiple times the assignment is decided by the
    Hungarian algorithm on minimum-image fractional-coordinate distances between
    orbit representatives.  Element-substitution costs do not affect that
    assignment; they are accumulated afterward from the resulting pairs.

    Args:
        letters1: Relabeled Wyckoff letter for each orbit in struct1.
        elems1: Element symbol for each orbit in struct1.
        letters2: Relabeled Wyckoff letter for each orbit in struct2.
        elems2: Element symbol for each orbit in struct2.
        orbit_sizes1: Number of atoms in each orbit of struct1.
        cost_fn: Pairwise element substitution cost function.
        repr_coords1: Representative fractional coordinate for each orbit in
            struct1 (lex-minimum atom within the orbit).
        repr_coords2: Representative fractional coordinate for each orbit in
            struct2.

    Returns:
        Tuple of ``(total_cost, orbit_mapping)`` where
        ``orbit_mapping[k]`` is the orbit index in struct2 matched to orbit
        ``k`` in struct1, and ``total_cost`` is the atom-fraction-weighted sum
        of substitution costs.
    """
    total_atoms1 = sum(orbit_sizes1)

    by_letter1: dict[str, list[int]] = {}
    for k, letter in enumerate(letters1):
        by_letter1.setdefault(letter, []).append(k)

    by_letter2: dict[str, list[int]] = {}
    for m, letter in enumerate(letters2):
        by_letter2.setdefault(letter, []).append(m)

    orbit_mapping = [0] * len(letters1)
    total_cost = 0.0

    for letter, idxs1 in by_letter1.items():
        idxs2 = by_letter2.get(letter, [])
        if not idxs2:
            continue
        if len(idxs1) == 1 and len(idxs2) == 1:
            orbit_mapping[idxs1[0]] = idxs2[0]
            weight = orbit_sizes1[idxs1[0]] / total_atoms1
            total_cost += weight * cost_fn(elems1[idxs1[0]], elems2[idxs2[0]])
        else:
            dist_matrix = np.array(
                [
                    [
                        _min_img_frac_dist(repr_coords1[k], repr_coords2[m])
                        for m in idxs2
                    ]
                    for k in idxs1
                ],
                dtype=float,
            )
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            for r, c in zip(row_ind, col_ind):
                orbit_mapping[idxs1[r]] = idxs2[c]
                weight = orbit_sizes1[idxs1[r]] / total_atoms1
                total_cost += weight * cost_fn(elems1[idxs1[r]], elems2[idxs2[c]])

    return total_cost, orbit_mapping


def match_wyckoff(
    structs1: list[Structure] | list[WyckoffData],
    structs2: list[Structure] | list[WyckoffData],
    cost: Literal["uniform", "mod_petti", "cs"] = "uniform",
    symprec: float = 0.1,
) -> tuple[list[WyckoffMatch], list[WyckoffData], list[WyckoffData]]:
    """Match every structure in *structs1* against compatible structures in *structs2*.

    Two structures are compatible when they share the same space group number,
    or belong to an enantiomorphic space-group pair after point-inverting the
    second structure, and have the same multiset of Wyckoff letters (regardless
    of element).  For each pair of Euclidean-normalizer representations,
    repeated occurrences of the same letter are assigned by minimum-image
    fractional-coordinate distance.  The requested element-substitution cost is
    calculated from that geometric assignment, and the lowest-cost
    representation pair is retained.  All Euclidean-normalizer choices (i.e.
    all origin settings) for both structures are considered.

    Either raw :class:`~pymatgen.core.Structure` objects or pre-computed
    :class:`WyckoffData` objects (from :func:`precompute`) may be supplied for
    either list.  Passing :class:`WyckoffData` avoids repeating the symmetry
    analysis when the same structures are matched multiple times.

    Args:
        structs1: First list of structures or pre-computed :class:`WyckoffData`.
            List indices appear as ``idx1`` in returned matches.
        structs2: Second list of structures or pre-computed :class:`WyckoffData`.
            List indices appear as ``idx2``.
        cost: Substitution cost policy.  ``"uniform"`` charges 1 per orbit
            where the elements differ; ``"mod_petti"`` charges the absolute
            difference of modified Pettifor numbers; ``"cs"`` charges the
            absolute difference of chemical-scale values.  The policy ranks
            normalizer-representation pairs but does not change the geometric
            assignment among repeated occurrences of one Wyckoff letter.
        symprec: Symmetry precision forwarded to
            :class:`~pymatgen.symmetry.analyzer.SpacegroupAnalyzer`.
            Ignored when pre-computed :class:`WyckoffData` is supplied.

    Returns:
        Tuple of ``(matches, data1, data2)`` where *matches* is a list of
        :class:`WyckoffMatch` objects (at most one per ordered pair, the one
        with the lowest cost across all representation combinations), and
        *data1* / *data2* are the :class:`WyckoffData` lists used internally
        (pre-computed or freshly computed).
    """
    cost_fn: Callable[[str, str], float] = {
        "uniform": subst_cost_uniform,
        "mod_petti": subst_cost_mod_petti,
        "cs": subst_cost_cs,
    }[cost]

    data1_list: list[WyckoffData] = (
        list(structs1)  # type: ignore[arg-type]
        if structs1 and isinstance(structs1[0], WyckoffData)
        else [precompute(s, symprec) for s in structs1]  # type: ignore[union-attr]
    )
    data2_list: list[WyckoffData] = (
        list(structs2)  # type: ignore[arg-type]
        if structs2 and isinstance(structs2[0], WyckoffData)
        else [precompute(s, symprec) for s in structs2]  # type: ignore[union-attr]
    )

    # Index: (spg, letter_key) -> struct2 index, data view, relabeling index.
    train_index: dict[
        tuple[int, tuple[str, ...]], list[tuple[int, WyckoffData, int]]
    ] = {}
    for j, data2 in enumerate(data2_list):
        variants = [data2]
        if data2.spg_num in _ENANTIOMORPHIC_SPACE_GROUP:
            variants.append(_inverted_data(data2))
        for variant in variants:
            for k2, lkey in enumerate(variant.letter_key):
                train_index.setdefault((variant.spg_num, lkey), []).append(
                    (j, variant, k2)
                )

    results: list[WyckoffMatch] = []

    print(
        f"Matching {len(data1_list)} structures against {len(data2_list)} candidates …"
    )
    for i, data1 in tqdm(
        enumerate(data1_list), total=len(data1_list), desc="Wyckoff match"
    ):
        # best[j] = (min_cost, repr1_idx, repr2_idx, orbit_map, struct2 data view)
        best: dict[int, tuple[float, int, int, list[int], WyckoffData]] = {}

        orbit_sizes1 = [len(idxs) for idxs in data1.orbit_atom_indices]

        seen_repr1: set[int] = set()
        for k1, lkey in enumerate(data1.letter_key):
            r1 = data1.repr_idx_for_relabeling[k1]
            if r1 in seen_repr1:
                continue
            seen_repr1.add(r1)
            idx_key = (data1.spg_num, lkey)
            for j, data2, k2 in train_index.get(idx_key, []):
                cost_val, orbit_map = _match_orbits(
                    data1.relabeled_letters[k1],
                    data1.orbit_elements,
                    data2.relabeled_letters[k2],
                    data2.orbit_elements,
                    orbit_sizes1,
                    cost_fn,
                    data1.orbit_repr_coords,
                    data2.orbit_repr_coords,
                )
                r2 = data2.repr_idx_for_relabeling[k2]
                if j not in best or cost_val < best[j][0]:
                    best[j] = (cost_val, r1, r2, orbit_map, data2)

        for j, (
            cost_val,
            repr1_idx,
            repr2_idx,
            orbit_map,
            data2,
        ) in best.items():
            # Build per-atom correspondence: atom_map[j2] = j1 where j2 indexes
            # struct2 atoms and j1 indexes the corresponding struct1 atom.
            n2 = sum(len(idxs) for idxs in data2.orbit_atom_indices)
            atom_map = np.full(n2, -1, dtype=np.int64)
            for orbit_k, orbit_m in enumerate(orbit_map):
                atoms1 = data1.orbit_atom_indices[orbit_k]
                atoms2 = data2.orbit_atom_indices[orbit_m]
                coords1 = data1.frac_coords[atoms1]
                coords2 = data2.frac_coords[atoms2]
                dist = np.array(
                    [
                        [
                            _min_img_frac_dist(coords1[a], coords2[b])
                            for b in range(len(atoms2))
                        ]
                        for a in range(len(atoms1))
                    ],
                    dtype=float,
                )
                ri, ci = linear_sum_assignment(dist)
                for r, c in zip(ri, ci):
                    atom_map[atoms2[c]] = atoms1[r]

            elems2 = data2.orbit_elements
            results.append(
                WyckoffMatch(
                    idx1=i,
                    idx2=j,
                    cost_uniform=_orbit_cost(
                        orbit_map,
                        data1.orbit_elements,
                        elems2,
                        orbit_sizes1,
                        subst_cost_uniform,
                    ),
                    cost_mod_petti=_orbit_cost(
                        orbit_map,
                        data1.orbit_elements,
                        elems2,
                        orbit_sizes1,
                        subst_cost_mod_petti,
                    ),
                    cost_cs=_orbit_cost(
                        orbit_map,
                        data1.orbit_elements,
                        elems2,
                        orbit_sizes1,
                        subst_cost_cs,
                    ),
                    atom_map=atom_map,
                    repr1_idx=repr1_idx,
                    repr2_idx=repr2_idx,
                )
            )

    return results, data1_list, data2_list
