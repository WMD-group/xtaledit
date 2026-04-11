"""Anonymous structure matching with full atom-level mapping.

Unlike ``fit_anonymous``, this module:

* performs **complete** anonymisation — A matches B as long as ``len(A)`` is
  a multiple of ``len(B)`` (or vice versa), regardless of chemistry;
* returns the atom-level mapping for every matched pair;
* records the supercell transformation used when one structure is a supercell
  of the other.

Implementation note
-------------------
``StructureMatcher._anonymous_match`` rejects pairs that differ in the *number
of distinct species*, which breaks full anonymisation.  We bypass it entirely:
all atoms in both structures are replaced with the dummy element ``"He"`` before
the internal ``_preprocess`` + ``_strict_match`` pipeline is called.  With a
single species the species-count check trivially passes and there is exactly one
permutation (He → He), so the result is a purely geometric match.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.structure_matcher import StructureMatcher
from tqdm import tqdm

from ._subst_cost import subst_cost_mod_petti, subst_cost_uniform


@dataclass
class AnonMatch:
    """Result of an anonymous structure match between one pair of structures.

    Attributes:
        idx1: Index of the matched structure in the first input list.
        idx2: Index of the matched structure in the second input list.
        fu: Supercell factor (integer ≥ 1).  The smaller structure is expanded
            by this factor to produce a cell commensurate with the larger one.
        s1_supercell: If ``True``, *struct1* is the smaller structure that was
            expanded into a supercell; if ``False``, *struct2* was expanded.
        supercell_matrix: 3 × 3 integer array — the transformation matrix
            applied to the lattice of the **smaller** structure to construct the
            supercell used in the match.
        mapping: 1-D integer array of length ``N_large`` (the number of atoms
            in the larger or supercell-expanded side).

            * When ``s1_supercell=True`` (struct1 was expanded):
              ``mapping[i] = j`` means struct2 atom ``i`` corresponds to atom
              ``j`` in the supercell of struct1.  The original struct1 atom is
              ``j // fu``; the lattice-point replica index is ``j % fu``.

            * When ``s1_supercell=False`` (struct2 was expanded):
              ``mapping[i] = j`` means atom ``i`` in the supercell of struct2
              corresponds to struct1 atom ``j``.  The original struct2 atom is
              ``i // fu``.

        translation: Fractional translation vector (shape ``(3,)``) applied to
            align the structures after the supercell construction.
        rms: Normalised RMS distance of the match (same units as
            ``StructureMatcher.stol``).
    """

    idx1: int
    idx2: int
    fu: int
    s1_supercell: bool
    supercell_matrix: np.ndarray  # (3, 3) int
    mapping: np.ndarray  # (N_large,) int32
    translation: np.ndarray  # (3,) float
    rms: float
    cost_uniform: float = float("nan")
    cost_mod_petti: float = float("nan")

    def __setstate__(self, state: dict) -> None:
        state.setdefault("cost_uniform", float("nan"))
        state.setdefault("cost_mod_petti", float("nan"))
        self.__dict__.update(state)


def _anonymize(struct: Structure) -> Structure:
    """Return a copy of *struct* with every species replaced by ``"He"``."""
    anon = struct.copy()
    anon.replace_species({sp: "He" for sp in struct.composition})
    return anon


def _compute_atom_costs(
    mapping: np.ndarray,
    fu: int,
    s1_supercell: bool,
    species1: list[str],
    species2: list[str],
) -> tuple[float, float]:
    """Compute per-atom-averaged substitution costs from an anonymous match mapping.

    Args:
        mapping: Integer array of length ``N_large`` as returned by
            :func:`_match_pair`.
        fu: Supercell factor stored in the :class:`AnonMatch`.
        s1_supercell: Whether struct1 was expanded into the supercell.
        species1: Element symbols for each atom of the original struct1.
        species2: Element symbols for each atom of the original struct2.

    Returns:
        Tuple ``(cost_uniform, cost_mod_petti)``, each averaged over ``N_large``
        atom pairs.
    """
    n = len(mapping)
    uniform = 0.0
    mod_petti = 0.0
    if s1_supercell:
        # mapping[i]=j: struct2 atom i ↔ struct1_supercell atom j → struct1 atom j // fu
        for i, j in enumerate(mapping):
            e1 = species1[int(j) // fu]
            e2 = species2[i]
            uniform += subst_cost_uniform(e1, e2)
            mod_petti += subst_cost_mod_petti(e1, e2)
    else:
        # mapping[i]=j: struct2_supercell atom i ↔ struct1 atom j; orig s2 = i // fu
        for i, j in enumerate(mapping):
            e1 = species1[int(j)]
            e2 = species2[i // fu]
            uniform += subst_cost_uniform(e1, e2)
            mod_petti += subst_cost_mod_petti(e1, e2)
    return uniform / n, mod_petti / n


def _match_pair(
    matcher: StructureMatcher,
    s1_anon: Structure,
    s2_anon: Structure,
    idx1: int,
    idx2: int,
    species1: list[str],
    species2: list[str],
) -> AnonMatch | None:
    """Attempt an anonymous geometric match between one pair of structures.

    Args:
        matcher: Configured ``StructureMatcher`` instance.  Both structures
            must have already been anonymised (all species set to ``"He"``).
        s1_anon: Anonymised version of the first structure.
        s2_anon: Anonymised version of the second structure.
        idx1: Index of the first structure in the caller's list (stored in the
            returned :class:`AnonMatch` without modification).
        idx2: Index of the second structure in the caller's list.
        species1: Element symbols for each atom of the original struct1.
        species2: Element symbols for each atom of the original struct2.

    Returns:
        An :class:`AnonMatch` if a match was found, otherwise ``None``.
    """
    n1, n2 = len(s1_anon), len(s2_anon)
    if n1 >= n2:
        if n1 % n2 != 0:
            return None
    else:
        if n2 % n1 != 0:
            return None

    # _preprocess: (optionally) Niggli-reduce, determine fu / s1_supercell,
    # rescale lattice volumes.  skip_structure_reduction=True keeps atom order.
    s1_proc, s2_proc, fu, s1_supercell = matcher._preprocess(
        s1_anon, s2_anon, niggli=False, skip_structure_reduction=True
    )

    match = matcher._strict_match(
        s1_proc, s2_proc, fu, s1_supercell, use_rms=True, break_on_match=True
    )
    if match is None:
        return None

    rms, _dists, sc_m, translation, raw_mapping = match
    mapping = np.asarray(raw_mapping, dtype=np.int32)
    cost_u, cost_mp = _compute_atom_costs(mapping, fu, s1_supercell, species1, species2)
    return AnonMatch(
        idx1=idx1,
        idx2=idx2,
        fu=fu,
        s1_supercell=s1_supercell,
        supercell_matrix=np.asarray(sc_m, dtype=int),
        mapping=mapping,
        translation=np.asarray(translation, dtype=float),
        rms=float(rms),
        cost_uniform=cost_u,
        cost_mod_petti=cost_mp,
    )


# Module-level state populated by _init_worker in each spawned subprocess.
_g_s2_by_n: dict[int, list[tuple[int, Structure, list[str]]]] = {}
_g_all_s2_counts: set[int] = set()
_g_matcher: StructureMatcher | None = None
_g_compatible_ns_cache: dict[int, tuple[int, ...]] = {}


def _init_worker(
    s2_by_n: dict[int, list[tuple[int, Structure, list[str]]]],
    matcher_kwargs: dict[str, Any],
) -> None:
    """Pool initializer: populate per-worker globals once per subprocess."""
    global _g_s2_by_n, _g_all_s2_counts, _g_matcher, _g_compatible_ns_cache
    _g_s2_by_n = s2_by_n
    _g_all_s2_counts = set(s2_by_n)
    _g_matcher = StructureMatcher(**matcher_kwargs)
    _g_compatible_ns_cache = {}


def _init_worker_fork(matcher_kwargs: dict[str, Any]) -> None:
    """Pool initializer for fork context: s2 data already inherited via fork."""
    global _g_matcher, _g_compatible_ns_cache
    _g_matcher = StructureMatcher(**matcher_kwargs)
    _g_compatible_ns_cache = {}


def _process_struct1(args: tuple[int, Structure]) -> list[AnonMatch]:
    """Worker function: match one struct1 against all compatible structs2."""
    i, s1 = args
    n1 = len(s1)
    compatible_ns = _g_compatible_ns_cache.get(n1)
    if compatible_ns is None:
        compatible_ns = tuple(
            n2
            for n2 in _g_all_s2_counts
            if (n1 >= n2 and n1 % n2 == 0) or (n2 > n1 and n2 % n1 == 0)
        )
        _g_compatible_ns_cache[n1] = compatible_ns
    if not compatible_ns:
        return []
    assert _g_matcher is not None, "_init_worker was not called"
    s1_anon = _anonymize(s1)
    species1 = [site.specie.symbol for site in s1]
    matches: list[AnonMatch] = []
    for n2 in compatible_ns:
        for j, s2_anon, species2 in _g_s2_by_n[n2]:
            m = _match_pair(_g_matcher, s1_anon, s2_anon, i, j, species1, species2)
            if m is not None:
                matches.append(m)
    return matches


def match_anonymous(
    structs1: list[Structure],
    structs2: list[Structure],
    n_jobs: int = 1,
    **matcher_kwargs,
) -> list[AnonMatch]:
    """Match every structure in *structs1* against every compatible structure in
    *structs2* using purely geometric (fully anonymous) matching.

    Two structures are *compatible* if the atom count of one is an integer
    multiple of the atom count of the other.  No chemical species information
    is used.

    Args:
        structs1: First list of structures (e.g. AI-generated).  Their
            list indices appear as ``idx1`` in the returned matches.
        structs2: Second list of structures (e.g. training set).  Their
            list indices appear as ``idx2``.
        n_jobs: Number of worker processes.  ``1`` (default) runs
            sequentially in the main process.  ``-1`` uses
            ``max(1, os.cpu_count() // 3)`` workers.
        **matcher_kwargs: Forwarded to the ``StructureMatcher`` constructor.

    Returns:
        List of :class:`AnonMatch` objects, one per matched pair.  Order
        is deterministic only when ``n_jobs=1``; with ``n_jobs != 1`` the
        order depends on task completion order.
    """
    # Enable supercell matching by default; callers can override with
    # attempt_supercell=False.
    matcher_kwargs.setdefault("attempt_supercell", True)

    # Anonymise structs2 once and group by atom count for O(1) pre-filtering.
    s2_by_n: dict[int, list[tuple[int, Structure, list[str]]]] = defaultdict(list)
    for j, s in enumerate(structs2):
        s2_by_n[len(s)].append((j, _anonymize(s), [site.specie.symbol for site in s]))

    results: list[AnonMatch] = []

    if n_jobs == 1:
        matcher = StructureMatcher(**matcher_kwargs)
        all_s2_counts: set[int] = set(s2_by_n)
        compatible_ns_cache: dict[int, tuple[int, ...]] = {}

        for i, s1 in enumerate(tqdm(structs1, desc="Anon matching", unit="struct")):
            n1 = len(s1)
            compatible_ns = compatible_ns_cache.get(n1)
            if compatible_ns is None:
                compatible_ns = tuple(
                    n2
                    for n2 in all_s2_counts
                    if (n1 >= n2 and n1 % n2 == 0) or (n2 > n1 and n2 % n1 == 0)
                )
                compatible_ns_cache[n1] = compatible_ns
            if not compatible_ns:
                continue

            s1_anon = _anonymize(s1)
            species1 = [site.specie.symbol for site in s1]
            for n2 in compatible_ns:
                for j, s2_anon, species2 in s2_by_n[n2]:
                    m = _match_pair(matcher, s1_anon, s2_anon, i, j, species1, species2)
                    if m is not None:
                        results.append(m)
    else:
        num_workers = max(1, (os.cpu_count() or 1) // 3) if n_jobs == -1 else n_jobs
        use_fork = sys.platform != "win32"
        if use_fork:
            global _g_s2_by_n, _g_all_s2_counts
            _g_s2_by_n = dict(s2_by_n)
            _g_all_s2_counts = set(s2_by_n)
        ctx = multiprocessing.get_context("fork" if use_fork else "spawn")
        init_fn = _init_worker_fork if use_fork else _init_worker
        init_args: tuple = (
            (matcher_kwargs,) if use_fork else (dict(s2_by_n), matcher_kwargs)
        )
        with ctx.Pool(
            processes=num_workers,
            initializer=init_fn,
            initargs=init_args,
        ) as pool:
            task_iter = pool.imap_unordered(
                _process_struct1,
                enumerate(structs1),
                chunksize=1,
            )
            for batch in tqdm(
                task_iter,
                total=len(structs1),
                desc="Anon matching",
                unit="struct",
            ):
                results.extend(batch)
        if use_fork:
            _g_s2_by_n = {}
            _g_all_s2_counts = set()

    return results
