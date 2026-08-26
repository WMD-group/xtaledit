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
of distinct species*, which breaks full anonymisation. We bypass it entirely
and use ``FrameworkComparator`` with the internal ``_preprocess`` +
``_strict_match`` pipeline. This performs purely geometric matching without
copying every structure to replace its species.
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import sys
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.structure_matcher import FrameworkComparator, StructureMatcher
from tqdm import tqdm

from ._subst_cost import subst_cost_cs, subst_cost_mod_petti, subst_cost_uniform


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
    cost_cs: float = float("nan")

    def __setstate__(self, state: dict) -> None:
        state.setdefault("cost_uniform", float("nan"))
        state.setdefault("cost_mod_petti", float("nan"))
        state.setdefault("cost_cs", float("nan"))
        self.__dict__.update(state)


def _anonymous_matcher_kwargs(matcher_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return matcher kwargs normalized for full anonymous matching."""
    resolved = dict(matcher_kwargs)
    required = {
        "primitive_cell": (False, "False"),
        "attempt_supercell": (True, "True"),
        "allow_subset": (False, "False"),
        "supercell_size": ("num_sites", "'num_sites'"),
        "ignored_species": ((), "an empty sequence"),
    }
    for key, (supported, description) in required.items():
        requested = resolved.get(key, supported)
        incompatible = (
            bool(requested) if key == "ignored_species" else requested != supported
        )
        if incompatible:
            warnings.warn(
                f"match_anonymous does not support {key}={requested!r}; "
                f"using {description}",
                UserWarning,
                stacklevel=3,
            )
        resolved[key] = supported

    comparator = resolved.get("comparator")
    if comparator is not None and not isinstance(comparator, FrameworkComparator):
        warnings.warn(
            "match_anonymous only supports FrameworkComparator; "
            "using FrameworkComparator()",
            UserWarning,
            stacklevel=3,
        )
    resolved["comparator"] = FrameworkComparator()
    return resolved


def _compute_atom_costs(
    mapping: np.ndarray,
    fu: int,
    s1_supercell: bool,
    species1: list[str],
    species2: list[str],
) -> tuple[float, float, float]:
    """Compute per-atom-averaged substitution costs from an anonymous match mapping.

    Args:
        mapping: Integer array of length ``N_large`` as returned by
            :func:`_match_pair`.
        fu: Supercell factor stored in the :class:`AnonMatch`.
        s1_supercell: Whether struct1 was expanded into the supercell.
        species1: Element symbols for each atom of the original struct1.
        species2: Element symbols for each atom of the original struct2.

    Returns:
        Tuple ``(cost_uniform, cost_mod_petti, cost_cs)``, each averaged over
        ``N_large`` atom pairs.
    """
    n = len(mapping)
    uniform = 0.0
    mod_petti = 0.0
    cs = 0.0
    if s1_supercell:
        # mapping[i]=j: struct2 atom i ↔ struct1_supercell atom j → struct1 atom j // fu
        for i, j in enumerate(mapping):
            e1 = species1[int(j) // fu]
            e2 = species2[i]
            uniform += subst_cost_uniform(e1, e2)
            mod_petti += subst_cost_mod_petti(e1, e2)
            cs += subst_cost_cs(e1, e2)
    else:
        # mapping[i]=j: struct2_supercell atom i ↔ struct1 atom j; orig s2 = i // fu
        for i, j in enumerate(mapping):
            e1 = species1[int(j)]
            e2 = species2[i // fu]
            uniform += subst_cost_uniform(e1, e2)
            mod_petti += subst_cost_mod_petti(e1, e2)
            cs += subst_cost_cs(e1, e2)
    return uniform / n, mod_petti / n, cs / n


def _match_pair(
    matcher: StructureMatcher,
    s1: Structure,
    s2: Structure,
    idx1: int,
    idx2: int,
    species1: list[str],
    species2: list[str],
) -> AnonMatch | None:
    """Attempt an anonymous geometric match between one pair of structures.

    Args:
        matcher: Configured ``StructureMatcher`` using ``FrameworkComparator``.
        s1: First structure.
        s2: Second structure.
        idx1: Index of the first structure in the caller's list (stored in the
            returned :class:`AnonMatch` without modification).
        idx2: Index of the second structure in the caller's list.
        species1: Element symbols for each atom of the original struct1.
        species2: Element symbols for each atom of the original struct2.

    Returns:
        An :class:`AnonMatch` if a match was found, otherwise ``None``.
    """
    n1, n2 = len(s1), len(s2)
    if n1 >= n2:
        if n1 % n2 != 0:
            return None
    else:
        if n2 % n1 != 0:
            return None

    # _preprocess: (optionally) Niggli-reduce, determine fu / s1_supercell,
    # rescale lattice volumes.  skip_structure_reduction=True keeps atom order.
    s1_proc, s2_proc, fu, s1_supercell = matcher._preprocess(
        s1, s2, niggli=False, skip_structure_reduction=True
    )

    match = matcher._strict_match(
        s1_proc, s2_proc, fu, s1_supercell, use_rms=True, break_on_match=True
    )
    if match is None:
        return None

    rms, _dists, sc_m, translation, raw_mapping = match
    mapping = np.asarray(raw_mapping, dtype=np.int32)
    cost_u, cost_mp, cost_cs = _compute_atom_costs(
        mapping, fu, s1_supercell, species1, species2
    )
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
        cost_cs=cost_cs,
    )


# Module-level state populated by _init_worker in each spawned subprocess.
_g_s2_by_n: dict[int, list[tuple[int, Structure, list[str]]]] = {}
_g_all_s2_counts: set[int] = set()
_g_matcher: StructureMatcher | None = None
_g_compatible_ns_cache: dict[int, tuple[int, ...]] = {}

_ActiveProcess = tuple[multiprocessing.Process, float, int]


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


def _iter_struct1_matches(args: tuple[int, Structure]):
    """Yield matches for one struct1 against all compatible structs2."""
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
        return
    assert _g_matcher is not None, "_init_worker was not called"
    species1 = [site.specie.symbol for site in s1]
    for n2 in compatible_ns:
        for j, s2, species2 in _g_s2_by_n[n2]:
            m = _match_pair(_g_matcher, s1, s2, i, j, species1, species2)
            if m is not None:
                yield m


def _process_struct1(args: tuple[int, Structure]) -> list[AnonMatch]:
    """Worker function: match one struct1 against all compatible structs2."""
    return list(_iter_struct1_matches(args))


def _process_struct1_streaming(
    args: tuple[int, Structure],
    result_queue: multiprocessing.Queue,
    batch_size: int,
) -> None:
    """Child target that streams partial matches before completion."""
    i, _s1 = args
    try:
        batch: list[AnonMatch] = []
        for match in _iter_struct1_matches(args):
            batch.append(match)
            if len(batch) >= batch_size:
                result_queue.put(("matches", i, batch))
                batch = []
        if batch:
            result_queue.put(("matches", i, batch))
        result_queue.put(("done", i, None))
    except BaseException:  # noqa: BLE001
        result_queue.put(("error", i, traceback.format_exc()))


def _terminate_all(active: dict[int, _ActiveProcess]) -> None:
    """Terminate and join active child processes."""
    for process, _started, _n_sites in active.values():
        if process.is_alive():
            process.terminate()
        process.join()


def _match_anonymous_with_timeout(
    structs1: list[Structure],
    num_workers: int,
    timeout_sec: float,
    batch_size: int = 1,
) -> list[AnonMatch]:
    """Run anonymous matching with per-struct1 timeout and partial recovery."""
    if sys.platform == "win32":
        raise ValueError("timeout_sec is only supported on platforms with fork")
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")

    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    tasks = iter(enumerate(structs1))
    active: dict[int, _ActiveProcess] = {}
    partial_counts: dict[int, int] = defaultdict(int)
    results: list[AnonMatch] = []
    completed = 0

    def start_next() -> bool:
        try:
            args = next(tasks)
        except StopIteration:
            return False
        i, s1 = args
        process = ctx.Process(
            target=_process_struct1_streaming,
            args=(args, result_queue, batch_size),
        )
        process.start()
        active[i] = (process, time.monotonic(), len(s1))
        return True

    try:
        with tqdm(total=len(structs1), desc="Anon matching", unit="struct") as pbar:
            while len(active) < num_workers and start_next():
                pass

            while completed < len(structs1):
                got_message = False
                while True:
                    try:
                        kind, idx, payload = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    got_message = True
                    if kind == "matches":
                        assert isinstance(payload, list)
                        results.extend(payload)
                        partial_counts[idx] += len(payload)
                    elif kind == "done":
                        process_info = active.pop(idx, None)
                        if process_info is not None:
                            process, _started, _n_sites = process_info
                            process.join()
                            completed += 1
                            pbar.update(1)
                            while len(active) < num_workers and start_next():
                                pass
                    elif kind == "error":
                        _terminate_all(active)
                        raise RuntimeError(
                            f"anonymous match worker failed for idx1={idx}:\n{payload}"
                        )
                    else:
                        _terminate_all(active)
                        raise RuntimeError(f"unknown worker message kind: {kind!r}")

                now = time.monotonic()
                timed_out: list[int] = []
                for idx, (process, started, n_sites) in active.items():
                    if now - started > timeout_sec:
                        process.terminate()
                        process.join()
                        timed_out.append(idx)
                        print(
                            "Warning: anonymous match timed out for "
                            f"idx1={idx} n_sites={n_sites} after {timeout_sec:g}s; "
                            f"keeping {partial_counts[idx]} partial matches",
                            file=sys.stderr,
                        )

                for idx in timed_out:
                    active.pop(idx, None)
                    completed += 1
                    pbar.update(1)
                    while len(active) < num_workers and start_next():
                        pass

                if not got_message and not timed_out:
                    time.sleep(0.1)
    finally:
        _terminate_all(active)
        result_queue.close()
        result_queue.join_thread()

    return results


def match_anonymous(
    structs1: list[Structure],
    structs2: list[Structure],
    n_jobs: int = 1,
    timeout_sec: float | None = None,
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
        timeout_sec: Optional per-``structs1`` timeout in seconds for parallel
            matching. Timed-out structures keep matches emitted before timeout.
        **matcher_kwargs: Forwarded to the ``StructureMatcher`` constructor.
            Full anonymous matching requires ``primitive_cell=False``,
            ``attempt_supercell=True``, ``allow_subset=False``,
            ``supercell_size="num_sites"``, no ignored species, and
            ``FrameworkComparator``. Incompatible values are replaced with
            these settings and emit ``UserWarning``.

    Returns:
        List of :class:`AnonMatch` objects, one per matched pair.  Order
        is deterministic only when ``n_jobs=1``; with ``n_jobs != 1`` the
        order depends on task completion order.
    """
    matcher_kwargs = _anonymous_matcher_kwargs(matcher_kwargs)

    # Group structs2 by atom count for O(1) pre-filtering.
    s2_by_n: dict[int, list[tuple[int, Structure, list[str]]]] = defaultdict(list)
    for j, s in enumerate(structs2):
        s2_by_n[len(s)].append((j, s, [site.specie.symbol for site in s]))

    results: list[AnonMatch] = []

    if n_jobs == 1:
        if timeout_sec is not None:
            raise ValueError("timeout_sec is only supported when n_jobs != 1")
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

            species1 = [site.specie.symbol for site in s1]
            for n2 in compatible_ns:
                for j, s2, species2 in s2_by_n[n2]:
                    m = _match_pair(matcher, s1, s2, i, j, species1, species2)
                    if m is not None:
                        results.append(m)
    else:
        num_workers = max(1, (os.cpu_count() or 1) // 3) if n_jobs == -1 else n_jobs
        use_fork = sys.platform != "win32"
        if use_fork:
            global _g_s2_by_n, _g_all_s2_counts, _g_matcher, _g_compatible_ns_cache
            _g_s2_by_n = dict(s2_by_n)
            _g_all_s2_counts = set(s2_by_n)
            _g_matcher = StructureMatcher(**matcher_kwargs)
            _g_compatible_ns_cache = {}
        if timeout_sec is not None:
            if not use_fork:
                raise ValueError("timeout_sec is only supported on platforms with fork")
            try:
                results.extend(
                    _match_anonymous_with_timeout(structs1, num_workers, timeout_sec)
                )
            finally:
                _g_s2_by_n = {}
                _g_all_s2_counts = set()
                _g_matcher = None
                _g_compatible_ns_cache = {}
            return results
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
            _g_matcher = None
            _g_compatible_ns_cache = {}

    return results
