"""StructureMatcher-based structure matching utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from tqdm import tqdm


def match_structures(
    a: list[Structure],
    b: list[Structure],
    fit_kwargs: dict[str, Any] | None = None,
    **matcher_kwargs: Any,
) -> np.ndarray:
    """Match structures in *a* against structures in *b* using StructureMatcher.

    Pre-filters pairs to the same chemical system before calling
    ``StructureMatcher.fit``.

    Args:
        a: Generated structures. Their indices in this list appear in column 0
            of the returned array.
        b: Reference (training) structures. Their indices appear in column 1.
        fit_kwargs: Forwarded to ``StructureMatcher.fit``.
        **matcher_kwargs: Forwarded to the ``StructureMatcher`` constructor.

    Returns:
        Integer array of shape ``(n_matches, 2)`` and dtype ``int32`` where
        each row ``[i, j]`` means ``a[i]`` matches ``b[j]``. An empty result
        has shape ``(0, 2)``.
    """
    # Group training structures by chemical system.
    train_by_sys: dict[frozenset[str], list[tuple[int, Structure]]] = defaultdict(list)
    for j, s in enumerate(b):
        key = frozenset(s.composition.chemical_system_set)
        train_by_sys[key].append((j, s))

    # Only enqueue indices of generated structures that share a chemsys with training.
    tasks: list[int] = [
        i
        for i, s in enumerate(a)
        if frozenset(s.composition.chemical_system_set) in train_by_sys
    ]

    matcher = StructureMatcher(**matcher_kwargs)
    pairs: list[tuple[int, int]] = []

    for a_idx in tqdm(tasks, desc="Matching", unit="struct"):
        a_struct = a[a_idx]
        sys_key = frozenset(a_struct.composition.chemical_system_set)
        train_pairs = train_by_sys.get(sys_key, [])
        pairs.extend(
            (a_idx, b_idx)
            for b_idx, b_struct in train_pairs
            if matcher.fit(a_struct, b_struct, **(fit_kwargs or {}))
        )

    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(pairs, dtype=np.int32)


def match_paired_structures(
    a: list[Structure],
    b: list[Structure],
    fit_kwargs: dict[str, Any] | None = None,
    **matcher_kwargs: Any,
) -> np.ndarray:
    """Match corresponding structures from two equally sized lists.

    Args:
        a: First list of structures.
        b: Second list of structures. ``b[i]`` is compared with ``a[i]``.
        fit_kwargs: Forwarded to ``StructureMatcher.fit``.
        **matcher_kwargs: Forwarded to the ``StructureMatcher`` constructor.

    Returns:
        Boolean array of shape ``(len(a),)`` where element ``i`` is true if
        ``a[i]`` matches ``b[i]``.

    Raises:
        ValueError: If ``a`` and ``b`` do not have the same length.
    """
    if len(a) != len(b):
        raise ValueError(
            f"paired structure lists differ in length: {len(a)} != {len(b)}"
        )

    matcher = StructureMatcher(**matcher_kwargs)
    return np.array(
        [
            matcher.fit(a_struct, b_struct, **(fit_kwargs or {}))
            for a_struct, b_struct in tqdm(
                zip(a, b, strict=True),
                total=len(a),
                desc="Matching pairs",
                unit="struct",
            )
        ],
        dtype=np.bool_,
    )
