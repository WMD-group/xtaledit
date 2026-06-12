"""StructureMatcher-based structure matching utilities."""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.structure_matcher import SpeciesComparator, StructureMatcher
from tqdm import tqdm


def _strict_matcher_kwargs(matcher_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return matcher kwargs restricted to strict species-aware matching."""
    resolved = dict(matcher_kwargs)
    unsupported = {
        "ignored_species": ((), "an empty sequence"),
        "allow_subset": (False, "False"),
        "comparator": (None, "None"),
    }
    for key, (supported, description) in unsupported.items():
        requested = resolved.get(key, supported)
        if requested != supported:
            warnings.warn(
                f"match_structures does not support {key}={requested!r}; "
                f"using {description}",
                UserWarning,
                stacklevel=3,
            )
        resolved[key] = supported
    return resolved


def match_structures(
    a: list[Structure],
    b: list[Structure],
    fit_kwargs: dict[str, Any] | None = None,
    **matcher_kwargs: Any,
) -> np.ndarray:
    """Match structures in *a* against structures in *b* using StructureMatcher.

    Pre-filters pairs using the default species comparator's composition hash
    before calling ``StructureMatcher.fit``. ``ignored_species``,
    ``allow_subset=True``, and custom comparators are unsupported and replaced
    with strict species-aware defaults.

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
    matcher_kwargs = _strict_matcher_kwargs(matcher_kwargs)
    comparator = SpeciesComparator()

    # StructureMatcher.fit applies this same hash check before geometric matching.
    train_by_hash: dict[Any, list[tuple[int, Structure]]] = defaultdict(list)
    for j, s in enumerate(b):
        key = comparator.get_hash(s.composition)
        train_by_hash[key].append((j, s))

    # Only enqueue structures with a compatible fractional composition.
    tasks: list[int] = [
        i
        for i, s in enumerate(a)
        if comparator.get_hash(s.composition) in train_by_hash
    ]

    matcher = StructureMatcher(**matcher_kwargs)
    pairs: list[tuple[int, int]] = []

    for a_idx in tqdm(tasks, desc="Matching", unit="struct"):
        a_struct = a[a_idx]
        composition_hash = comparator.get_hash(a_struct.composition)
        train_pairs = train_by_hash.get(composition_hash, [])
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
