"""Generate substituted structures by mapping gen elements onto training geometries.

For each AI-generated structure, finds the top-k most similar training structures
(lowest substitution cost) from anonymous or Wyckoff matching results, then replaces
every element in each training structure with the corresponding element from the
generated structure based on the stored atom mapping.

Sources:
    input/gen/preprocessed/<model>/relaxed_niggli.pkl.gz  -> list[Structure]
    input/train/preprocessed/train.pkl.gz                 -> list[Structure]
    --sm-anon-input                                        -> list[AnonMatch]
    --wyckoff-input                                        -> list[WyckoffMatch]

Outputs:
    --sm-anon-output  -> list[SubstitutedEntry]
    --wyckoff-output  -> list[SubstitutedEntry]
"""

from __future__ import annotations

import argparse
import gzip
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pymatgen.core import Structure

from src.config import INPUT_DIR
from src.sm_anon import AnonMatch
from src.wyckoff_match import WyckoffMatch

GEN_PRE_DIR = INPUT_DIR / "gen" / "preprocessed"
TRAIN_PATH = INPUT_DIR / "train" / "preprocessed" / "train.pkl.gz"


@dataclass
class SubstitutedEntry:
    """Training structure with elements replaced by those of a matched gen structure.

    Attributes:
        gen_idx: Index of the generated structure in the preprocessed gen list.
        train_idx: Index of the training structure in the preprocessed train list.
        rank: 1-based rank of this training match among all matches for this gen
            structure (ranked by the cost used for selection).
        cost_uniform: Uniform substitution cost of the match (0 = identical chemistry).
        cost_mod_petti: Modified Pettifor substitution cost of the match.
        structure: Training structure with element types replaced by those of the
            generated structure according to the atom mapping.
    """

    gen_idx: int
    train_idx: int
    rank: int
    cost_uniform: float
    cost_mod_petti: float
    structure: Structure


def _load(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def _save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


def _substitute_anon(
    m: AnonMatch,
    gen: list[Structure],
    train: list[Structure],
) -> Structure:
    """Return training structure copy with elements replaced by gen structure elements.

    Args:
        m: Anonymous match providing the atom-level correspondence.
        gen: List of generated structures (indexed by ``m.idx1``).
        train: List of training structures (indexed by ``m.idx2``).

    Returns:
        Copy of ``train[m.idx2]`` (possibly a supercell) with each atom's element
        replaced by the element of the corresponding atom in ``gen[m.idx1]``.
    """
    gen_elems = [site.species_string for site in gen[m.idx1]]
    if m.s1_supercell:
        # gen is the smaller structure (expanded by fu); train has fu * len(gen) atoms.
        # mapping[i]=j: train atom i <-> gen_supercell atom j; gen original = j // fu
        new = train[m.idx2].copy()
        for i, j in enumerate(m.mapping):
            new.replace(i, gen_elems[int(j) // m.fu])
    else:
        # train is smaller (expanded by fu); gen has fu * len(train) atoms.
        # mapping[i]=j: train_supercell atom i <-> gen atom j
        new = train[m.idx2].copy()
        new.make_supercell(m.supercell_matrix)
        for i, j in enumerate(m.mapping):
            new.replace(i, gen_elems[int(j)])
    return new


def _substitute_wyckoff(
    m: WyckoffMatch,
    gen: list[Structure],
    train: list[Structure],
) -> Structure:
    """Return training structure copy with elements replaced by gen structure elements.

    Args:
        m: Wyckoff match providing the atom-level correspondence.
        gen: List of generated structures (indexed by ``m.idx1``).
        train: List of training structures (indexed by ``m.idx2``).

    Returns:
        Copy of ``train[m.idx2]`` with each atom's element replaced by the element
        of the corresponding atom in ``gen[m.idx1]``.
    """
    gen_elems = [site.species_string for site in gen[m.idx1]]
    new = train[m.idx2].copy()
    # atom_map[j2]=j1: train atom j2 <-> gen atom j1
    for j2, j1 in enumerate(m.atom_map):
        new.replace(j2, gen_elems[int(j1)])
    return new


def _top_k_entries(
    matches: list[AnonMatch] | list[WyckoffMatch],
    k: int,
    cost_attr: Literal["cost_uniform", "cost_mod_petti"],
    gen: list[Structure],
    train: list[Structure],
    substitute_fn: Callable[..., Structure],
) -> list[SubstitutedEntry]:
    """Select top-k training matches per gen structure and build SubstitutedEntry list.

    Args:
        matches: Match objects from either anonymous or Wyckoff matching.
        k: Maximum number of training matches to keep per gen structure.
        cost_attr: Name of the cost field used for ranking (``"cost_uniform"`` or
            ``"cost_mod_petti"``).
        gen: List of generated structures.
        train: List of training structures.
        substitute_fn: Callable ``(match, gen, train) -> Structure`` that builds
            the substituted structure.

    Returns:
        List of :class:`SubstitutedEntry` objects, at most ``k`` per gen structure,
        ordered by gen index then rank.
    """
    cost: Callable[[AnonMatch | WyckoffMatch], float] = (
        (lambda m: m.cost_uniform)
        if cost_attr == "cost_uniform"
        else (lambda m: m.cost_mod_petti)
    )
    by_gen: dict[int, list[AnonMatch | WyckoffMatch]] = defaultdict(list)
    for m in matches:
        if not math.isnan(cost(m)):
            by_gen[m.idx1].append(m)

    results: list[SubstitutedEntry] = []
    for gen_idx in sorted(by_gen):
        ranked = sorted(by_gen[gen_idx], key=cost)[:k]
        for rank, m in enumerate(ranked, 1):
            results.append(
                SubstitutedEntry(
                    gen_idx=m.idx1,
                    train_idx=m.idx2,
                    rank=rank,
                    cost_uniform=m.cost_uniform,
                    cost_mod_petti=m.cost_mod_petti,
                    structure=substitute_fn(m, gen, train),
                )
            )
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        help="Model name (subdirectory of input/gen/preprocessed/).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of training matches to keep per generated structure (default: 5).",
    )
    p.add_argument(
        "--cost",
        choices=["uniform", "mod_petti"],
        default="uniform",
        help="Substitution cost used to rank and select matches (default: uniform).",
    )
    p.add_argument(
        "--sm-anon-input",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to the anonymous-match file (sm_anon.pkl.gz).",
    )
    p.add_argument(
        "--sm-anon-output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output path for substituted entries derived from anonymous matches.",
    )
    p.add_argument(
        "--wyckoff-input",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to the Wyckoff-match file (wyckoff_match_*.pkl.gz).",
    )
    p.add_argument(
        "--wyckoff-output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output path for substituted entries derived from Wyckoff matches.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    anon_in, anon_out = args.sm_anon_input, args.sm_anon_output
    wyck_in, wyck_out = args.wyckoff_input, args.wyckoff_output
    if (anon_in is None) != (anon_out is None):
        raise SystemExit(
            "error: --sm-anon-input and --sm-anon-output must be specified together"
        )
    if (wyck_in is None) != (wyck_out is None):
        raise SystemExit(
            "error: --wyckoff-input and --wyckoff-output must be specified together"
        )
    if anon_in is None and wyck_in is None:
        raise SystemExit(
            "error: at least one source pair (--sm-anon-input/output or "
            "--wyckoff-input/output) must be provided"
        )


def main() -> None:
    args = parse_args()
    _validate_args(args)

    cost_attr: Literal["cost_uniform", "cost_mod_petti"] = (
        "cost_uniform" if args.cost == "uniform" else "cost_mod_petti"
    )

    gen_path = GEN_PRE_DIR / args.model / "relaxed_niggli.pkl.gz"
    if not gen_path.exists():
        raise SystemExit(f"error: generated structures not found at {gen_path}")

    print(f"Loading generated structures from {gen_path}")
    gen: list[Structure] = _load(gen_path)
    print(f"  {len(gen):,} generated structures")

    print(f"Loading training structures from {TRAIN_PATH}")
    train: list[Structure] = _load(TRAIN_PATH)
    print(f"  {len(train):,} training structures")

    if args.sm_anon_input is not None:
        out = args.sm_anon_output
        assert out is not None
        if out.exists() and not args.force:
            print(f"sm_anon: output exists at {out}, skipping (--force to overwrite)")
        else:
            print(f"Loading anonymous matches from {args.sm_anon_input}")
            anon_matches: list[AnonMatch] = _load(args.sm_anon_input)
            print(f"  {len(anon_matches):,} matches")
            entries = _top_k_entries(
                anon_matches, args.top_k, cost_attr, gen, train, _substitute_anon
            )
            _save(entries, out)
            print(f"sm_anon: {len(entries):,} substituted entries -> {out}")

    if args.wyckoff_input is not None:
        out = args.wyckoff_output
        assert out is not None
        if out.exists() and not args.force:
            print(f"wyckoff: output exists at {out}, skipping (--force to overwrite)")
        else:
            print(f"Loading Wyckoff matches from {args.wyckoff_input}")
            wyck_matches: list[WyckoffMatch] = _load(args.wyckoff_input)
            print(f"  {len(wyck_matches):,} matches")
            entries = _top_k_entries(
                wyck_matches, args.top_k, cost_attr, gen, train, _substitute_wyckoff
            )
            _save(entries, out)
            print(f"wyckoff: {len(entries):,} substituted entries -> {out}")


if __name__ == "__main__":
    main()
