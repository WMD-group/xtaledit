"""Generate substituted structures by mapping gen elements onto training geometries.

For each AI-generated structure, finds the top-k most similar training structures
(lowest substitution cost) from anonymous or Wyckoff matching results, then replaces
every element in each training structure with the corresponding element from the
generated structure based on the stored atom mapping.

Sources:
    input/gen/preprocessed/<model>/relaxed_niggli.pkl.gz  -> list[Structure]
    input/train/preprocessed/train.pkl.gz                 -> list[Structure]
    sm_anon_input                                          -> list[AnonMatch]
    wyckoff_input                                          -> list[WyckoffMatch]

Outputs:
    sm_anon_output  -> list[SubstitutedEntry]
    wyckoff_output  -> list[SubstitutedEntry]
"""

from __future__ import annotations

import argparse
import gzip
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal

from pymatgen.core import Structure

from src import substituted_entry
from src.config import INPUT_DIR
from src.sm_anon import AnonMatch
from src.wyckoff_match import WyckoffMatch
from src.yaml_config import (
    get_bool,
    get_int,
    get_path,
    get_string,
    load_yaml_config,
)

GEN_PRE_DIR = INPUT_DIR / "gen" / "preprocessed"
TRAIN_PATH = INPUT_DIR / "train" / "preprocessed" / "train.pkl.gz"


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
) -> list[substituted_entry.SubstitutedEntry]:
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

    results: list[substituted_entry.SubstitutedEntry] = []
    for gen_idx in sorted(by_gen):
        ranked = sorted(by_gen[gen_idx], key=cost)[:k]
        for rank, m in enumerate(ranked, 1):
            results.append(
                substituted_entry.SubstitutedEntry(
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
        "config",
        type=Path,
        metavar="CONFIG.yaml",
        help="YAML configuration file.",
    )
    cli_args = p.parse_args()
    config = load_yaml_config(
        cli_args.config,
        allowed_keys={
            "model",
            "top_k",
            "cost",
            "sm_anon_input",
            "sm_anon_output",
            "wyckoff_input",
            "wyckoff_output",
            "force",
        },
        required_keys={"model"},
    )
    top_k = get_int(config, "top_k", default=5)
    if top_k <= 0:
        raise SystemExit("error: config key 'top_k' must be positive")
    cost = get_string(config, "cost", default="uniform")
    if cost not in {"uniform", "mod_petti"}:
        raise SystemExit("error: config key 'cost' must be 'uniform' or 'mod_petti'")
    return argparse.Namespace(
        model=get_string(config, "model"),
        top_k=top_k,
        cost=cost,
        sm_anon_input=get_path(config, "sm_anon_input", default=None),
        sm_anon_output=get_path(config, "sm_anon_output", default=None),
        wyckoff_input=get_path(config, "wyckoff_input", default=None),
        wyckoff_output=get_path(config, "wyckoff_output", default=None),
        force=get_bool(config, "force", default=False),
    )


def _validate_args(args: argparse.Namespace) -> None:
    anon_in, anon_out = args.sm_anon_input, args.sm_anon_output
    wyck_in, wyck_out = args.wyckoff_input, args.wyckoff_output
    if (anon_in is None) != (anon_out is None):
        raise SystemExit(
            "error: config keys 'sm_anon_input' and 'sm_anon_output' "
            "must be specified together"
        )
    if (wyck_in is None) != (wyck_out is None):
        raise SystemExit(
            "error: config keys 'wyckoff_input' and 'wyckoff_output' "
            "must be specified together"
        )
    if anon_in is None and wyck_in is None:
        raise SystemExit(
            "error: at least one input/output pair for anonymous or Wyckoff "
            "matches must be configured"
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
            print(f"sm_anon: output exists at {out}, skipping (set force: true)")
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
            print(f"wyckoff: output exists at {out}, skipping (set force: true)")
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
