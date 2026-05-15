"""Check relaxed substituted structures against their source generated structures.

Sources:
    --gen-path PATH                         -> list[Structure]
    substituted_relaxed_*.pkl.gz            -> list[SubstitutedEntry]

Outputs:
    substituted_relaxed_abc_gen_matches.pkl.gz -> list[dict]
    <output>.pkl.gz                            -> list[dict]
"""

from __future__ import annotations

import argparse
import ast
import gzip
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure

# Allow unpickling SubstitutedEntry objects saved from the sibling script module.
sys.path.insert(0, str(Path(__file__).parent))
from substitute_structures import SubstitutedEntry  # noqa: E402

from src.sm_fit import match_paired_structures


def _parse_kwarg(s: str) -> tuple[str, Any]:
    """Parse ``KEY=VALUE`` into ``(key, value)`` via ``ast.literal_eval``."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {s!r}")
    key, _, raw = s.partition("=")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    return key.strip(), value


def _load(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def _save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


def derive_output_path(input_path: Path, output_stem: str | None = None) -> Path:
    """Return the output path derived from a relaxed substituted entries path."""
    if output_stem is not None:
        return input_path.with_name(f"{output_stem}.pkl.gz")
    return input_path.with_name(
        f"{input_path.name.removesuffix('.pkl.gz')}_gen_matches.pkl.gz"
    )


def _validate_gen_indices(
    entries: list[SubstitutedEntry],
    gen: list[Structure],
) -> None:
    bad = [
        entry.gen_idx
        for entry in entries
        if entry.gen_idx < 0 or entry.gen_idx >= len(gen)
    ]
    if bad:
        raise SystemExit(
            "error: substituted entries reference generated structure indices outside "
            f"--gen-path range 0..{len(gen) - 1}; first invalid index is {bad[0]}"
        )


def _build_records(
    entries: list[SubstitutedEntry],
    matches: np.ndarray,
) -> list[dict[str, int | float | bool]]:
    return [
        {
            "entry_idx": i,
            "gen_idx": entry.gen_idx,
            "train_idx": entry.train_idx,
            "rank": entry.rank,
            "cost_uniform": entry.cost_uniform,
            "cost_mod_petti": entry.cost_mod_petti,
            "match": bool(match),
        }
        for i, (entry, match) in enumerate(zip(entries, matches, strict=True))
    ]


def process_file(
    path: Path,
    gen: list[Structure],
    args: argparse.Namespace,
    matcher_kwargs: dict[str, Any],
    fit_kwargs: dict[str, Any],
) -> None:
    """Check one relaxed substituted entries file and write match records."""
    out_path = derive_output_path(path, args.output)
    if out_path.exists() and not args.force:
        print(
            f"{path.name}: output exists at {out_path}, skipping (--force to overwrite)"
        )
        return

    print(f"Loading substituted entries from {path}")
    entries: list[SubstitutedEntry] = _load(path)
    print(f"  {len(entries):,} entries")
    _validate_gen_indices(entries, gen)

    relaxed = [entry.structure for entry in entries]
    generated = [gen[entry.gen_idx] for entry in entries]
    matches = match_paired_structures(
        relaxed,
        generated,
        fit_kwargs=fit_kwargs or None,
        **matcher_kwargs,
    )

    records = _build_records(entries, matches)
    _save(records, out_path)

    n_matched = int(matches.sum())
    n_unmatched = len(matches) - n_matched
    print(f"  matched: {n_matched:,}/{len(matches):,}")
    print(f"  unmatched: {n_unmatched:,}/{len(matches):,}")
    print(f"  output -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paths",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="One or more substituted_relaxed_*.pkl.gz files to check.",
    )
    p.add_argument(
        "--gen-path",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the generated structures pickle used to build substitutions.",
    )
    p.add_argument(
        "--sm",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="StructureMatcher constructor kwarg (repeatable).",
    )
    p.add_argument(
        "--fit",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="StructureMatcher.fit kwarg (repeatable).",
    )
    p.add_argument(
        "--output",
        metavar="STEM",
        help=(
            "Output filename stem (writes <input-dir>/<STEM>.pkl.gz). "
            "Only valid with one input path. If omitted, output is derived from "
            "the input filename."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is not None and len(args.paths) > 1:
        raise SystemExit("error: --output can only be used with one input path")
    if not args.gen_path.exists():
        raise SystemExit(f"error: generated structures not found: {args.gen_path}")
    for path in args.paths:
        if not path.exists():
            raise SystemExit(f"error: substituted entries file not found: {path}")

    print(f"Loading generated structures from {args.gen_path}")
    gen: list[Structure] = _load(args.gen_path)
    print(f"  {len(gen):,} generated structures")

    matcher_kwargs: dict[str, Any] = dict(_parse_kwarg(kv) for kv in args.sm)
    fit_kwargs: dict[str, Any] = dict(_parse_kwarg(kv) for kv in args.fit)

    for path in args.paths:
        process_file(path, gen, args, matcher_kwargs, fit_kwargs)


if __name__ == "__main__":
    main()
