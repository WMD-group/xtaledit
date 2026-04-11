"""Match AI-generated Niggli-reduced structures against training structures.

For each model in input/gen/preprocessed/<model>_relaxed_niggli.pkl.gz, runs
StructureMatcher against the Niggli-reduced training set and saves the sparse
result as a compressed NumPy archive.

Sources:
    input/gen/preprocessed/<model>/relaxed_niggli.pkl.gz  -> list[Structure]
    input/train/preprocessed/train.pkl.gz                 -> list[Structure]

Outputs:
    results/<model>/<output>.npz  -> int32 array of shape (n_matches, 2)
                                             columns: [gen_idx, train_idx]

Already-existing output files are skipped.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.config import INPUT_DIR, RESULTS_DIR
from src.sm_fit import match_structures

GEN_PRE_DIR = INPUT_DIR / "gen" / "preprocessed"
TRAIN_PATH = INPUT_DIR / "train" / "preprocessed" / "train.pkl.gz"


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default=None,
        help="Model name (subdirectory of input/gen/preprocessed/). "
        "If omitted, all models are processed.",
    )
    p.add_argument(
        "--sm",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="StructureMatcher constructor kwarg (repeatable). "
        "E.g. --sm ltol=0.2 --sm primitive_cell=False",
    )
    p.add_argument(
        "--fit",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="StructureMatcher.fit kwarg (repeatable). "
        "E.g. --fit symmetric=True --fit skip_structure_reduction=False",
    )
    p.add_argument(
        "--output",
        default="sm_fit",
        help="Output filename stem (default: sm_fit → results/<model>/sm_fit.npz).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.model is not None:
        p = GEN_PRE_DIR / args.model / "relaxed_niggli.pkl.gz"
        if not p.exists():
            print(f"Error: no file found for model '{args.model}' at {p}")
            return
        gen_files = [p]
    else:
        gen_files = sorted(GEN_PRE_DIR.glob("*/relaxed_niggli.pkl.gz"))

    if not gen_files:
        print("No generated structure files found.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading training structures from {TRAIN_PATH}")
    train = _load(TRAIN_PATH)
    print(f"Loaded {len(train):,} training structures")

    matcher_kwargs: dict[str, Any] = dict(_parse_kwarg(kv) for kv in args.sm)
    fit_kwargs: dict[str, Any] = dict(_parse_kwarg(kv) for kv in args.fit)

    for gen_path in gen_files:
        stem = gen_path.parent.name
        out_path = RESULTS_DIR / stem / f"{args.output}.npz"

        if out_path.exists():
            print(f"{stem}: matches already exist at {out_path}, skipping")
            continue

        print(f"{stem}: loading {gen_path}")
        gen = _load(gen_path)
        n_gen, n_train = len(gen), len(train)
        print(f"{stem}: matching {n_gen:,} generated vs {n_train:,} training")

        matches = match_structures(
            gen,
            train,
            fit_kwargs=fit_kwargs or None,
            **matcher_kwargs,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, matches=matches)
        print(f"{stem}: {len(matches):,} matches -> {out_path}")


if __name__ == "__main__":
    main()
