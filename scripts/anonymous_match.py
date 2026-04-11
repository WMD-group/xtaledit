"""Match AI-generated Niggli-reduced structures against training structures anonymously.

For each model in input/gen/preprocessed/<model>_relaxed_niggli.pkl.gz, runs
fully anonymous (geometry-only) matching against the Niggli-reduced training set
and saves the results as a compressed pickle archive.

Sources:
    input/gen/preprocessed/<model>_relaxed_niggli.pkl.gz  -> list[Structure]
    input/train/preprocessed/train.pkl.gz                 -> list[Structure]

Outputs:
    results/<model>/<output>.pkl.gz  -> list[AnonMatch]

Already-existing output files are skipped.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import pickle
from pathlib import Path
from typing import Any

from src.config import INPUT_DIR, RESULTS_DIR
from src.sm_anon import match_anonymous

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


def _save(obj: Any, path: Path) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default=None,
        help="Model name (stem of <model>_relaxed_niggli.pkl.gz). "
        "If omitted, all models are processed.",
    )
    p.add_argument(
        "--sm",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="StructureMatcher constructor kwarg (repeatable). "
        "E.g. --sm ltol=0.2 --sm stol=0.3",
    )
    p.add_argument(
        "--output",
        default="sm_anon",
        help=(
            "Output filename stem (default: sm_anon → results/<model>/sm_anon.pkl.gz)."
        ),
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel worker processes (default: 1). "
            "Pass -1 to use max(1, cpu_count // 3) workers."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.model is not None:
        p = GEN_PRE_DIR / f"{args.model}_relaxed_niggli.pkl.gz"
        if not p.exists():
            print(f"Error: no file found for model '{args.model}' at {p}")
            return
        gen_files = [p]
    else:
        gen_files = sorted(GEN_PRE_DIR.glob("*_relaxed_niggli.pkl.gz"))

    if not gen_files:
        print("No generated structure files found.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading training structures from {TRAIN_PATH}")
    train = _load(TRAIN_PATH)
    print(f"Loaded {len(train):,} training structures")

    matcher_kwargs: dict[str, Any] = dict(_parse_kwarg(kv) for kv in args.sm)

    for gen_path in gen_files:
        stem = gen_path.name.removesuffix("_relaxed_niggli.pkl.gz")
        out_path = RESULTS_DIR / stem / f"{args.output}.pkl.gz"

        if out_path.exists():
            print(f"{stem}: matches already exist at {out_path}, skipping")
            continue

        print(f"{stem}: loading {gen_path}")
        gen = _load(gen_path)
        n_gen, n_train = len(gen), len(train)
        print(f"{stem}: matching {n_gen:,} generated vs {n_train:,} training")

        matches = match_anonymous(gen, train, n_jobs=args.jobs, **matcher_kwargs)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save(matches, out_path)
        print(f"{stem}: {len(matches):,} matches -> {out_path}")


if __name__ == "__main__":
    main()
