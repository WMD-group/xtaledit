"""Match AI-generated Niggli-reduced structures against training structures.

For each model in input/gen/preprocessed/<model>_relaxed_niggli.pkl.gz, runs
StructureMatcher against the Niggli-reduced training set and saves the sparse
result as a compressed NumPy archive.

Sources:
    input/gen/preprocessed/<model>/relaxed_niggli.pkl.gz  -> list[Structure]
    input/train/preprocessed/train.pkl.gz                 -> list[Structure]

Outputs:
    results/raw/<model>/<output>.npz  -> int32 array of shape (n_matches, 2)
                                                 columns: [gen_idx, train_idx]

Already-existing output files are skipped.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.config import INPUT_DIR, RAW_RESULTS_DIR
from src.sm_fit import match_structures
from src.yaml_config import get_mapping, get_string, load_yaml_config

GEN_PRE_DIR = INPUT_DIR / "gen" / "preprocessed"
TRAIN_PATH = INPUT_DIR / "train" / "preprocessed" / "train.pkl.gz"


def _load(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


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
        allowed_keys={"model", "structure_matcher", "fit", "output"},
    )
    return argparse.Namespace(
        model=get_string(config, "model", default=None),
        matcher_kwargs=get_mapping(config, "structure_matcher", default={}),
        fit_kwargs=get_mapping(config, "fit", default={}),
        output=get_string(config, "output", default="sm_fit"),
    )


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

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading training structures from {TRAIN_PATH}")
    train = _load(TRAIN_PATH)
    print(f"Loaded {len(train):,} training structures")

    for gen_path in gen_files:
        stem = gen_path.parent.name
        out_path = RAW_RESULTS_DIR / stem / f"{args.output}.npz"

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
            fit_kwargs=args.fit_kwargs or None,
            **args.matcher_kwargs,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, matches=matches)
        print(f"{stem}: {len(matches):,} matches -> {out_path}")


if __name__ == "__main__":
    main()
