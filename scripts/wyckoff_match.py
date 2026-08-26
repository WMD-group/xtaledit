"""Match AI-generated Niggli-reduced structures against training structures via Wyckoff.

For each model in input/gen/preprocessed/<model>_relaxed_niggli.pkl.gz, runs
Wyckoff-based matching against the Niggli-reduced training set and saves the
match results and precomputed orbit data as compressed pickle archives.

Sources:
    input/gen/preprocessed/<model>/relaxed_niggli.pkl.gz  -> list[Structure]
    input/train/preprocessed/train.pkl.gz                 -> list[Structure]

Outputs:
    results/raw/<model>/wyckoff_match_s={symprec}_c={cost}.pkl.gz
        -> list[WyckoffMatch]
    results/raw/<model>/wyckoff_repr_s={symprec}.pkl.gz
        -> list[WyckoffData]
    results/raw/train/wyckoff_repr_s={symprec}.pkl.gz
        -> list[WyckoffData]

Already-existing output files are skipped.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Any

from src.config import INPUT_DIR, RAW_RESULTS_DIR
from src.wyckoff_match import WyckoffData, match_wyckoff
from src.yaml_config import get_float, get_string, load_yaml_config

GEN_PRE_DIR = INPUT_DIR / "gen" / "preprocessed"
TRAIN_PATH = INPUT_DIR / "train" / "preprocessed" / "train.pkl.gz"


def _load(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def _save(obj: Any, path: Path) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


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
        allowed_keys={"model", "cost", "symprec"},
    )
    cost = get_string(config, "cost", default="uniform")
    if cost not in {"uniform", "mod_petti", "cs"}:
        raise SystemExit(
            "error: config key 'cost' must be 'uniform', 'mod_petti', or 'cs'"
        )
    symprec = get_float(config, "symprec", default=0.1)
    if symprec <= 0:
        raise SystemExit("error: config key 'symprec' must be positive")
    return argparse.Namespace(
        model=get_string(config, "model", default=None),
        cost=cost,
        symprec=symprec,
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

    data_train_path = (
        RAW_RESULTS_DIR / "train" / f"wyckoff_repr_s={args.symprec}.pkl.gz"
    )

    print(f"Loading training structures from {TRAIN_PATH}")
    train = _load(TRAIN_PATH)
    print(f"Loaded {len(train):,} training structures")

    # Load cached train repr if already computed for this symprec.
    train_data: list[WyckoffData] | None = None
    if data_train_path.exists():
        print(f"Loading cached train repr from {data_train_path}")
        train_data = _load(data_train_path)

    for gen_path in gen_files:
        stem = gen_path.parent.name
        out_dir = RAW_RESULTS_DIR / stem
        matches_path = out_dir / f"wyckoff_match_s={args.symprec}_c={args.cost}.pkl.gz"
        data_gen_path = out_dir / f"wyckoff_repr_s={args.symprec}.pkl.gz"

        if matches_path.exists() and data_gen_path.exists():
            print(f"{stem}: outputs already exist at {out_dir}, skipping")
            continue

        print(f"{stem}: loading {gen_path}")
        gen = _load(gen_path)
        n_gen, n_train = len(gen), len(train)
        print(f"{stem}: matching {n_gen:,} generated vs {n_train:,} training")

        matches, gen_data, train_data = match_wyckoff(
            gen,
            train_data if train_data is not None else train,
            cost=args.cost,
            symprec=args.symprec,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        _save(matches, matches_path)
        _save(gen_data, data_gen_path)
        if not data_train_path.exists():
            data_train_path.parent.mkdir(parents=True, exist_ok=True)
            _save(train_data, data_train_path)
        print(f"{stem}: {len(matches):,} matches -> {out_dir}")


if __name__ == "__main__":
    main()
