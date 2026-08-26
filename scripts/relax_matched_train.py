"""Relax training structures that directly match generated structures.

Sources:
    results/raw/<model>/sm_fit.npz              -> [gen_idx, train_idx] pairs
    input/train/preprocessed/train.pkl.gz       -> list[Structure]

Outputs in results/raw/<model>/:
    matched_train_indices.npz                   -> sorted unique train indices
    matched_train_relaxed.pkl.gz                -> raw relaxed structures
    matched_train_relaxed_niggli.pkl.gz         -> Niggli-reduced relaxed structures
    matched_train_relax_infos.pkl.gz            -> relaxation info dictionaries
    matched_train_ehull_unrelaxed.pkl.gz        -> initial e-above-hull values
    matched_train_ehull_relaxed.pkl.gz          -> relaxed e-above-hull values
"""

import argparse
import gzip
import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.core import Structure
from tqdm import tqdm

from src.config import INPUT_DIR, RAW_RESULTS_DIR
from src.energy import build_phase_diagram, compute_ehulls, relax_structures

TRAIN_PATH = INPUT_DIR / "train" / "preprocessed" / "train.pkl.gz"
PPD_CACHE = INPUT_DIR / "ppd_cache.pkl.gz"


def _load(path: Path) -> Any:
    with gzip.open(path, "rb") as file:
        return pickle.load(file)  # noqa: S301


def _save(obj: Any, path: Path) -> None:
    with gzip.open(path, "wb") as file:
        pickle.dump(obj, file)


def _get_ppd() -> PatchedPhaseDiagram:
    if PPD_CACHE.exists():
        return _load(PPD_CACHE)
    with MPRester(os.environ["MP_API_KEY"]) as mpr:
        return build_phase_diagram(mpr, cache_path=PPD_CACHE)


def _reduce(structure: Structure) -> Structure:
    return structure.get_primitive_structure().get_reduced_structure(
        reduction_algo="niggli"
    )


def process_model(args: argparse.Namespace) -> None:
    """Select and relax directly matched training structures."""
    out_dir = RAW_RESULTS_DIR / args.model
    matches_path = out_dir / "sm_fit.npz"
    if not matches_path.exists():
        raise SystemExit(f"error: match file not found: {matches_path}")

    with np.load(matches_path) as data:
        if "matches" not in data.files:
            raise SystemExit(f"error: {matches_path} does not contain 'matches'")
        matches = data["matches"]
    if matches.ndim != 2 or matches.shape[1] != 2:
        raise SystemExit(
            f"error: {matches_path} has shape {matches.shape}; expected (n, 2)"
        )
    if not np.issubdtype(matches.dtype, np.integer):
        raise SystemExit(f"error: {matches_path} matches must contain integers")

    train_indices = np.unique(matches[:, 1])
    indices_path = out_dir / "matched_train_indices.npz"
    relaxed_path = out_dir / "matched_train_relaxed.pkl.gz"
    niggli_path = out_dir / "matched_train_relaxed_niggli.pkl.gz"
    infos_path = out_dir / "matched_train_relax_infos.pkl.gz"
    ehull_unrelaxed_path = out_dir / "matched_train_ehull_unrelaxed.pkl.gz"
    ehull_relaxed_path = out_dir / "matched_train_ehull_relaxed.pkl.gz"
    dependent_paths = (
        relaxed_path,
        niggli_path,
        infos_path,
        ehull_unrelaxed_path,
        ehull_relaxed_path,
    )

    if indices_path.exists() and not args.force:
        with np.load(indices_path) as data:
            if "train_indices" not in data.files:
                raise SystemExit(
                    f"error: {indices_path} does not contain 'train_indices'"
                )
            saved_indices = data["train_indices"]
        if not np.array_equal(saved_indices, train_indices):
            raise SystemExit(
                f"error: matches changed since {indices_path} was written; "
                "rerun with --force"
            )
    elif any(path.exists() for path in dependent_paths) and not args.force:
        raise SystemExit(
            f"error: dependent outputs exist without {indices_path}; rerun with --force"
        )

    if (
        not args.force
        and indices_path.exists()
        and all(path.exists() for path in dependent_paths)
    ):
        print(f"{args.model}: matched training relaxation already done, skipping")
        return

    np.savez_compressed(indices_path, train_indices=train_indices)
    print(f"{args.model}: selected {len(train_indices):,} unique training structures")

    if not len(train_indices):
        for path in dependent_paths:
            _save([], path)
        print(f"{args.model}: no matching training structures")
        return

    if not TRAIN_PATH.exists():
        raise SystemExit(f"error: training structures not found: {TRAIN_PATH}")
    train: list[Structure] = _load(TRAIN_PATH)
    if train_indices[0] < 0 or train_indices[-1] >= len(train):
        raise SystemExit(
            f"error: training index outside range 0..{len(train) - 1} in {matches_path}"
        )
    selected = [train[int(index)] for index in train_indices]

    relaxation_regenerated = (
        args.force or not relaxed_path.exists() or not infos_path.exists()
    )
    if relaxation_regenerated:
        relaxed, infos = relax_structures(
            selected,
            device=args.device,
            fmax=args.fmax,
            max_steps=args.max_steps,
            forward_batch_size=args.forward_batch_size,
        )
        _save(relaxed, relaxed_path)
        _save(infos, infos_path)
        print(f"{args.model}: relaxed -> {relaxed_path}")
    else:
        relaxed = _load(relaxed_path)
        infos = _load(infos_path)
        if len(relaxed) != len(selected) or len(infos) != len(selected):
            raise SystemExit("error: saved relaxation length mismatch; use --force")
        print(f"{args.model}: relaxation already done, skipping")

    if (
        relaxation_regenerated
        or not ehull_unrelaxed_path.exists()
        or not ehull_relaxed_path.exists()
    ):
        ppd = _get_ppd()
        initial_energies = [
            info.get("energy_per_atom_eV_initial", math.nan)
            if info is not None
            else math.nan
            for info in infos
        ]
        relaxed_energies = [
            info.get("energy_per_atom_eV", math.nan) if info is not None else math.nan
            for info in infos
        ]
        if relaxation_regenerated or not ehull_unrelaxed_path.exists():
            _save(
                compute_ehulls(selected, initial_energies, ppd),
                ehull_unrelaxed_path,
            )
        if relaxation_regenerated or not ehull_relaxed_path.exists():
            _save(
                compute_ehulls(relaxed, relaxed_energies, ppd),
                ehull_relaxed_path,
            )
        print(f"{args.model}: e-above-hull calculated")
    else:
        print(f"{args.model}: e-above-hull already done, skipping")

    if relaxation_regenerated or not niggli_path.exists():
        reduced = [
            _reduce(structure)
            for structure in tqdm(relaxed, desc=f"{args.model} niggli")
        ]
        _save(reduced, niggli_path)
        print(f"{args.model}: Niggli-reduced -> {niggli_path}")
    else:
        print(f"{args.model}: Niggli reduction already done, skipping")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model name, for example mattergen.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fmax", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--forward-batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.fmax <= 0:
        parser.error("--fmax must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.forward_batch_size <= 0:
        parser.error("--forward-batch-size must be positive")
    return args


def main() -> None:
    process_model(parse_args())


if __name__ == "__main__":
    main()
