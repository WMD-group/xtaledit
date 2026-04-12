"""Relax substituted structures produced by substitute_structures.py.

Loads a list[SubstitutedEntry] from one or more substituted_*.pkl.gz files,
relaxes the .structure of each entry with MACE-MP, and writes the updated
entries alongside per-structure relaxation info dicts.

Sources:
    substituted_abc.pkl.gz   -> list[SubstitutedEntry]

Outputs (same directory as input):
    substituted_relaxed_niggli_abc.pkl.gz        -> list[SubstitutedEntry] (relaxed)
    substituted_relaxed_niggli_abc_infos.pkl.gz  -> list[dict | None]
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from pathlib import Path
from typing import Any

# Allow importing SubstitutedEntry from the sibling script.
sys.path.insert(0, str(Path(__file__).parent))
from substitute_structures import SubstitutedEntry  # noqa: E402

from src.energy import relax_structures


def _load(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def _save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


def derive_output_paths(input_path: Path) -> tuple[Path, Path]:
    """Return (entries_out, infos_out) derived from *input_path*.

    Args:
        input_path: Path to a ``substituted_*.pkl.gz`` file.

    Returns:
        Tuple of output paths in the same directory:

        - entries output: ``substituted_relaxed_niggli_<rest>.pkl.gz``
        - infos output:   ``substituted_relaxed_niggli_<rest>_infos.pkl.gz``

    Raises:
        SystemExit: If the filename does not start with ``substituted_``.
    """
    name = input_path.name
    prefix = "substituted_"
    if not name.startswith(prefix):
        raise SystemExit(
            f"error: input filename must start with '{prefix}', got: {name}"
        )
    rest = name[len(prefix) :]  # e.g. "abc.pkl.gz"
    entries_out = input_path.parent / f"substituted_relaxed_niggli_{rest}"
    # Strip .pkl.gz to build infos name, then re-add extension.
    base = f"substituted_relaxed_niggli_{rest}".removesuffix(".pkl.gz")
    infos_out = input_path.parent / f"{base}_infos.pkl.gz"
    return entries_out, infos_out


def process_file(path: Path, args: argparse.Namespace) -> None:
    """Relax substituted structures in *path* and write outputs.

    Args:
        path: Path to a ``substituted_*.pkl.gz`` file.
        args: Parsed CLI arguments providing device, fmax, max_steps, force.
    """
    entries_out, infos_out = derive_output_paths(path)

    if entries_out.exists() and infos_out.exists() and not args.force:
        print(f"{path.name}: outputs already exist, skipping (--force to overwrite)")
        return

    print(f"Loading {path}")
    entries: list[SubstitutedEntry] = _load(path)
    print(f"  {len(entries):,} entries")

    structures = [e.structure for e in entries]

    print(
        f"Relaxing on {args.device} (fmax={args.fmax}, max_steps={args.max_steps}) ..."
    )
    relaxed_structures, infos = relax_structures(
        structures,
        device=args.device,
        fmax=args.fmax,
        max_steps=args.max_steps,
    )

    for entry, relaxed in zip(entries, relaxed_structures):
        entry.structure = relaxed.get_primitive_structure().get_reduced_structure(
            reduction_algo="niggli"
        )

    _save(entries, entries_out)
    print(f"  entries -> {entries_out}")

    _save(infos, infos_out)
    print(f"  infos   -> {infos_out}")

    n_converged = sum(
        1 for info in infos if info is not None and info.get("converged", False)
    )
    n_failed = sum(1 for info in infos if info is None)
    print(f"  converged: {n_converged}/{len(infos)}, failed: {n_failed}/{len(infos)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paths",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="One or more substituted_*.pkl.gz files to relax.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device string (default: cuda).",
    )
    p.add_argument(
        "--fmax",
        type=float,
        default=1e-3,
        metavar="FMAX",
        help="Force convergence threshold in eV/Å (default: 1e-3).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        metavar="N",
        help="Maximum L-BFGS steps per structure (default: 1000).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.paths:
        if not path.exists():
            raise SystemExit(f"error: file not found: {path}")
        process_file(path, args)


if __name__ == "__main__":
    main()
