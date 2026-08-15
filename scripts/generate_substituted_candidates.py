"""Generate random Wyckoff-orbit substitutions from MP20 training structures.

Each candidate starts from a uniformly sampled training structure. Every Wyckoff
orbit is selected independently with probability ``p``, conditioned on at least
one orbit being selected. Selected orbits receive replacement elements sampled
from the MP20 test-set modified-Pettifor cost distribution.

Output:
    input/gen/raw/subst_p={p}.pkl.gz -> list[Structure]
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import random
from pathlib import Path

from pymatgen.core import Structure

from src.config import INPUT_DIR
from src.mod_petti import MOD_PETTI
from src.wyckoff_match import precompute

TRAIN_PATH = INPUT_DIR / "train" / "preprocessed" / "train.pkl.gz"
GEN_RAW_DIR = INPUT_DIR / "gen" / "raw"
DEFAULT_N = 10_000
DEFAULT_P = 0.3148565564765805
DEFAULT_SEED = 42

# Changed-atom costs from selected MP20 test/train Wyckoff parent pairs.
COST_COUNTS = {
    1: 14630,
    2: 3574,
    3: 1526,
    4: 1106,
    5: 1346,
    6: 674,
    7: 533,
    8: 346,
    9: 385,
    10: 203,
    11: 252,
    12: 174,
    13: 255,
    14: 204,
    15: 244,
    16: 174,
    17: 200,
    18: 217,
    19: 116,
    20: 114,
    21: 103,
    22: 80,
    23: 73,
    24: 83,
    25: 89,
    26: 81,
    27: 59,
    28: 75,
    29: 70,
    30: 44,
    31: 46,
    32: 30,
    33: 27,
    34: 52,
    35: 54,
    36: 29,
    37: 30,
    38: 43,
    39: 48,
    40: 32,
    41: 32,
    42: 80,
    43: 51,
    44: 22,
    45: 14,
    46: 35,
    47: 36,
    48: 23,
    49: 11,
    50: 34,
    51: 16,
    52: 20,
    53: 16,
    54: 18,
    55: 22,
    56: 39,
    57: 27,
    58: 29,
    59: 43,
    60: 33,
    61: 30,
    62: 15,
    63: 19,
    64: 11,
    65: 45,
    66: 8,
    67: 14,
    68: 22,
    70: 14,
    71: 25,
    72: 13,
    73: 13,
    74: 4,
    75: 5,
    76: 7,
    77: 5,
    78: 7,
    79: 6,
    80: 7,
    81: 21,
    82: 10,
    83: 13,
    84: 1,
    85: 53,
    86: 18,
    87: 11,
    88: 1,
    89: 9,
    90: 5,
    91: 7,
    92: 12,
    93: 2,
}


def _load(path: Path) -> list[Structure]:
    with gzip.open(path, "rb") as file:
        return pickle.load(file)  # noqa: S301


def _save(structures: list[Structure], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as file:
        pickle.dump(structures, file)


def _replacement_candidates(
    training: list[Structure],
) -> dict[str, dict[int, list[str]]]:
    elements = sorted(
        {element.symbol for structure in training for element in structure.composition}
    )
    scale = {
        element: value for element, value in MOD_PETTI.items() if value is not None
    }
    missing = sorted(set(elements) - scale.keys())
    if missing:
        raise SystemExit(
            "error: MP20 elements missing modified Pettifor indices: "
            + ", ".join(missing)
        )

    candidates = {}
    for original in elements:
        by_cost: dict[int, list[str]] = {}
        for replacement in elements:
            cost = abs(scale[original] - scale[replacement])
            if replacement != original and cost in COST_COUNTS:
                by_cost.setdefault(cost, []).append(replacement)
        candidates[original] = by_cost

    unavailable = sorted(x for x, by_cost in candidates.items() if not by_cost)
    if unavailable:
        raise SystemExit(
            "error: empirical cost distribution gives no MP20 replacement for: "
            + ", ".join(unavailable)
        )
    return candidates


def _substitute(
    structure: Structure,
    candidates: dict[str, dict[int, list[str]]],
    p: float,
    symprec: float,
    rng: random.Random,
) -> tuple[Structure, float, int, int]:
    orbits = precompute(structure, symprec=symprec).orbit_atom_indices
    selected: list[list[int]] = []
    while not selected:
        selected = [orbit for orbit in orbits if rng.random() < p]

    result = structure.copy()
    changed_atom_cost = 0
    changed_atoms = 0
    for orbit in selected:
        original = structure[orbit[0]].specie.symbol
        by_cost = candidates[original]
        costs = sorted(by_cost)
        cost = rng.choices(costs, weights=[COST_COUNTS[value] for value in costs], k=1)[
            0
        ]
        replacement = rng.choice(by_cost[cost])
        changed_atom_cost += cost * len(orbit)
        changed_atoms += len(orbit)
        for atom_index in orbit:
            result.replace(atom_index, replacement)
    return result, len(selected) / len(orbits), changed_atom_cost, changed_atoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--N",
        type=int,
        default=DEFAULT_N,
        help=f"Number of candidate structures (default: {DEFAULT_N}).",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=DEFAULT_P,
        help=f"Per-orbit selection probability (default: {DEFAULT_P}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=0.01,
        help="Symmetry tolerance used to identify Wyckoff orbits (default: 0.01).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing output file."
    )
    args = parser.parse_args()

    if args.N <= 0:
        raise SystemExit("error: N must be positive")
    if not 0 < args.p <= 1:
        raise SystemExit("error: p must satisfy 0 < p <= 1")
    if args.symprec <= 0:
        raise SystemExit("error: --symprec must be positive")
    return args


def main() -> None:
    args = parse_args()
    output_path = GEN_RAW_DIR / f"subst_p={args.p}.pkl.gz"
    if output_path.exists() and not args.force:
        raise SystemExit(f"error: output exists: {output_path} (use --force)")
    if not TRAIN_PATH.exists():
        raise SystemExit(f"error: MP20 training structures not found: {TRAIN_PATH}")

    training = _load(TRAIN_PATH)
    if not training:
        raise SystemExit(f"error: no MP20 training structures found: {TRAIN_PATH}")

    candidates = _replacement_candidates(training)
    rng = random.Random(args.seed)
    generated = []
    orbit_fractions = []
    changed_atom_cost = 0
    changed_atoms = 0
    for _ in range(args.N):
        structure, orbit_fraction, atom_cost, atom_count = _substitute(
            rng.choice(training), candidates, args.p, args.symprec, rng
        )
        generated.append(structure)
        orbit_fractions.append(orbit_fraction)
        changed_atom_cost += atom_cost
        changed_atoms += atom_count

    _save(generated, output_path)
    print(f"Saved {len(generated):,} candidate structures to {output_path}")
    print(
        "Average fraction of orbits substituted: "
        f"{sum(orbit_fractions) / len(orbit_fractions):.6f}"
    )
    print(
        "Mean changed-atom modified-Pettifor cost: "
        f"{changed_atom_cost / changed_atoms:.6f}"
    )


if __name__ == "__main__":
    main()
