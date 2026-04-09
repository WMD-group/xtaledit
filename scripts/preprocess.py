"""Preprocess crystal structures: relax, compute e-above-hull, and Niggli-reduce.

For each AI-generated model in input/gen/raw/<model>.pkl.gz, this script:
  1. Relaxes structures with MACE-MP and saves Niggli-reduced relaxed structures.
  2. Saves per-structure relaxation infos (energies, convergence).
  3. Computes e-above-hull for unrelaxed (MACE initial energy) and relaxed structures.
  4. Niggli-reduces the raw unrelaxed structures.

For the training set in input/train/raw/train.csv, Niggli reduction is applied.

Sources:
    input/gen/raw/*.pkl.gz              -> list[Structure] (pickle), one file per model
    input/train/raw/train.csv           -> CSV with "cif" column

Outputs:
    input/gen/preprocessed/<model>.pkl.gz                   -> Niggli-reduced unrelaxed
    input/gen/preprocessed/<model>_relaxed.pkl.gz           -> Niggli-reduced relaxed
    input/gen/preprocessed/<model>_relax_infos.pkl.gz        -> relaxation infos
    input/gen/preprocessed/<model>_ehull_unrelaxed.pkl.gz   -> list[float] e-above-hull
    input/gen/preprocessed/<model>_ehull_relaxed.pkl.gz     -> list[float] e-above-hull
    input/train/preprocessed/train.pkl.gz                   -> list[Structure] (pickle)

Already-existing output files are skipped.
"""

import gzip
import math
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.core import Structure
from tqdm import tqdm

from src.energy import build_phase_diagram, compute_ehulls, relax_structures

load_dotenv()
MP_API_KEY: str = os.environ["MP_API_KEY"]

ROOT = Path(__file__).resolve().parent.parent
GEN_RAW_DIR = ROOT / "input" / "gen" / "raw"
GEN_OUT_DIR = ROOT / "input" / "gen" / "preprocessed"
TRAIN_RAW = ROOT / "input" / "train" / "raw" / "train.csv"
TRAIN_OUT = ROOT / "input" / "train" / "preprocessed" / "train.pkl.gz"
PPD_CACHE = ROOT / "input" / "ppd_cache.pkl.gz"


def _reduce(s: Structure) -> Structure:
    return s.get_primitive_structure().get_reduced_structure(reduction_algo="niggli")


def _save(obj: Any, path: Path) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


def _load(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def _stem(path: Path) -> str:
    return path.name.removesuffix(".pkl.gz")


def _get_ppd() -> PatchedPhaseDiagram:
    with MPRester(MP_API_KEY) as mpr:
        return build_phase_diagram(mpr, cache_path=PPD_CACHE)


def preprocess_gen() -> None:
    ppd: PatchedPhaseDiagram | None = None

    for src in sorted(GEN_RAW_DIR.glob("*.pkl.gz")):
        stem = _stem(src)

        dst_niggli = GEN_OUT_DIR / src.name
        dst_relaxed = GEN_OUT_DIR / f"{stem}_relaxed.pkl.gz"
        dst_infos = GEN_OUT_DIR / f"{stem}_relax_infos.pkl.gz"
        dst_ehull_u = GEN_OUT_DIR / f"{stem}_ehull_unrelaxed.pkl.gz"
        dst_ehull_r = GEN_OUT_DIR / f"{stem}_ehull_relaxed.pkl.gz"

        structures: list[Structure] = _load(src)

        # Step 1: Relax
        relaxed: list[Structure] | None = None
        infos: list[dict[str, Any] | None] | None = None

        if dst_relaxed.exists() and dst_infos.exists():
            print(f"{stem}: relaxation already done, skipping")
        else:
            print(f"{stem}: relaxing {len(structures)} structures")
            relaxed, infos = relax_structures(structures)
            niggli_relaxed = [
                _reduce(s) for s in tqdm(relaxed, desc=f"{stem} niggli (relaxed)")
            ]
            _save(niggli_relaxed, dst_relaxed)
            _save(infos, dst_infos)
            print(f"{stem}: relaxed -> {dst_relaxed}")

        # Step 2: E-above-hull
        if not dst_ehull_u.exists() or not dst_ehull_r.exists():
            if relaxed is None:
                print(f"{stem}: loading relaxed/infos for e-hull")
                relaxed = _load(dst_relaxed)
                infos = _load(dst_infos)
            assert relaxed is not None and infos is not None

            if ppd is None:
                print("Loading phase diagram (this may take a while on first run)...")
                ppd = _get_ppd()

            e_initial = [
                i.get("energy_per_atom_eV_initial", math.nan)
                if i is not None
                else math.nan
                for i in infos
            ]
            e_relaxed = [
                i.get("energy_per_atom_eV", math.nan) if i is not None else math.nan
                for i in infos
            ]

            if not dst_ehull_u.exists():
                _save(compute_ehulls(structures, e_initial, ppd), dst_ehull_u)
                print(f"{stem}: ehull (unrelaxed) -> {dst_ehull_u}")
            else:
                print(f"{stem}: ehull (unrelaxed) already done, skipping")

            if not dst_ehull_r.exists():
                _save(compute_ehulls(relaxed, e_relaxed, ppd), dst_ehull_r)
                print(f"{stem}: ehull (relaxed) -> {dst_ehull_r}")
            else:
                print(f"{stem}: ehull (relaxed) already done, skipping")
        else:
            print(f"{stem}: ehull already done, skipping")

        # Step 3: Niggli-reduce unrelaxed
        if dst_niggli.exists():
            print(f"{stem}: niggli (unrelaxed) already done, skipping")
        else:
            reduced = [
                _reduce(s) for s in tqdm(structures, desc=f"{stem} niggli (unrelaxed)")
            ]
            _save(reduced, dst_niggli)
            print(f"{stem}: niggli (unrelaxed) -> {dst_niggli}")


def preprocess_train() -> None:
    if TRAIN_OUT.exists():
        print("train: already preprocessed, skipping")
        return

    df = pd.read_csv(TRAIN_RAW)

    reduced = [
        _reduce(Structure.from_str(cif, fmt="cif"))
        for cif in tqdm(df["cif"], desc="train")
    ]

    _save(reduced, TRAIN_OUT)
    print(f"train: {len(reduced)} structures -> {TRAIN_OUT}")


if __name__ == "__main__":
    preprocess_gen()
    preprocess_train()
