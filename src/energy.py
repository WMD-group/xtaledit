"""Energy utilities: MACE relaxation, e-above-hull, and phase diagram construction."""

from __future__ import annotations

import copy
import gzip
import math
import multiprocessing
import pickle
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from mace.calculators import mace_mp
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


def _relax_one(
    structure: Structure,
    base_calc: Any,
    fmax: float,
    max_steps: int,
) -> tuple[Structure, dict[str, Any] | None]:
    """Relax a single structure.

    Args:
        structure: Input pymatgen structure.
        base_calc: Shared ``MACECalculator`` whose model weights are reused.
        fmax: Force convergence threshold in eV/Å.
        max_steps: Maximum LBFGS steps.

    Returns:
        ``(relaxed_structure, info_dict)`` where ``info_dict`` may be a partial dict
        containing only the keys that were successfully computed, or ``None`` if even
        the initial energy evaluation failed.
    """
    calc = copy.copy(base_calc)
    calc.results = {}
    calc.atoms = None  # type: ignore[assignment]

    atoms: Any = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calc
    cell_filter = None
    optimizer = None

    try:
        n_atoms = len(atoms)
        info: dict[str, Any] = {}
        relaxed_structure_out = structure

        try:
            energy_initial = float(atoms.get_potential_energy())
            info["energy_per_atom_eV_initial"] = energy_initial / n_atoms
        except Exception:
            pass

        try:
            cell_filter = FrechetCellFilter(atoms)
            optimizer = LBFGS(cast(Any, cell_filter), logfile=None)  # type: ignore[arg-type]
            converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
            forces = atoms.get_forces()
            max_force = float(np.linalg.norm(forces, axis=1).max())
            info["max_force_eV_per_A"] = max_force
            info["converged"] = converged
            info["n_steps"] = optimizer.nsteps
        except Exception:
            pass

        try:
            energy = float(atoms.get_potential_energy())
            relaxed_structure_out = AseAtomsAdaptor.get_structure(atoms)
            info["energy_per_atom_eV"] = energy / n_atoms
        except Exception:
            pass

        return relaxed_structure_out, info or None
    finally:
        del atoms, cell_filter, optimizer


def _gpu_worker(
    args: tuple[list[Structure], str, float, int],
) -> list[tuple[Structure, dict[str, Any] | None]]:
    """Worker function run in a child process for one GPU.

    Args:
        args: Tuple of ``(structures, device, fmax, max_steps)``.

    Returns:
        List of ``(relaxed_structure, info_dict)`` pairs, one per input structure.
    """
    structures, device, fmax, max_steps = args
    model_path = str(Path.home() / ".cache" / "mace" / "macemp0mediummodel")
    base_calc = mace_mp(model=model_path, default_dtype="float64", device=device)
    results = [
        _relax_one(s, base_calc, fmax, max_steps)
        for s in tqdm(structures, desc=device, position=0, leave=True)
    ]
    del base_calc
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return results


def relax_structures(
    structures: list[Structure],
    device: str = "cuda",
    fmax: float = 1e-3,
    max_steps: int = 1000,
) -> tuple[list[Structure], list[dict[str, Any] | None]]:
    """Relax a list of structures with MACE-MP.

    When ``device`` is ``"cuda"`` and multiple GPUs are available, structures
    are distributed evenly across all GPUs, each running in its own child
    process.  Otherwise relaxation runs sequentially on the single device.

    Args:
        structures: Input pymatgen structures.
        device: PyTorch device string (e.g. ``"cuda"``, ``"cpu"``).
            When ``"cuda"`` is given and more than one GPU is detected,
            multi-GPU mode is used automatically.
        fmax: Force convergence threshold in eV/Å.
        max_steps: Maximum LBFGS steps per structure.

    Returns:
        Tuple of ``(relaxed_structures, infos)``. ``relaxed_structures``
        contains one relaxed ``Structure`` per input (original on failure).
        ``infos`` contains per-structure dicts with keys
        ``energy_per_atom_eV_initial``, ``energy_per_atom_eV``,
        ``max_force_eV_per_A``, ``converged``, ``n_steps``,
        or ``None`` on failure.
    """
    num_gpus = torch.cuda.device_count() if device == "cuda" else 0

    if num_gpus > 1:
        chunk_size = math.ceil(len(structures) / num_gpus)
        chunks = [
            structures[i : i + chunk_size]
            for i in range(0, len(structures), chunk_size)
        ]
        worker_args = [
            (chunk, f"cuda:{i}", fmax, max_steps) for i, chunk in enumerate(chunks)
        ]

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_gpus) as pool:
            chunk_results = pool.map(_gpu_worker, worker_args)

        relaxed_structures: list[Structure] = []
        infos: list[dict[str, Any] | None] = []
        for chunk_result in chunk_results:
            for rs, rs_info in chunk_result:
                relaxed_structures.append(rs)
                infos.append(rs_info)
        return relaxed_structures, infos

    # Sequential path: single GPU or CPU.
    base_calc = mace_mp(model="medium-mpa-0", default_dtype="float64", device=device)
    relaxed_structures = []
    infos = []
    for structure in tqdm(structures, desc="Relaxing"):
        rs, rs_info = _relax_one(structure, base_calc, fmax, max_steps)
        relaxed_structures.append(rs)
        infos.append(rs_info)

    del base_calc
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return relaxed_structures, infos


def compute_ehulls(
    structures: list[Structure],
    energies_per_atom: list[float],
    ppd: PatchedPhaseDiagram,
) -> list[float]:
    """Compute e-above-hull for each structure.

    Args:
        structures: Pymatgen structures (used for composition).
        energies_per_atom: DFT/ML total energy per atom in eV, one per structure.
        ppd: Prebuilt ``PatchedPhaseDiagram`` from the Materials Project.

    Returns:
        E-above-hull values in eV/atom. ``math.nan`` on failure.
    """

    results: list[float] = []
    for structure, e_per_atom in tqdm(
        zip(structures, energies_per_atom), total=len(structures), desc="E-above-hull"
    ):
        try:
            n_atoms = structure.num_sites
            entry = PDEntry(structure.composition, e_per_atom * n_atoms)
            result = ppd.get_e_above_hull(entry, allow_negative=True)
            if result is None:
                raise ValueError("composition outside phase diagram space")
            results.append(float(result))
        except Exception:
            results.append(math.nan)
    return results


def build_phase_diagram(
    mpr: MPRester,
    *,
    cache_path: Path | None = None,
) -> PatchedPhaseDiagram:
    """Fetch MP GGA/GGA+U entries and build a PatchedPhaseDiagram.

    The diagram is built from *uncorrected* DFT energies because MACE-MP is
    trained on raw VASP energies from the Materials Project without
    MP2020Compatibility corrections applied.  Using corrected reference energies
    would introduce a systematic offset when evaluating ML energies.

    ``MaterialsProject2020Compatibility.process_entries`` is called on
    GGA/GGA+U entries solely as a compatibility filter: entries that cannot be
    processed (e.g. mixed GGA/GGA+U systems) are silently excluded.  The
    resulting energy corrections are not used; ``uncorrected_energy`` is read
    from each surviving entry before constructing the diagram.

    Args:
        mpr: An initialised ``MPRester`` instance.
        cache_path: Optional path for a gzip-pickled ``PatchedPhaseDiagram``.
            If the file exists it is loaded directly (API call skipped).
            If the file does not exist the diagram is built and saved here.

    Returns:
        PatchedPhaseDiagram built from raw (uncorrected) GGA/GGA+U energies.
    """
    if cache_path is not None and cache_path.exists():
        with gzip.open(cache_path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301

    docs = mpr.thermo.search(fields=["entries"])
    all_entries: list[Any] = []
    for doc in docs:
        all_entries.extend(v for v in doc.entries.values() if v is not None)

    # Pre-filter to GGA/GGA+U before running process_entries to avoid paying
    # the correction cost for entries that would be discarded anyway.
    all_entries = [
        e for e in all_entries if e.parameters.get("run_type") in ("GGA", "GGA_U")
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Failed to guess oxidation states.*")
        all_entries = MaterialsProject2020Compatibility().process_entries(
            all_entries, clean=True
        )

    all_entries_uncorrected = [
        PDEntry(composition=e.composition, energy=e.uncorrected_energy)
        for e in all_entries
    ]
    ppd = PatchedPhaseDiagram(all_entries_uncorrected)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_path, "wb") as fh:
            pickle.dump(ppd, fh)

    return ppd
