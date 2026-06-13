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
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from ase.stress import full_3x3_to_voigt_6_stress
from mace.calculators import mace_mp
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

MACE_MODEL = "medium-mpa-0"


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


def _compute_batch_results(
    atoms_list: list[Any],
    base_calc: Any,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Run MACE forward passes over multiple ASE Atoms objects.

    Replicates ``MACECalculator._atoms_to_batch`` for *N* structures
    simultaneously.  The returned dicts carry the same unit-converted quantities
    that ``MACECalculator.calculate`` would set on ``self.results``.

    When ``len(atoms_list) > batch_size``, the list is split into chunks of at
    most ``batch_size`` structures and processed sequentially to avoid GPU OOM.

    Args:
        atoms_list: ASE ``Atoms`` objects at their current positions/cells.
        base_calc: ``MACECalculator`` whose model weights, z-table, and unit
            conversions are reused.
        batch_size: Maximum number of structures per single MACE forward pass.

    Returns:
        One dict per structure with keys:

        - ``energy`` (float, eV)
        - ``forces`` (ndarray [N_atoms, 3], eV/Å)
        - ``stress`` (ndarray [6], eV/Å³ Voigt) — only present when the
          model type supports stress and compilation is not active.
    """
    if len(atoms_list) > batch_size:
        results: list[dict[str, Any]] = []
        for i in range(0, len(atoms_list), batch_size):
            results.extend(
                _compute_batch_results(
                    atoms_list[i : i + batch_size], base_calc, batch_size
                )
            )
        return results

    from mace import data as mace_data
    from mace.tools import torch_geometric

    # Build dataset — mirrors _atoms_to_batch but for N structures.
    arrays_keys = {**base_calc.arrays_keys, base_calc.charges_key: "charges"}
    keyspec = mace_data.KeySpecification(
        info_keys=base_calc.info_keys, arrays_keys=arrays_keys
    )
    dataset = [
        mace_data.AtomicData.from_config(
            mace_data.config_from_atoms(
                atoms, key_specification=keyspec, head_name=base_calc.head
            ),
            z_table=base_calc.z_table,
            cutoff=base_calc.r_max,
            heads=base_calc.available_heads,
        )
        for atoms in atoms_list
    ]

    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader)).to(base_calc.device)

    compute_stress = base_calc.model_type in (
        "MACE",
        "EnergyDipoleMACE",
    ) and not getattr(base_calc, "use_compile", False)

    out = base_calc.models[0](
        batch.to_dict(),
        compute_stress=compute_stress,
        training=False,
    )

    unit_e = base_calc.energy_units_to_eV
    unit_f = base_calc.energy_units_to_eV / base_calc.length_units_to_A
    unit_s = base_calc.energy_units_to_eV / base_calc.length_units_to_A**3

    energies = out["energy"].detach().cpu().numpy() * unit_e  # [n]
    forces_all = out["forces"].detach().cpu().numpy() * unit_f  # [n_atoms_total, 3]

    stress_raw = out.get("stress")
    if compute_stress and stress_raw is not None:
        stresses = stress_raw.detach().cpu().numpy() * unit_s  # [n, 3, 3]
    else:
        stresses = None

    batch_idx = batch.batch.cpu().numpy()  # [n_atoms_total] -> structure index

    results: list[dict[str, Any]] = []
    for i in range(len(atoms_list)):
        mask = batch_idx == i
        item: dict[str, Any] = {
            "energy": float(energies[i]),
            "forces": forces_all[mask],
        }
        if stresses is not None:
            item["stress"] = full_3x3_to_voigt_6_stress(stresses[i])
        results.append(item)
    return results


def _relax_batch(
    structures: list[Structure],
    base_calc: Any,
    fmax: float,
    max_steps: int,
    desc: str | None = None,
    forward_batch_size: int = 64,
) -> list[tuple[Structure, dict[str, Any] | None]]:
    """Relax a list of structures using batched MACE forward passes.

    Each iteration issues **one** MACE forward pass that covers all
    still-active (not-yet-converged) structures simultaneously.  Per-structure
    L-BFGS curvature history is maintained independently.  The pre-computed
    forces are injected into each ``Atoms`` object via
    :class:`~ase.calculators.singlepoint.SinglePointCalculator` so that
    ``LBFGS.step()`` can read them without triggering a second forward pass.

    Args:
        structures: Input pymatgen structures.
        base_calc: Shared ``MACECalculator`` whose model weights are reused.
        fmax: Force convergence threshold in eV/Å (applied to generalized
            forces from :class:`~ase.filters.FrechetCellFilter`, same
            criterion as ``LBFGS.run``).
        max_steps: Maximum L-BFGS steps per structure.
        desc: Optional description string for the ``tqdm`` progress bar.
            Pass ``None`` to suppress the bar.
        forward_batch_size: Maximum structures per single MACE forward pass.
            Passed through to :func:`_compute_batch_results` to prevent GPU OOM.

    Returns:
        List of ``(relaxed_structure, info_dict)`` pairs.  ``info_dict``
        keys match :func:`_relax_one`.
    """
    if not structures:
        return []

    n = len(structures)
    atoms_list: list[Any] = [AseAtomsAdaptor.get_atoms(s) for s in structures]
    filters = [FrechetCellFilter(a) for a in atoms_list]
    optimizers = [LBFGS(cast(Any, f), logfile=None) for f in filters]

    infos: list[dict[str, Any] | None] = [{} for _ in range(n)]
    final_structures: list[Structure] = list(structures)
    active = list(range(n))

    step_iter: Any = (
        tqdm(range(max_steps + 1), desc=desc, leave=True)
        if desc
        else range(max_steps + 1)
    )

    for step in step_iter:
        if not active:
            break

        active_atoms = [atoms_list[i] for i in active]

        try:
            batch_res = _compute_batch_results(
                active_atoms, base_calc, forward_batch_size
            )
        except Exception as exc:
            import sys
            import traceback

            print(
                f"[_relax_batch] fatal error at step {step}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            for i in active:
                infos[i] = None
            break

        # Inject pre-computed results so that LBFGS.step() can call
        # filter.get_forces() without issuing another MACE forward pass.
        for local_idx, global_idx in enumerate(active):
            res = batch_res[local_idx]
            atoms_list[global_idx].calc = SinglePointCalculator(
                atoms_list[global_idx],
                energy=res["energy"],
                forces=res["forces"],
                stress=res.get("stress"),
            )

        if step == 0:
            for local_idx, global_idx in enumerate(active):
                n_atoms = len(atoms_list[global_idx])
                infos[global_idx]["energy_per_atom_eV_initial"] = (  # type: ignore[index]
                    batch_res[local_idx]["energy"] / n_atoms
                )

        newly_converged: list[int] = []

        for local_idx, global_idx in enumerate(active):
            try:
                gen_forces = filters[global_idx].get_forces()  # (N_dof, 3)
            except Exception:
                newly_converged.append(global_idx)
                continue

            # Convergence uses generalized forces (cell + atomic), matching LBFGS.run.
            max_gen_force = float(np.linalg.norm(gen_forces, axis=1).max())
            # max_force_eV_per_A reports atomic-only forces, consistent with _relax_one.
            atomic_forces = batch_res[local_idx]["forces"]
            infos[global_idx]["max_force_eV_per_A"] = float(  # type: ignore[index]
                np.linalg.norm(atomic_forces, axis=1).max()
            )

            if max_gen_force <= fmax or step == max_steps:
                infos[global_idx]["converged"] = max_gen_force <= fmax  # type: ignore[index]
                infos[global_idx]["n_steps"] = step  # type: ignore[index]
                n_atoms = len(atoms_list[global_idx])
                infos[global_idx]["energy_per_atom_eV"] = (  # type: ignore[index]
                    batch_res[local_idx]["energy"] / n_atoms
                )
                final_structures[global_idx] = AseAtomsAdaptor.get_structure(
                    atoms_list[global_idx]
                )
                newly_converged.append(global_idx)
                continue

            # One L-BFGS step; reads forces from the SinglePointCalculator set above.
            optimizers[global_idx].step()

        active = [idx for idx in active if idx not in set(newly_converged)]

    return [(final_structures[i], infos[i]) for i in range(n)]


def _gpu_worker(
    args: tuple[list[Structure], str, float, int, int],
) -> list[tuple[Structure, dict[str, Any] | None]]:
    """Worker function run in a child process for one GPU.

    Args:
        args: Tuple of ``(structures, device, fmax, max_steps, forward_batch_size)``.

    Returns:
        List of ``(relaxed_structure, info_dict)`` pairs, one per input structure.
    """
    structures, device, fmax, max_steps, forward_batch_size = args
    base_calc = mace_mp(model=MACE_MODEL, default_dtype="float64", device=device)
    results = _relax_batch(
        structures,
        base_calc,
        fmax,
        max_steps,
        desc=device,
        forward_batch_size=forward_batch_size,
    )
    del base_calc
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return results


def relax_structures(
    structures: list[Structure],
    device: str = "cuda",
    fmax: float = 1e-3,
    max_steps: int = 1000,
    forward_batch_size: int = 64,
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
        forward_batch_size: Maximum structures per single MACE forward pass.
            Reduce if GPU OOM occurs (default 64 works on an A100-64GB).

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
            (chunk, f"cuda:{i}", fmax, max_steps, forward_batch_size)
            for i, chunk in enumerate(chunks)
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
    base_calc = mace_mp(model=MACE_MODEL, default_dtype="float64", device=device)
    results = _relax_batch(
        structures,
        base_calc,
        fmax,
        max_steps,
        desc="Relaxing",
        forward_batch_size=forward_batch_size,
    )

    del base_calc
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    if not results:
        return [], []
    relaxed_structures, infos = map(list, zip(*results))
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
